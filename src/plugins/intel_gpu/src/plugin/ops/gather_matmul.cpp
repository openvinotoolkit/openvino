// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/gather_matmul.hpp"

#include <cstring>

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"

namespace ov::intel_gpu {
using namespace cldnn;

static void CreateGatherMatmulOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatherMatmul>& op) {
    auto inputs = p.GetInputInfo(op);
    // Inputs: A, B (transposed), indices, [bias]
    validate_inputs_count(op, {3, 4});

    const auto& a_shape = op->get_input_partial_shape(0);
    OPENVINO_ASSERT(a_shape.rank().is_static() && a_shape.size() >= 1,
                    "GatherMatmul requires static input A rank, got shape ", a_shape);
    // n_activated_experts is overridden at runtime from the actual input shape
    // (see GatherMatmulOCLImpl::update_rt_params); pass 0 when dim[0] is dynamic.
    const int32_t n_activated_experts =
        a_shape[0].is_static() ? static_cast<int32_t>(a_shape[0].get_length()) : 0;

    // Placeholder inputs have shape_size <= 1.
    const bool has_bias = op->get_input_size() == 4 && ov::shape_size(op->get_input_shape(3)) > 1;

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::gather_matmul bgm(layerName, inputs, has_bias, /*has_zp=*/false, n_activated_experts);

    p.add_primitive(*op, bgm);
}

// t1/u1 expert scales given as FP8 E4M3 constants go to the device as bytes (one byte per group instead of two).
// Returns true when inputs[4] now points at a u8 data primitive in the scales' [E, N, G] order.
static bool set_fp8_scales(ProgramBuilder& p,
                           const std::shared_ptr<ov::op::internal::GatherMatmulCompressed>& op,
                           std::vector<cldnn::input_info>& inputs,
                           const std::string& layerName) {
    const auto scales = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(4));
    if (!scales || scales->get_element_type() != ov::element::f8e4m3) {
        return false;
    }
    const size_t count = ov::shape_size(scales->get_shape());
    std::vector<int64_t> dims;
    for (auto d : scales->get_shape()) {
        dims.push_back(static_cast<int64_t>(d));
    }
    const cldnn::layout fp8_layout(ov::PartialShape(dims), cldnn::data_types::u8, cldnn::format::get_default_format(dims.size()));
    auto mem = p.get_engine().allocate_memory(fp8_layout, false);
    {
        auto& stream = p.get_engine().get_service_stream();
        cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::write> lock{mem, stream};
        std::memcpy(lock.data(), scales->get_data_ptr(), count);
    }
    const std::string id = layerName + "_fp8_scales";
    p.add_primitive(*op, cldnn::data(id, mem));
    inputs[4] = cldnn::input_info(id);
    return true;
}

static void CreateGatherMatmulCompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatherMatmulCompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    // Inputs: A, B, indices, bias, scales, zp (placeholder when absent).
    validate_inputs_count(op, {6});

    const auto& a_shape = op->get_input_partial_shape(0);
    OPENVINO_ASSERT(a_shape.rank().is_static() && a_shape.size() >= 1 && a_shape[0].is_static(),
                    "GatherMatmulCompressed requires static n_activated_experts (input A dim[0]), got shape ",
                    a_shape);
    const int32_t n_activated_experts = static_cast<int32_t>(a_shape[0].get_length());

    // Placeholder inputs have shape_size <= 1.
    const bool has_bias = ov::shape_size(op->get_input_shape(3)) > 1;
    const bool has_zp = ov::shape_size(op->get_input_shape(5)) > 1;

    const std::string layerName = layer_type_name_ID(op);

    // Base-3 ternary experts, marked by ExpandGatherMatmulScalarU2ZeroPoint: repack the u2 constant (codes {0, 1, 2},
    // zero point 1) into t1 bytes here, so the program and the model cache hold only 1.625 bits per weight.
    if (op->get_rt_info().count("gpu_gathermatmul_t1")) {
        const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(weights && weights->get_element_type() == ov::element::u2, "GatherMatmul t1: weights must be a u2 constant");
        const auto& w_shape = weights->get_shape();
        OPENVINO_ASSERT(w_shape.size() == 3 && w_shape[2] % 64 == 0, "GatherMatmul t1: unsupported weights shape ", w_shape);
        const size_t E = w_shape[0], N = w_shape[1], K = w_shape[2], KB = K / 64;
        const auto* src = static_cast<const uint8_t*>(weights->get_data_ptr());
        auto code = [&](size_t idx) -> uint32_t {  // u2, LSB-first
            return (src[idx >> 2] >> ((idx & 3) * 2)) & 0x3u;
        };
        const cldnn::layout t1_layout(ov::PartialShape{static_cast<int64_t>(E), static_cast<int64_t>(N), static_cast<int64_t>(KB * 13)},
                                      cldnn::data_types::u8,
                                      cldnn::format::bfyx);
        auto mem = p.get_engine().allocate_memory(t1_layout, false);
        {
            auto& stream = p.get_engine().get_service_stream();
            cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::write> lock{mem, stream};
            uint8_t* dst = lock.data();
            ov::parallel_for(E * N, [&](size_t row) {
                const size_t base = row * K;
                uint8_t* out = dst + row * KB * 13;
                for (size_t b = 0; b < KB; ++b) {
                    for (size_t j = 0; j < 13; ++j) {
                        uint32_t v = 0;
                        for (size_t k = 0; k < 5; ++k) {
                            const size_t i = 5 * j + k;  // value within the 64-block; 64 is padding
                            const uint32_t t = i < 64 ? code(base + b * 64 + i) : 1u;
                            OPENVINO_ASSERT(t <= 2, "GatherMatmul t1: weights are not ternary");
                            v = v * 3 + t;
                        }
                        out[b * 13 + j] = static_cast<uint8_t>((v * 256 + 242) / 243);
                    }
                }
            });
        }
        const std::string t1_id = layerName + "_t1_weights";
        p.add_primitive(*op, cldnn::data(t1_id, mem));
        inputs[1] = cldnn::input_info(t1_id);
        const bool fp8 = set_fp8_scales(p, op, inputs, layerName);
        cldnn::gather_matmul bgm(layerName, inputs, has_bias, /*has_zp=*/false, n_activated_experts);
        bgm.scales_fp8 = fp8;
        bgm.weights_t1 = true;
        bgm.t1_k = static_cast<int64_t>(K);
        p.add_primitive(*op, bgm);
        return;
    }

    // Binary experts, marked by ExpandGatherMatmulScalarZeroPoint: the u1 bits (MSB first) go as raw bytes.
    if (op->get_rt_info().count("gpu_gathermatmul_u1")) {
        const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        // the scalar zero point may still sit behind the Convert that ConvertPrecision left on it
        const auto zp = ov::util::get_constant_from_source(op->input_value(5));
        OPENVINO_ASSERT(weights && weights->get_element_type() == ov::element::u1 && zp, "GatherMatmul u1: unexpected inputs");
        const auto& w_shape = weights->get_shape();
        OPENVINO_ASSERT(w_shape.size() == 3 && w_shape[2] % 64 == 0, "GatherMatmul u1: unsupported weights shape ", w_shape);
        const size_t K = w_shape[2];
        // The u1 constant is already a u1 data primitive holding the same MSB-first bytes (inputs[1]): use it as is, so
        // the device does not hold a second copy until the end of the build.
        const bool fp8 = set_fp8_scales(p, op, inputs, layerName);
        cldnn::gather_matmul bgm(layerName, inputs, has_bias, /*has_zp=*/false, n_activated_experts);
        bgm.scales_fp8 = fp8;
        bgm.weights_u1 = true;
        bgm.t1_k = static_cast<int64_t>(K);
        bgm.u1_zp = zp->cast_vector<float>()[0];
        p.add_primitive(*op, bgm);
        return;
    }

    const cldnn::gather_matmul bgm(layerName, inputs, has_bias, has_zp, n_activated_experts);

    p.add_primitive(*op, bgm);
}

REGISTER_FACTORY_IMPL(internal, GatherMatmul);
REGISTER_FACTORY_IMPL(internal, GatherMatmulCompressed);

}  // namespace ov::intel_gpu
