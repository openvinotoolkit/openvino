// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/graph_util.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "ov_ops/moe_expert.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/moe_expert.hpp"

namespace ov::intel_gpu {

template<typename Type>
void repack(Type* dst, const Type* src, size_t N, size_t K) {
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) {
            dst[k * N + n] = src[n * K + k];
        }
    }
}

static bool repack_zp_scale(std::vector<uint8_t>& dst, const uint8_t* src, const ov::Shape& shape, const ov::element::Type type) {
    OPENVINO_ASSERT(shape.size() == 3, "repack_zp_scale expects zp/scale's rank is 3, current is ", shape.size());
    auto groups_count = shape[1];
    if (groups_count == 1)
        return false;
    auto N = shape[0];
    auto K = shape[1];
    auto element_size = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
    if (type == ov::element::u4 || type == ov::element::i4) {
        dst.resize(element_size / 2);
        for (size_t n = 0; n < N; n += 2) {
            for (size_t k = 0; k < K; k += 2) {
                auto src1 = src[n * K / 2 + k / 2];
                auto src2 = src[(n + 1) * K / 2 + k / 2];
                dst[k * N / 2 + n / 2] = ((src2 & 0xf) << 4) | (src1 & 0xf);
                dst[(k + 1) * N / 2 + n / 2] = (src2 & 0xf0) | ((src1 & 0xf0) >> 4);
            }
        }
    } else if (type == ov::element::u8 || type == ov::element::i8) {
        dst.resize(element_size);
        repack(dst.data(), src, N, K);
    } else if (type == ov::element::f16) {
        dst.resize(element_size * sizeof(uint16_t));
        repack(reinterpret_cast<uint16_t*>(dst.data()), reinterpret_cast<const uint16_t*>(src), N, K);
    } else if (type == ov::element::f32) {
        dst.resize(element_size * sizeof(float));
        repack(reinterpret_cast<float*>(dst.data()), reinterpret_cast<const float*>(src), N, K);
    } else {
        OPENVINO_ASSERT(false, "repack_zp_scale does not support type: ", type);
    }
    return true;
}

static void prepare_weights(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert2>& op, std::vector<cldnn::moe_expert::mlp_params>& params) {
    const auto& bodys = op->get_body();
    auto alloc = [&] (const std::shared_ptr<ov::Node>& node, bool try_repack = false) {
        auto op = ov::as_type_ptr<ov::op::v0::Constant>(node);
        ov::Shape const_shape = op->get_shape();
        auto constFormat = cldnn::format::get_default_format(const_shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
        auto layout = cldnn::layout(const_shape, out_dtype, constFormat);
        auto data = op->get_data_ptr<uint8_t>();
        const auto cache_key = std::make_tuple(data, const_shape, out_dtype);
        std::vector<uint8_t> repacked_buf;
        if (try_repack) {
            auto repacked = repack_zp_scale(repacked_buf, data, const_shape, op->get_output_element_type(0));
            if (repacked) {
                data = repacked_buf.data();
                auto new_shape = ov::Shape{const_shape[1], const_shape[0], 1};
                layout = cldnn::layout(new_shape, out_dtype, constFormat);
            }
        }
        auto mem = p.get_engine().allocate_memory(layout, cldnn::allocation_type::usm_device, false);
        auto& stream = p.get_engine().get_service_stream();
        // cldnn::mem_lock<uint8_t> lock{mem, stream};
        // auto buf = lock.data();
        // std::memcpy(&buf[0], &data[0], layout.bytes_count());
        mem->copy_from(stream, data, 0, 0, layout.bytes_count(), true);
        return mem;
    };
    OPENVINO_ASSERT(op->get_config().expert_num == bodys.size());
    params.resize(bodys.size());
    for (size_t i = 0; i < bodys.size(); i++) {
        auto internal_body = bodys[i];
        for (auto& node : internal_body->get_ordered_ops()) {
            auto& rt = node->get_rt_info();
            if (rt.count("__weight_const__")) {
                auto idx = rt["__weight_const__"].as<int>();
                OPENVINO_ASSERT(idx >= 0 && idx < 3);
                params[i].param[idx].weight = alloc(node);
            }
            if (rt.count("__scale_const__")) {
                auto idx = rt["__scale_const__"].as<int>();
                OPENVINO_ASSERT(idx >= 0 && idx < 3);
                params[i].param[idx].scale = alloc(node, true);
                params[i].param[idx].scale_ba = alloc(node, false);
            }
            if (rt.count("__zp_const__")) {
                auto idx = rt["__zp_const__"].as<int>();
                OPENVINO_ASSERT(idx >= 0 && idx < 3);
                params[i].param[idx].zp = alloc(node, true);
                params[i].param[idx].zp_ba = alloc(node, false);
            }
        }
    }
}

static void CreateMOEExpert2Op(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert2>& op) {
    auto inputs = p.GetInputInfo(op);
    OPENVINO_ASSERT(inputs.size() == 4, "Inputs count should be 4");

    const std::string layerName = layer_type_name_ID(op);
    std::vector<cldnn::moe_expert::mlp_params> params;
    prepare_weights(p, op, params);

    const cldnn::moe_expert moe(layerName,
                                inputs,
                                op->get_config(),
                                params
                                );

    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOEExpert2);

}  // namespace ov::intel_gpu
