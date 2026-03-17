// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_compressed_generate_opt.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

// Sub-group / vectorisation constants — must match gemm_generate_opt.cl definitions.
static constexpr int SG_SIZE  = 16;   // Intel GPU sub-group width for f16
static constexpr int VEC_SIZE = 8;    // half8 / u4-nibble-vec per lane
static constexpr int GROUP_SIZE = SG_SIZE * VEC_SIZE;  // 128 → matches typical per-channel group size

// -----------------------------------------------------------------------
// Kernel generator
// Reuses the "gemm_generate_opt" .cl template, but emits `IS_WEIGHT_INT4=1`
// together with WOQ-specific constants so the `#if IS_WEIGHT_INT4` branch
// inside the template gets compiled.
// -----------------------------------------------------------------------
class FCCompressedOptGenerator : public KernelGenerator {
public:
    FCCompressedOptGenerator() : KernelGenerator("gemm_generate_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        // After update_impl_params, input_layouts are reordered to:
        //   [0] activation (f16)  [1] weight (u4/i4)  [2] scale (f16)  [3] ZP(opt, u4)
        const auto& in0  = params.input_layouts[0];  // activation
        const auto& in1  = params.input_layouts[1];  // weight

        const auto& shape_a = in0.get_shape();
        const auto& shape_w = in1.get_shape();        // [N, K] after reshape

        // Derive tensor dimensions from (updated) activation and weight layouts.
        const size_t rank = shape_a.size();
        const size_t K    = shape_a[rank - 1];        // reduction dimension
        const size_t N    = shape_w[0];               // weight rows = output features
        // Batch: product of all leading dims of activation above K.
        size_t B = 1;
        for (size_t i = 0; i + 1 < rank; ++i)
            B *= shape_a[i];

        const size_t num_groups = K / GROUP_SIZE;

        // Determine u4 vs i4.
        const bool weight_is_signed = (in1.data_type == data_types::i4);

        // Detect optional ZP — it is present when num_inputs > 2 and the 4th layout
        // exists and its dtype is u4/i4.
        const bool has_zp = (params.input_layouts.size() > 3) &&
                            (params.input_layouts[3].data_type == data_types::u4 ||
                             params.input_layouts[3].data_type == data_types::i4);

        // Dispatch/size constants (mirrors GemmGenerateOptGenerator).
        jit.add({
            make_jit_constant("K_SIZE",    static_cast<int>(K)),
            make_jit_constant("N_SIZE",    static_cast<int>(N)),
            make_jit_constant("B_SIZE",    static_cast<int>(B)),
            make_jit_constant("SG_SIZE",   SG_SIZE),
            make_jit_constant("VEC_SIZE",  VEC_SIZE),
            make_jit_constant("WG_TILES",  SG_SIZE),
        });

        // WOQ-specific constants.
        jit.add({
            make_jit_constant("IS_WEIGHT_INT4",   1),
            make_jit_constant("WEIGHT_IS_SIGNED", weight_is_signed ? 1 : 0),
            make_jit_constant("HAS_ZP",           has_zp ? 1 : 0),
            make_jit_constant("GROUP_SIZE",       GROUP_SIZE),
            make_jit_constant("NUM_GROUPS",       static_cast<int>(num_groups)),
        });

        // Float32 accumulator for numerical stability.
        jit.add(make_type_jit_constants("ACCUMULATOR", data_types::f32));

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            assert(!params.is_dynamic());

            const auto& in0   = params.input_layouts[0];
            const auto& in1   = params.input_layouts[1];
            const auto& shape_a = in0.get_shape();

            const size_t rank  = shape_a.size();
            const size_t N     = in1.get_shape()[0];  // weight [N, K]
            size_t B = 1;
            for (size_t i = 0; i + 1 < rank; ++i)
                B *= shape_a[i];

            const size_t n_groups = (N + SG_SIZE - 1) / SG_SIZE;

            auto& wgs = kd.params.workGroups;
            wgs.global = {n_groups * SG_SIZE, B, 1};
            wgs.local  = {static_cast<size_t>(SG_SIZE), 1, 1};
        }};
    }
};

// -----------------------------------------------------------------------
class FCCompressedOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FCCompressedOptImpl)

    Stage::Ptr gemv_stage = make_stage<FCCompressedOptGenerator>();

    FCCompressedOptImpl() : PrimitiveImplOCL(FCCompressedGenerateOpt::get_type_info_static()) {}
    FCCompressedOptImpl(const program_node& node, const RuntimeParams& params) : FCCompressedOptImpl() {
        add_stage(gemv_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<FCCompressedOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> FCCompressedGenerateOpt::create_impl(const program_node& node,
                                                                      const RuntimeParams& params) const {
    assert(node.is_type<fully_connected>());
    return std::make_unique<FCCompressedOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FCCompressedOptImpl)
