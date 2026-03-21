// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_generate_opt.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "gemm_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

// -----------------------------------------------------------------------
// Sub-group width and vectorisation factor
// Each sub-group lane reads VEC elements of K per step, so the inner loop
// body processes SG_SIZE * VEC = 16 * 8 = 128 elements of K per iteration.
// K must be divisible by 128 (enforced in support_shapes).
// -----------------------------------------------------------------------
static constexpr int SG_SIZE  = 16;  // Intel GPU natural sub-group size for f16
static constexpr int VEC_SIZE = 8;   // half8 per sub-group lane per step

// Each work-group computes WG_TILES output values along N.
// A work-group of SG_SIZE lanes produces SG_SIZE output values per iteration.
static constexpr int WG_TILES = SG_SIZE;  // = 16 output elements per work-group

class GemmGenerateOptGenerator : public KernelGenerator {
public:
    GemmGenerateOptGenerator() : KernelGenerator("gemm_generate_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto& in0  = params.input_layouts[0];
        const auto& in1  = params.input_layouts[1];
        const auto& desc = params.typed_desc<gemm>();

        const auto& shape_a = in0.get_shape();
        const auto& shape_b = in1.get_shape();

        // shape_a: [..., M, K], M == 1 in generate phase
        // shape_b: [..., N, K] (transpose_input1 stored N-first)
        const size_t rank = shape_a.size();
        const size_t K    = shape_a[rank - 1];
        const size_t N    = (shape_b.size() >= 2) ? shape_b[shape_b.size() - 2] : 1;
        // Batch dimension: product of all leading dims
        size_t B = 1;
        for (size_t i = 0; i + 2 < rank; ++i)
            B *= shape_a[i];

        jit.add({
            make_jit_constant("K_SIZE",   static_cast<int>(K)),
            make_jit_constant("N_SIZE",   static_cast<int>(N)),
            make_jit_constant("B_SIZE",   static_cast<int>(B)),
            make_jit_constant("SG_SIZE",  SG_SIZE),
            make_jit_constant("VEC_SIZE", VEC_SIZE),
            make_jit_constant("WG_TILES", WG_TILES),
        });

        // Float32 accumulator for numerical stability with f16 inputs.
        jit.add(make_type_jit_constants("ACCUMULATOR", data_types::f32));

        (void)desc;  // descriptor used in validate_impl; not needed for JIT here
        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            assert(!params.is_dynamic());

            const auto& in0   = params.input_layouts[0];
            const auto& in1   = params.input_layouts[1];
            const auto& shape_a = in0.get_shape();
            const auto& shape_b = in1.get_shape();

            const size_t rank  = shape_a.size();
            const size_t K     = shape_a[rank - 1];
            const size_t N     = (shape_b.size() >= 2) ? shape_b[shape_b.size() - 2] : 1;
            size_t B = 1;
            for (size_t i = 0; i + 2 < rank; ++i)
                B *= shape_a[i];

            // One work-group (= one sub-group) per output element along N.
            // Each sub-group cooperatively reduces over K for coalesced memory access.
            auto& wgs = kd.params.workGroups;
            wgs.global = {N * SG_SIZE, B, 1};
            wgs.local  = {static_cast<size_t>(SG_SIZE), 1, 1};

            (void)K;
        }};
    }
};

// -----------------------------------------------------------------------
class GemmGenerateOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GemmGenerateOptImpl)

    Stage::Ptr gemm_stage = make_stage<GemmGenerateOptGenerator>();

    GemmGenerateOptImpl() : PrimitiveImplOCL(GemmGenerateOpt::get_type_info_static()) {}
    GemmGenerateOptImpl(const program_node& node, const RuntimeParams& params) : GemmGenerateOptImpl() {
        add_stage(gemm_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GemmGenerateOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GemmGenerateOpt::create_impl(const program_node& node,
                                                              const RuntimeParams& params) const {
    assert(node.is_type<gemm>());
    return std::make_unique<GemmGenerateOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GemmGenerateOptImpl)
