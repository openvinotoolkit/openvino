// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_woq_u2.hpp"

#include "common_utils/kernel_generator_base.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {

// Kernel work decomposition (must match woq_u2_gemm_dual.cm defaults NCOL=2, MROWS=32, WG_M=WG_N=8):
// one 8 x 8 work-group computes a 256 x 256 tile, each thread 32 rows x 32 columns.
constexpr size_t wg_n = 8;
constexpr size_t wg_m = 8;
constexpr size_t cols_per_thread = 32;
constexpr size_t rows_per_thread = 32;

// Dependencies of a compressed FC without bias: 0 = input, 1 = weights, 2 = scale, 3 = zero point.
constexpr uint32_t scale_idx = 2;
constexpr uint32_t zp_idx = 3;

// Index into the FC's dependencies of the SwiGLU multiply's external operand (up_proj's output), or -1
// if this FC has no fused epilogue. validate_impl (FullyConnectedWoqU2ImplementationManager::
// is_supported_swiglu_epilogue) guarantees at most this one fused pattern reaches create_impl.
int32_t epi_dep_idx(const RuntimeParams& params) {
    for (const auto& fd : params.fused_desc) {
        if (fd.has_outer_dep())
            return fd.outer_dep_start_idx;
    }
    return -1;
}

class WoqU2FCGenerator : public KernelGenerator {
public:
    WoqU2FCGenerator() : KernelGenerator("woq_u2_gemm_dual") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -Qxcm_register_file_size=128 ";
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto epi_idx = epi_dep_idx(params);
        const bool epi_f16 = epi_idx >= 0 && params.get_input_layout(static_cast<size_t>(epi_idx)).data_type == data_types::f16;
        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            // N-major unless a test selected the group-major path (see fully_connected_woq_u2.hpp).
            make_jit_constant("WLAYOUT", static_cast<int>(woq_u2_weight_layout_for_tests())),
            make_jit_constant("OUT_F16", params.get_output_layout(0).data_type == data_types::f16 ? 1 : 0),
            make_jit_constant("EPI_F16", epi_f16 ? 1 : 0),
        });
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        // Kernel ABI: (A, Wq, scales, zps, epi, C, M, K, N, epi_mode). Plain FC: epi_mode = 0, epi bound
        // to a harmless placeholder (the kernel never reads it then). gate_proj-style SwiGLU epilogue
        // (see FullyConnectedWoqU2ImplementationManager::is_supported_swiglu_epilogue): epi_mode = 2,
        // epi bound to the fused multiply's external operand (up_proj's output).
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});          // A
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});          // Wq
        args.push_back({ArgumentDescriptor::Types::INPUT, scale_idx});  // scales
        args.push_back({ArgumentDescriptor::Types::INPUT, zp_idx});     // zero points (u8)
        const auto epi_idx = epi_dep_idx(params);
        if (epi_idx >= 0)
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(epi_idx)});  // epi
        else
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});     // epi (unused, epi_mode = 0)
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});         // C
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});         // M
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});         // K
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});         // N
        args.push_back({ArgumentDescriptor::Types::SCALAR, 3});         // epi_mode
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const auto& wshape = params.get_input_layout(1).get_shape();  // [N, K]
            const auto& in_shape = params.get_input_layout(0).get_shape();
            const size_t N = wshape[0];
            const size_t K = wshape[1];
            size_t M = 1;
            for (size_t i = 0; i + 1 < in_shape.size(); i++)
                M *= in_shape[i];

            auto& wgs = kd.params.workGroups;
            const size_t tiles_n = (N / cols_per_thread + wg_n - 1) / wg_n;
            const size_t tiles_m = ((M + rows_per_thread - 1) / rows_per_thread + wg_m - 1) / wg_m;
            wgs.global = {tiles_n * wg_n, std::max<size_t>(tiles_m, 1) * wg_m, 1};
            wgs.local = {wg_n, wg_m, 1};

            auto& scalars = kd.params.scalars;
            scalars.resize(4);
            // epi_mode 2: gate_proj-style SwiGLU epilogue (swish(acc) * epi); 0: no epilogue. validate_impl
            // only allows that one fused pattern through, so has_fused_primitives() alone decides it here.
            const int32_t epi_mode = params.has_fused_primitives() ? 2 : 0;
            const size_t vals[4] = {M, K, N, static_cast<size_t>(epi_mode)};
            for (size_t i = 0; i < 4; i++) {
                scalars[i].t = ScalarDescriptor::Types::INT32;
                scalars[i].v.s32 = static_cast<int32_t>(vals[i]);
            }
        }};
    }
};

class WoqU2FCImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::WoqU2FCImpl)

    Stage::Ptr fc = make_stage<WoqU2FCGenerator>();

    WoqU2FCImpl() : PrimitiveImplOCL(FullyConnectedWoqU2ImplementationManager::get_type_info_static()) {}
    WoqU2FCImpl(const program_node& node, const RuntimeParams& params) : WoqU2FCImpl() {
        add_stage(fc, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<WoqU2FCImpl>(this);
    }

protected:
    // The generic base only exposes the FC's activation input; weights, scales and zero points are the
    // node's remaining dependencies. Bind all of them in dependency order so INPUT i == dependency i.
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data args = PrimitiveImplCM::get_arguments(instance);
        args.inputs.clear();
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            args.inputs.push_back(instance.dep_memory_ptr(i));
        return args;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> FullyConnectedWoqU2ImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<fully_connected>());
    return std::make_unique<WoqU2FCImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::WoqU2FCImpl)
