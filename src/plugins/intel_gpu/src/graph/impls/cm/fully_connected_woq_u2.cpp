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

// Tiled kernel work decomposition (must match woq_u2_gemm_dual.cm defaults NCOL=2, MROWS=32,
// WG_M=WG_N=8): one 8 x 8 work-group computes a 256 x 256 tile, each thread 32 rows x 32 columns.
constexpr size_t wg_n = 8;
constexpr size_t wg_m = 8;
constexpr size_t cols_per_thread = 32;
constexpr size_t rows_per_thread = 32;

// GEMV kernel (woq_u2_gemm_dual_gemv.cm, built with NCB=1, RBG=1, LS=16, PFG=4): each thread computes
// 16 columns x 8 rows over all of K, LS threads per work-group, no SLM. Used when M <= gemv_max_m.
constexpr size_t gemv_cols_per_thread = 16;  // 16 * NCB
constexpr size_t gemv_rows_per_thread = 8;   // 8 * RBG
constexpr size_t gemv_ls = 16;               // LS
constexpr size_t gemv_max_m = 8;
// Above this many 16-column blocks each GEMV thread takes 2 adjacent blocks (cseq = 2); passed to the kernel
// as GEMV_MAXT, so the grid below and the kernel's own choice always agree.
constexpr size_t gemv_maxt = 960;

// M = product of the activation's leading dims (static params only).
size_t rows_of(const RuntimeParams& params) {
    const auto& in_shape = params.get_input_layout(0).get_shape();
    size_t M = 1;
    for (size_t i = 0; i + 1 < in_shape.size(); i++)
        M *= in_shape[i];
    return M;
}

bool has_bias(const RuntimeParams& params) {
    return params.bias_layout.has_value();
}

// The epilogue (epi_mode and the dependency providing its tensor) of this FC; validate_impl guarantees it
// is supported (create_impl asserts it).
WoqU2Epilogue epilogue_of(const RuntimeParams& params) {
    return woq_u2_epilogue(params.fused_desc, has_bias(params));
}

// Shared by both kernels: same ABI, JIT constants and scalars; they differ in source, build options and
// launch grid.
class WoqU2GeneratorBase : public KernelGenerator {
public:
    using KernelGenerator::KernelGenerator;

protected:

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto epi = epilogue_of(params);
        const bool epi_f16 = epi.dep >= 0 && params.get_input_layout(static_cast<size_t>(epi.dep)).data_type == data_types::f16;
        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            // N-major unless a test selected the group-major path (see fully_connected_woq_u2.hpp).
            make_jit_constant("WLAYOUT", static_cast<int>(woq_u2_weight_layout_for_tests())),
            make_jit_constant("OUT_F16", params.get_output_layout(0).data_type == data_types::f16 ? 1 : 0),
            make_jit_constant("EPI_F16", epi_f16 ? 1 : 0),      // epilogue tensor is half
        });
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        // Kernel ABI: (A, Wq, scales, zps, epi, C, M, K, N, epi_mode). epi is the epilogue tensor (FC bias,
        // fused add operand or SwiGLU multiply operand, see woq_u2_epilogue); without an epilogue
        // (epi_mode = 0) the kernel never reads it and the output buffer is bound in its place.
        const bool bias = has_bias(params);
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});                                            // A
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});                                            // Wq
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(woq_u2_scale_idx(bias))});  // scales
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(woq_u2_zp_idx(bias))});     // zero points (u8)
        const auto epi = epilogue_of(params);
        if (epi.dep >= 0)
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(epi.dep)});  // epi
        else
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});     // epi (unused, epi_mode = 0)
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});         // C
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});         // M
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});         // K
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});         // N
        args.push_back({ArgumentDescriptor::Types::SCALAR, 3});         // epi_mode
        return args;
    }

    // Scalars (M, K, N, epi_mode) of either kernel.
    static void set_scalars(const RuntimeParams& params, KernelData& kd, size_t M) {
        const auto& wshape = params.get_input_layout(1).get_shape();  // [N, K]
        auto& scalars = kd.params.scalars;
        scalars.resize(4);
        // epi_mode from the actual bias / fused pattern: 0 none, 1 acc + epi, 2 swish(acc) * epi.
        const int32_t epi_mode = epilogue_of(params).mode;
        const size_t vals[4] = {M, wshape[1], wshape[0], static_cast<size_t>(epi_mode)};
        for (size_t i = 0; i < 4; i++) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(vals[i]);
        }
    }
};

// Tiled DPAS kernel (woq_u2_gemm_dual.cm): M > gemv_max_m.
class WoqU2TiledGenerator : public WoqU2GeneratorBase {
public:
    WoqU2TiledGenerator() : WoqU2GeneratorBase("woq_u2_gemm_dual") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -Qxcm_register_file_size=128 ";
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const size_t N = params.get_input_layout(1).get_shape()[0];
            const size_t M = rows_of(params);
            auto& wgs = kd.params.workGroups;
            const size_t tiles_n = (N / cols_per_thread + wg_n - 1) / wg_n;
            const size_t tiles_m = ((M + rows_per_thread - 1) / rows_per_thread + wg_m - 1) / wg_m;
            wgs.global = {tiles_n * wg_n, std::max<size_t>(tiles_m, 1) * wg_m, 1};
            wgs.local = {wg_n, wg_m, 1};
            set_scalars(params, kd, M);
        }};
    }
};

// GEMV kernel (woq_u2_gemm_dual_gemv.cm): M <= gemv_max_m (decode).
// blocks = ceil(N / 16), cseq = 2 if blocks > gemv_maxt else 1 (column blocks per thread),
// global = (ceil(blocks / cseq / 16) * 16, ceil(M / 8)), local = (16, 1).
class WoqU2GemvGenerator : public WoqU2GeneratorBase {
public:
    WoqU2GemvGenerator() : WoqU2GeneratorBase("woq_u2_gemm_dual_gemv") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -Qxcm_register_file_size=128 -DNCB=1 -DRBG=1 -DLS=16 -DPFG=4 -DGEMV_MAXT=" +
               std::to_string(gemv_maxt) + " ";
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const size_t N = params.get_input_layout(1).get_shape()[0];
            const size_t M = rows_of(params);
            auto& wgs = kd.params.workGroups;
            const size_t blocks = (N + gemv_cols_per_thread - 1) / gemv_cols_per_thread;
            const size_t cseq = blocks > gemv_maxt ? 2 : 1;
            const size_t groups_n = ((blocks + cseq - 1) / cseq + gemv_ls - 1) / gemv_ls;
            const size_t groups_m = (M + gemv_rows_per_thread - 1) / gemv_rows_per_thread;
            wgs.global = {groups_n * gemv_ls, std::max<size_t>(groups_m, 1), 1};
            wgs.local = {gemv_ls, 1, 1};
            set_scalars(params, kd, M);
        }};
    }
};

class WoqU2FCImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::WoqU2FCImpl)

    // Stage indices are their registration order: 0 = tiled, 1 = GEMV.
    Stage::Ptr tiled = make_stage<WoqU2TiledGenerator>();
    Stage::Ptr gemv = make_stage<WoqU2GemvGenerator>();

    WoqU2FCImpl() : PrimitiveImplOCL(FullyConnectedWoqU2ImplementationManager::get_type_info_static()) {}
    WoqU2FCImpl(const program_node& node, const RuntimeParams& params) : WoqU2FCImpl() {
        const auto epi = epilogue_of(params);
        OPENVINO_ASSERT(epi.ok(), "[GPU] cm::fully_connected::woq_u2 created for ", node.id(), " with an unsupported epilogue: ", epi.error);
        add_stage(tiled, params);
        add_stage(gemv, params);
    }

    // One kernel per execution, chosen by the runtime M (M may be dynamic): GEMV for M <= gemv_max_m.
    [[nodiscard]] std::vector<size_t> get_stages_execution_order(const cldnn::kernel_impl_params& impl_params) const override {
        if (impl_params.is_dynamic())
            return _order;
        return {rows_of(impl_params) <= gemv_max_m ? size_t(1) : size_t(0)};
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
