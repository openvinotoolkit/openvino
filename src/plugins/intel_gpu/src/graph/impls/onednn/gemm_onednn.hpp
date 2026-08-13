// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_inst.h"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct GemmImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::gemm")
    GemmImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<gemm>());
        const auto& config = node.get_program().get_config();
        if (!config.get_use_onednn())
            return false;
#ifdef OV_GPU_VENDOR_INTEL
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown)
            return false;
#endif

        const auto& gemm_node = node.as<gemm>();
        const auto& gemm_prim = gemm_node.get_primitive();
        const auto& in0_layout = node.get_input_layout(0);
        const auto& in1_layout = node.get_input_layout(1);
        const auto& out_layout = node.get_output_layout(0);

        // oneDNN GEMM requires at least 2D tensors with a known (static) rank
        // Check actual shape rank, not format rank
        auto has_rank_less_than = [](const ov::PartialShape& ps, int64_t min_rank) {
            const auto& r = ps.rank();
            return r.is_dynamic() || r.get_length() < min_rank;
        };

        const auto& in0_shape = in0_layout.get_partial_shape();
        const auto& in1_shape = in1_layout.get_partial_shape();
        const auto& out_shape = out_layout.get_partial_shape();
        if (has_rank_less_than(in0_shape, 2) ||
            has_rank_less_than(in1_shape, 2) ||
            has_rank_less_than(out_shape, 2))
            return false;

        auto in0_dt = in0_layout.data_type;
        auto in1_dt = in1_layout.data_type;
        auto out_dt = out_layout.data_type;

        static const std::vector<format::type> supported_formats = {
            format::any,
            format::bfyx,
            format::bfxy,
            format::byxf,
            format::byfx,
            format::bxfy,
            format::fybx,  //format used for gemm fusion
            format::fyxb,  //format used for gemm fusion
            format::xbfy, // format used for gemm fusion
            format::ybfx, // format used for gemm fusion
            format::bfzyx,
            format::bfwzyx,
        };

        if (gemm_prim->alpha != 1.0f || gemm_prim->beta != 0.0f)
            return false;

        if (out_layout.data_padding)
            return false;

#ifdef OV_GPU_VENDOR_INTEL
        if (one_of(in0_dt, {data_types::f32, data_types::i64}) || one_of(in1_dt, {data_types::f32, data_types::i64}))
            return false;
#else
        if (one_of(in0_dt, {data_types::i64}) || one_of(in1_dt, {data_types::i64}))
            return false;
#endif

        if (!one_of(in0_layout.format.value, supported_formats) ||
            !one_of(in1_layout.format.value, supported_formats) ||
            !one_of(out_layout.format.value, supported_formats))
            return false;

#ifdef OV_GPU_VENDOR_NVIDIA
        // cuDNN matmul doesn't support 4D/5D/6D format tags (abcde, abcdef)
        // Reject rank > 3 to fall back to OCL kernels
        if (in0_shape.rank().is_static() && in0_shape.rank().get_length() > 3)
            return false;
        if (in1_shape.rank().is_static() && in1_shape.rank().get_length() > 3)
            return false;
        if (out_shape.rank().is_static() && out_shape.rank().get_length() > 3)
            return false;

        // cuDNN matmul only supports a limited set of activation post-ops:
        // relu, tanh, elu, logistic (all with alpha=0), and at most one.
        {
            int activation_count = 0;
            for (auto& fo : gemm_node.get_fused_primitives()) {
                if (fo.is_type<activation>()) {
                    auto act_prim = fo.typed_desc<activation>();
                    auto func = act_prim->activation_function;
                    if (func == activation_func::none)
                        continue;
                    if (++activation_count > 1)
                        return false;
                    bool supported = (func == activation_func::relu ||
                                      func == activation_func::hyperbolic_tan ||
                                      func == activation_func::elu ||
                                      func == activation_func::logistic);
                    if (!supported)
                        return false;
                    if (act_prim->additional_params.a != 0.0f)
                        return false;
                } else {
                    return false;
                }
            }
        }
#endif

        //format bxfy not on whitelist for fusable_output_order_onednn in gemm_inst.h what makes assert error later
        //skipped onednn implementation as ocl is faster
        std::vector<size_t> out_transpose_order(std::begin(gemm_prim->output_transpose_order), std::end(gemm_prim->output_transpose_order));
        if (out_transpose_order == format::traits(format::bxfy)._order) {
            return false;
        }

        bool f16f16_case = everyone_is(data_types::f16, in0_dt, in1_dt) && one_of(out_dt, {data_types::f16, data_types::f32, data_types::i8});
        bool u8s8_case = one_of(in0_dt, {data_types::i8, data_types::u8}) &&
                         one_of(in1_dt, {data_types::i8, data_types::u8}) &&
                         one_of(out_dt, {data_types::f16, data_types::f32, data_types::i32, data_types::i8, data_types::u8});

#ifdef OV_GPU_VENDOR_INTEL
        if (!f16f16_case && !u8s8_case)
            return false;
#else
        bool f32f32_case = everyone_is(data_types::f32, in0_dt, in1_dt) && one_of(out_dt, {data_types::f32});
        if (!f32f32_case && !f16f16_case && !u8s8_case)
            return false;
#endif

        if (gemm_prim->indirect_a || gemm_prim->indirect_b)
            return false;

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<gemm>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            size_t out_rank = node.get_output_layout().get_rank();
            auto target_format = format::get_default_format(out_rank);

            in_fmts[idx] = target_format;

            if (out_fmts[0] == format::any) {
                out_fmts[0] = target_format;
            }
        }

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
