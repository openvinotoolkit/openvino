// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/format.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/utils.hpp"

#include "registry/implementation_manager.hpp"

#include "utils.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct ConvolutionImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::conv")
    ConvolutionImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<convolution>());
        const auto& config = node.get_program().get_config();
        if (!config.get_use_onednn()) {
            return false;
        }

#ifdef OV_GPU_VENDOR_INTEL
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown)
            return false;
#endif

        const auto& conv_node = node.as<convolution>();

        const auto& in_layout = conv_node.get_input_layout(0);
        const auto& out_layout = conv_node.get_output_layout(0);
        const auto& wei_layout = conv_node.weights().get_output_layout(false);

        auto in_fmt = in_layout.format;
        auto out_fmt = out_layout.format;

        auto in_dt = in_layout.data_type;
        auto wei_dt = wei_layout.data_type;
        auto out_dt = out_layout.data_type;

#ifdef OV_GPU_VENDOR_NVIDIA
        // On NVIDIA the only available oneDNN forward convolutions are the generic SYCL reference kernel,
        // which requires every memory descriptor to be plain (check_convolution_formats()), and cuDNN,
        // which requires inner_nblks == 0 (memory_format_ok()). A blocked layout such as b_fs_yx_fsv16
        // makes primitive descriptor creation fail, and compile_graph has no fallback to the OCL
        // implementation. Advertise only the plain formats, following the deconvolution/pooling pattern.
        static const std::vector<format> supported_formats = {
            format::any,
            format::bfyx,
            format::bfzyx,
            format::byxf,
            format::bzyxf,
        };
#else
        static const std::vector<format> supported_formats = {
            format::any,
            format::bfyx,
            format::bfzyx,
            format::byxf,
            format::bzyxf,
            format::b_fs_yx_fsv4,
            format::b_fs_yx_fsv8,
            format::b_fs_zyx_fsv8,
            format::b_fs_yx_fsv16,
            format::b_fs_zyx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv32,
            format::bs_fs_yx_bsv4_fsv2,
            format::bs_fs_yx_bsv4_fsv4,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_zyx_bsv8_fsv2,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_zyx_bsv8_fsv4,
            format::bs_fs_yx_bsv16_fsv2,
            format::bs_fs_zyx_bsv16_fsv2,
            format::bs_fs_yx_bsv16_fsv4,
            format::bs_fs_zyx_bsv16_fsv4,
            format::bs_fs_yx_bsv16_fsv8,
            format::bs_fs_zyx_bsv16_fsv8,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bs_fs_zyx_bsv32_fsv32,
        };
#endif

        if (!one_of(in_fmt, supported_formats) || !one_of(out_fmt, supported_formats)) {
            return false;
        }

#ifdef OV_GPU_VENDOR_NVIDIA
        // NVIDIA oneDNN convolutions accept plain weights only: cuDNN's memory_format_ok() requires
        // inner_nblks == 0 (NCHW_VECT_C is s8 only) and the generic SYCL reference kernel requires every
        // memory descriptor to be plain (check_convolution_formats()). The plugin queries the weights
        // descriptor with format_tag::any, so primitive descriptor creation still succeeds and asks for a
        // plain layout (oihw/ohwi), but converting blocked weights such as os_iyx_osv16 (Abcd16a) to it
        // needs an oneDNN reorder which the NVIDIA backend does not implement. That reorder is described
        // by a format::custom output layout, so the OCL reorder cannot take it over and compile_graph has
        // no fallback, which makes the whole compilation throw (fusings_gpu/conv_swap_xy_with_eltwise_diff_sizes).
        // Let the OCL convolution handle blocked weights instead.
        if (format::is_blocked(wei_layout.format))
            return false;
#endif

        auto prim = conv_node.get_primitive();
        if (prim->groups > 1 && !prim->grouped_weights_shape) {
            return false;
        }

        if (!is_supported_pad(in_layout) || !is_supported_pad(out_layout)) {
            return false;
        }

        bool f16_conv = everyone_is(data_types::f16, in_dt, wei_dt) && one_of(out_dt, {data_types::f16, data_types::bf16, data_types::f32, data_types::u8, data_types::i8});
        bool bf16_conv = everyone_is(data_types::bf16, in_dt, wei_dt) &&
                         one_of(out_dt, {data_types::f16, data_types::bf16, data_types::f32, data_types::u8, data_types::i8});
        bool f32_conv = everyone_is(data_types::f32, in_dt, wei_dt) && one_of(out_dt, {data_types::f32, data_types::u8, data_types::i8});
        bool int8_conv = one_of(in_dt, {data_types::i8, data_types::u8}) && one_of(wei_dt, {data_types::i8, data_types::u8}) &&
                         one_of(out_dt, {data_types::i32, data_types::f16, data_types::bf16, data_types::f32, data_types::u8, data_types::i8});

        if (!f16_conv && !bf16_conv && !f32_conv && !int8_conv) {
            return false;
        }

#ifdef OV_GPU_VENDOR_NVIDIA
        // On NVIDIA the oneDNN convolutions are cuDNN and the generic SYCL reference kernel.
        // Neither takes a post-op with an extra input tensor, and a pd creation failure is fatal
        // since compile_graph has no OCL fallback. Accept only fusions that become a single
        // eltwise post-op; the mapping is in program_node::create_onednn_primitive_attributes().
        bool eltwise_post_op_used = false;
        for (const auto& fo : conv_node.get_fused_primitives()) {
            // A fused reorder is not translated into a post-op at all
            if (fo.is_type<reorder>())
                continue;

            if (fo.is_type<activation>()) {
                const auto activation_prim = fo.typed_desc<activation>();
                const auto func = activation_prim->activation_function;
                if (func == activation_func::none)
                    continue;

                // relu with a slope input becomes a prelu post-op, every other activation
                // becomes a plain eltwise post-op
                if (activation_prim->additional_params_input.is_valid())
                    return false;

                // Being an eltwise post-op is not enough: the algorithm itself must be implemented
                // by one of the NVIDIA backends. cuDNN's attr_post_ops_ok() accepts relu, tanh, elu
                // and logistic only, all with alpha == 0 and at most one eltwise entry, while the
                // generic SYCL reference kernel accepts a somewhat different set
                // (sycl_post_ops_t::eltwise_ok()) but can drop out for unrelated reasons such as its
                // problem size heuristic. Restrict to what cuDNN guarantees, like the fully_connected
                // and gemm managers already do; everything else runs on the cldnn OCL kernels.
                if (eltwise_post_op_used)
                    return false;
                if (!one_of(func, {activation_func::relu,
                                   activation_func::hyperbolic_tan,
                                   activation_func::elu,
                                   activation_func::logistic}))
                    return false;
                if (activation_prim->additional_params.a != 0.0f)
                    return false;
                eltwise_post_op_used = true;
                continue;
            }

            // A fused eltwise becomes a sum or a binary post-op. cuDNN rejects even the sum, because
            // conv appends it with an explicit dt while cuDNN's skip mask has no sum_dt, so it would
            // land on the SYCL reference kernel. Everything else may need an extra input buffer.
            return false;
        }
#endif

        if (!is_supported_post_ops(conv_node)) {
            return false;
        }

        if (prim->deformable_mode) {
            return false;
        }

        // oneDNN only supports asymmetric weights quantization by scalar zero-points
        return !conv_node.weights_zero_points_term() ||
            conv_node.weights_zero_points().get_output_layout().count() == 1;
    }

    in_out_fmts_t query_formats(const program_node& node) const override;
};

}  // namespace onednn
}  // namespace cldnn
