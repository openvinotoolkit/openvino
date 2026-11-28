// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/utils.hpp"
#include "pooling_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils.hpp"

#include <algorithm>
#include <memory>

namespace cldnn {
namespace onednn {

struct PoolingImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::pool")
    PoolingImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::onednn, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<pooling>());
        const auto& config = node.get_program().get_config();
        if (!config.get_use_onednn()) {
            return false;
        }
#ifndef OV_GPU_VENDOR_NVIDIA
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown) {
            return false;
        }
#endif

        const auto& in_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        auto in_dt = in_layout.data_type;
        auto out_dt = out_layout.data_type;

        if (in_layout.data_padding || out_layout.data_padding) {
            return false;
        }

#ifdef OV_GPU_VENDOR_NVIDIA
        static const std::vector<format::type> supported_formats = {
            format::any,
            format::bfyx,
            format::bfzyx,
            format::byxf,
            format::bzyxf,
        };
        const bool fp_case = everyone_is(in_dt, out_dt) && one_of(in_dt, {ov::element::f32, ov::element::f16, ov::element::bf16});
        const bool u8s8_case = everyone_is(ov::element::i8, in_dt, out_dt);
#else
        static const std::vector<format::type> supported_formats = {
            format::any,
            format::bfyx,
            format::bfzyx,
            format::byxf,
            format::bzyxf,
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

        bool fp_case = one_of(in_dt, {ov::element::f16, ov::element::bf16}) && everyone_is(in_dt, out_dt);
        bool u8s8_case = one_of(in_dt, {ov::element::i8, ov::element::u8}) &&
                         one_of(out_dt, {ov::element::i8, ov::element::u8, ov::element::f32, ov::element::f16, ov::element::bf16});
#endif

        if (!fp_case && !u8s8_case) {
            return false;
        }

        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats)) {
            return false;
        }

#ifdef OV_GPU_VENDOR_NVIDIA
        const auto prim = node.as<pooling>().get_primitive();
        // Plain rank-three descriptors have one spatial axis, while pooling
        // geometry currently expands the kernel and stride to at least two.
        if (in_layout.get_partial_shape().rank() == 3 || out_layout.get_partial_shape().rank() == 3) {
            return false;
        }
        if (!prim->indices_output.pid.empty() || prim->num_outputs != 1 || get_post_ops_count(node) > 0 ||
            std::any_of(prim->dilation.begin(), prim->dilation.end(), [](size_t dilation) { return dilation > 1; })) {
            return false;
        }
        if (prim->mode == pooling_mode::average && has_unsupported_average_padding(*prim, in_layout, out_layout)) {
            return false;
        }
#endif
        return is_supported_post_ops(node);
    }

#ifdef OV_GPU_VENDOR_NVIDIA
private:
    static bool has_unsupported_average_padding(const pooling& prim, const layout& in_layout, const layout& out_layout);
#endif
};

}  // namespace onednn
}  // namespace cldnn
