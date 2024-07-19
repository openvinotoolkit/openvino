// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "impls/registry/implementation_manager.hpp"
#include "utils.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct PoolingImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("PoolingImplementationOnednn")
    PoolingImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<pooling>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad)
            return false;

        if (!is_supported_format(node.get_preferred_input_fmt(0)))
            return false;

        const auto& in_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        auto in_dt = in_layout.data_type;
        auto out_dt = out_layout.data_type;

        static const std::vector<format::type> supported_formats = {
            format::bfyx,
            format::b_fs_yx_fsv16,
            format::b_fs_zyx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv32,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
        };

        bool fp_case = data_type_traits::is_floating_point(in_dt) && in_dt == out_dt;
        bool u8s8_case = one_of(in_dt, {data_types::i8, data_types::u8}) && one_of(out_dt, {data_types::i8, data_types::u8});

        if (!fp_case && !u8s8_case)
            return false;

        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats))
            return false;

        if (!is_supported_post_ops(node))
            return false;

        return ImplementationManager::validate(node);
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    bool support_shapes(const kernel_impl_params& params) const override {
        return true;
    }
};

}  // namespace onednn
}  // namespace cldnn
