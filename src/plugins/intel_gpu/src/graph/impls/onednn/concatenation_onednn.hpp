// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_inst.h"
#include "impls/onednn/utils.hpp"
#include "impls/registry/implementation_manager.hpp"

#include <memory>
namespace cldnn {
namespace onednn {

struct ConcatenationImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::concat")
    ConcatenationImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::onednn, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<concatenation>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown)
            return false;

        static const std::vector<ov::element::Type_t> supported_types = { ov::element::f16, ov::element::u8, ov::element::i8 };
        static const std::vector<format::type> supported_in_fmts = {
            format::any,
            format::bfyx,
            format::byxf,
            format::bfzyx,
            format::bzyxf,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::b_fs_zyx_fsv16,
            format::b_fs_zyx_fsv32,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
            format::bs_fs_yx_bsv4_fsv4,
            format::bs_fs_yx_bsv8_fsv4,
        };

        const auto& out_layout = node.get_output_layout();

        if (!one_of(out_layout.data_type, supported_types))
            return false;

        if (out_layout.data_padding)
            return false;

        bool any_dep_is_onednn = false;
        for (const auto& dep : node.get_dependencies()) {
            const auto& in_layout = dep.first->get_output_layout(false, dep.second);
            if (!one_of(in_layout.data_type, supported_types))
                return false;

            if (!is_supported_pad(in_layout))
                return false;

            if (!one_of(in_layout.format.value, supported_in_fmts))
                return false;

            if (dep.first->get_preferred_impl_type() == impl_types::onednn)
                any_dep_is_onednn = true;
        }

        if (!any_dep_is_onednn && format::is_simple_data_format(out_layout.format))
            return false;

        return true;
    }
};

}  // namespace onednn
}  // namespace cldnn
