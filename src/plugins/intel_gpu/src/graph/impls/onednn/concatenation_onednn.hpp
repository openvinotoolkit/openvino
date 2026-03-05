// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_inst.h"
#include "impls/onednn/utils.hpp"
#include "registry/implementation_manager.hpp"

#include <memory>
namespace cldnn {
namespace onednn {

struct ConcatenationImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::concat")
    ConcatenationImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::onednn, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<concatenation>());
        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown || !config.get_use_onednn())
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

        auto is_feature_aligned = [](const layout& l) {
            if (!format::is_blocked(l.format))
                return true;

            const auto& order = format::internal_order(l.format);
            const size_t feature_dim_idx = order.find('f');
            if (feature_dim_idx == std::string::npos)
                return true;

            auto feature_dim = l.get_partial_shape()[feature_dim_idx];
            if (feature_dim.is_dynamic())
                return false;

            const auto& block_sizes = format::block_sizes(l.format);
            auto block_it = std::find_if(block_sizes.begin(), block_sizes.end(), [&](const auto& block) {
                return block.first == feature_dim_idx;
            });

            if (block_it == block_sizes.end())
                return true;

            const int feature_block_size = block_it->second;
            return feature_dim.get_length() % feature_block_size == 0;
        };

        // onednn concatenation doesn't support non-zero padding which can occur for unaligned feature.
        if (!is_feature_aligned(out_layout)) {
            return false;
        }

        const auto& concat_node = node.as<concatenation>();
        auto concat_axis = concat_node.get_primitive()->axis;

        size_t index = 0;
        for (const auto& dep : node.get_dependencies()) {
            const auto& in_layout = dep.first->get_output_layout(false, dep.second);

            if (!one_of(in_layout.data_type, supported_types))
                return false;

            if (!is_supported_pad(in_layout))
                return false;

            if (!one_of(in_layout.format.value, supported_in_fmts))
                return false;

            // WA: Onednn has an issue in simple_concat blocked format Odd value, will be fixed next release.
            if (index !=0 && concat_axis == 1 &&
                !format::is_simple_data_format(in_layout.format) &&
                in_layout.get_partial_shape()[1].is_static() &&
                in_layout.get_partial_shape()[1].get_length() % 2 != 0)
                return false;
            index++;
        }

        return true;
    }
};

}  // namespace onednn
}  // namespace cldnn
