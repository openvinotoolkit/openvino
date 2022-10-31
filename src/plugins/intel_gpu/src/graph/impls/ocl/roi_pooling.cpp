// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_pooling_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "roi_pooling/roi_pooling_kernel_selector.h"
#include "roi_pooling/roi_pooling_kernel_ref.h"

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::pool_type cldnn_2_pool_type(pooling_mode mode) {
    switch (mode) {
        case pooling_mode::max:
            return kernel_selector::pool_type::MAX;
        case pooling_mode::average:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::average_no_padding:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::bilinear:
            return kernel_selector::pool_type::BILINEAR;
        case pooling_mode::deformable_bilinear:
            return kernel_selector::pool_type::DEFORMABLE_BILINEAR;
        default:
            assert(0);
            return kernel_selector::pool_type::MAX;
    }
}
}  // namespace

struct roi_pooling_impl : typed_primitive_impl_ocl<roi_pooling> {
    using parent = typed_primitive_impl_ocl<roi_pooling>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<roi_pooling_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<roi_pooling>& instance, int32_t) const override {
        kernel_arguments_data args;

        if (instance.argument->mode == pooling_mode::deformable_bilinear && !instance.argument->no_trans)
            args.inputs = {
                instance.input_memory_ptr(),
                instance.rois_memory(),
                instance.trans_memory()};
        else
            args.inputs = {instance.input_memory_ptr(), instance.rois_memory()};

        args.outputs = { instance.output_memory_ptr() };

        return args;
    }

public:
    static std::unique_ptr<primitive_impl> create(const roi_pooling_node& arg, const kernel_impl_params& impl_param) {
        const auto& input_layout = impl_param.input_layouts[0];
        const auto& output_layout = impl_param.output_layout;
        const auto& rois_layout = impl_param.input_layouts[1];
        const auto& primitive = arg.get_primitive();

        const auto padding_filling_value = output_layout.data_padding.filling_value();

        CLDNN_ERROR_NOT_EQUAL(arg.id(),
                              "roi_pooling padding filling value",
                              padding_filling_value,
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in roi_pooling.");
        CLDNN_ERROR_NOT_PROPER_FORMAT(arg.id(),
                                      "Input_layout.format",
                                      input_layout.format.value,
                                      "output_layout.format",
                                      output_layout.format);
        auto roi_params = get_default_params<kernel_selector::roi_pooling_params>(impl_param);
        auto roi_optional_params =
            get_default_optional_params<kernel_selector::roi_pooling_optional_params>(arg.get_program());

        roi_params.inputs.push_back(convert_data_tensor(rois_layout));
        if (primitive->mode == pooling_mode::deformable_bilinear && !primitive->no_trans)
            roi_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[2]));
        roi_params.mode = cldnn_2_pool_type(primitive->mode);
        roi_params.position_sensitive = primitive->position_sensitive;
        roi_params.pooled_width = primitive->pooled_width;
        roi_params.pooled_height = primitive->pooled_height;
        roi_params.spatial_scale = primitive->spatial_scale;
        roi_params.spatial_bins_x = primitive->spatial_bins_x;
        roi_params.spatial_bins_y = primitive->spatial_bins_y;
        roi_params.trans_std = primitive->trans_std;
        roi_params.no_trans = primitive->no_trans;
        roi_params.part_size = primitive->part_size;
        roi_params.group_size = primitive->group_size;

        auto& kernel_selector = kernel_selector::roi_pooling_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(roi_params, roi_optional_params);

        return make_unique<roi_pooling_impl>(arg, best_kernel);
    }
};

namespace detail {

attach_roi_pooling_impl::attach_roi_pooling_impl() {
    auto formats = {format::bfyx,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv32,
                    format::bs_fs_yx_bsv32_fsv16};

    auto types = {data_types::f16, data_types::f32};

    implementation_map<roi_pooling>::add(impl_types::ocl, roi_pooling_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::roi_pooling_impl, cldnn::object_type::ROI_POOLING_IMPL)
