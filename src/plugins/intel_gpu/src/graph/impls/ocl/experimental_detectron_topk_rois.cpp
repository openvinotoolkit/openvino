// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "experimental_detectron_topk_rois_inst.h"
#include "ed_topkroi/topk_rois_ref.h"
#include "ed_topkroi/topk_rois_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct experimental_detectron_topk_rois_impl : typed_primitive_impl_ocl<experimental_detectron_topk_rois> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_topk_rois>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::experimental_detectron_topk_rois_kernel_selector;
    using kernel_params_t = kernel_selector::experimental_detectron_topk_roi_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::experimental_detectron_topk_rois_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<experimental_detectron_topk_rois_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<experimental_detectron_topk_rois>();
        auto params = get_default_params<kernel_selector::experimental_detectron_topk_roi_params>(impl_param);
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.max_rois = primitive->max_rois;

        return params;
    }
};

namespace detail {

attach_experimental_detectron_topk_rois_impl::attach_experimental_detectron_topk_rois_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {format::bfyx,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv16,
                    format::bs_fs_yx_bsv32_fsv32};
    implementation_map<experimental_detectron_topk_rois>::add(
        impl_types::ocl,
        typed_primitive_impl_ocl<experimental_detectron_topk_rois>::create<experimental_detectron_topk_rois_impl>,
        types,
        formats);
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_topk_rois_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::experimental_detectron_topk_rois)
