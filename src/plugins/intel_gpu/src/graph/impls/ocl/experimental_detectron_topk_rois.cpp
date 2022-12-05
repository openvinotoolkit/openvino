// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <experimental_detectron_topk_rois_inst.h>
#include "intel_gpu/runtime/error_handler.hpp"
#include <impls/implementation_map.hpp>
#include <ed_topkroi/topk_rois_ref.h>
#include <ed_topkroi/topk_rois_kernel_selector.h>
#include "primitive_base.hpp"
#include <vector>

namespace cldnn {
namespace ocl {

struct experimental_detectron_topk_rois_impl : typed_primitive_impl_ocl<experimental_detectron_topk_rois> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_topk_rois>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::experimental_detectron_topk_rois_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::experimental_detectron_topk_roi_params,
                                      kernel_selector::experimental_detectron_topk_roi_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_topk_rois_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<experimental_detectron_topk_rois>();
        auto params = get_default_params<kernel_selector::experimental_detectron_topk_roi_params>(impl_param);
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.max_rois = primitive->max_rois;

        return {params, {}};
    }
};

namespace detail {

attach_experimental_detectron_topk_rois_impl::attach_experimental_detectron_topk_rois_impl() {
    implementation_map<experimental_detectron_topk_rois>::add(
        impl_types::ocl,
        typed_primitive_impl_ocl<experimental_detectron_topk_rois>::create<experimental_detectron_topk_rois_impl>,
        {
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx)
        });
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_topk_rois_impl)
