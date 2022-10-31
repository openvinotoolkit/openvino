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

    static std::unique_ptr<primitive_impl> create(const experimental_detectron_topk_rois_node &arg, const kernel_impl_params& impl_param) {
        auto kernel_params = get_kernel_params(impl_param);
        auto& kernel_selector = kernel_selector_t::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(kernel_params.first, kernel_params.second);

        return make_unique<experimental_detectron_topk_rois_impl>(arg, best_kernel);
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
    implementation_map<experimental_detectron_topk_rois>::add(impl_types::ocl,
                                                              experimental_detectron_topk_rois_impl::create,
                                                              types,
                                                              formats);
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_topk_rois_impl,
                             cldnn::object_type::EXPERIMENTAL_DETECTRON_TOPK_ROIS_IMPL)