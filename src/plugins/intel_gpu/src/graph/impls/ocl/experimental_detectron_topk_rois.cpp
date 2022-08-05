// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <experimental_detectron_topk_rois_inst.h>
#include <edtopkroi/experimental_detectron_topk_rois_ref.h>
#include "intel_gpu/runtime/error_handler.hpp"
#include <impls/implementation_map.hpp>
#include <edtopkroi/experimental_detectron_topk_rois_kernel_selector.h>
#include "primitive_base.hpp"
#include <vector>

namespace cldnn {
namespace ocl {

struct experimental_detectron_topk_rois_impl : typed_primitive_impl_ocl<experimental_detectron_topk_rois> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_topk_rois>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_topk_rois_impl>(*this);
    }

    static primitive_impl *create(const experimental_detectron_topk_rois_node &arg) {
        auto params = get_default_params<kernel_selector::experimental_detectron_topk_roi_params>(
                arg);
        const auto& experimental_detectron_topk_rois_kernel_selector =
                kernel_selector::experimental_detectron_topk_rois_kernel_selector::Instance();
        const auto& primitive = arg.get_primitive();
        params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));
        params.max_rois = primitive->max_rois;
        auto best_kernels = experimental_detectron_topk_rois_kernel_selector.GetBestKernels(params,
                                                                                            kernel_selector::experimental_detectron_topk_roi_optional_params());
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");
        return new experimental_detectron_topk_rois_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_experimental_detectron_topk_rois_impl::attach_experimental_detectron_topk_rois_impl() {
    implementation_map<experimental_detectron_topk_rois>::add(impl_types::ocl,
                                                              experimental_detectron_topk_rois_impl::create, {
                                                                      std::make_tuple(data_types::f16, format::bfyx),
                                                                      std::make_tuple(data_types::f32, format::bfyx)
                                                              });
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn
