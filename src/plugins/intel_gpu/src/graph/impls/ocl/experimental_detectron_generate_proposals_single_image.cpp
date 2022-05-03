// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_generate_proposals_single_image_inst.hpp"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "edgpsi/experimental_detectron_generate_proposals_single_image_kernel_selector.h"
#include "edgpsi/experimental_detectron_generate_proposals_single_image_kernel_ref.h"


namespace cldnn {
namespace ocl {
struct experimental_detectron_generate_proposals_single_image_impl
        : public typed_primitive_impl_ocl<experimental_detectron_generate_proposals_single_image> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_generate_proposals_single_image>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_generate_proposals_single_image_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<experimental_detectron_generate_proposals_single_image>& instance, int32_t) const override {
        kernel_arguments_data args;
        const auto num_inputs = instance.inputs_memory_count();
        for (size_t i = 0; i < num_inputs; ++i) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        args.outputs.push_back(instance.output_memory_ptr());
        args.inputs.push_back(instance.output_roi_scores_memory());

        return args;
    }

public:
    static primitive_impl* create(const experimental_detectron_generate_proposals_single_image_node& arg) {
        auto params = get_default_params<kernel_selector::experimental_detectron_generate_proposals_single_image_params>(arg);
        auto optional_params = get_default_optional_params<
                kernel_selector::experimental_detectron_generate_proposals_single_image_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();

        params.min_size = primitive->min_size;
        params.nms_threshold  = primitive->nms_threshold;
        params.pre_nms_count = primitive->pre_nms_count;
        params.post_nms_count = primitive->post_nms_count;

        params.inputs.push_back(convert_data_tensor(arg.anchors().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.deltas().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.scores().get_output_layout()));

        params.inputs.push_back(convert_data_tensor(arg.output_roi_scores_node().get_output_layout()));

        const auto& kernel_selector = kernel_selector::experimental_detectron_generate_proposals_single_image_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "best_kernels.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new experimental_detectron_generate_proposals_single_image_impl(arg, best_kernels[0]);
    }
};

namespace detail {
attach_experimental_detectron_generate_proposals_single_image_impl::attach_experimental_detectron_generate_proposals_single_image_impl() {
    implementation_map<experimental_detectron_generate_proposals_single_image>::add(impl_types::ocl,
                                                                                    experimental_detectron_generate_proposals_single_image_impl::create, {
                                                 std::make_tuple(data_types::f16, format::bfyx),
                                                 std::make_tuple(data_types::f32, format::bfyx)
                                         });
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
