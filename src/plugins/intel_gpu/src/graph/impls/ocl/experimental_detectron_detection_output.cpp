// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ed_do/detection_output_kernel_ref.h"
#include "ed_do/detection_output_kernel_selector.h"
#include "experimental_detectron_detection_output_inst.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {
struct experimental_detectron_detection_output_impl
    : public typed_primitive_impl_ocl<experimental_detectron_detection_output> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_detection_output>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_detection_output_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<experimental_detectron_detection_output>& instance,
                                        int32_t unused) const override {
        kernel_arguments_data args = parent::get_arguments(instance, unused);
        args.inputs.push_back(instance.output_classes_memory());
        args.inputs.push_back(instance.output_scores_memory());

        return args;
    }

public:
    static primitive_impl* create(const experimental_detectron_detection_output_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::experimental_detectron_detection_output_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::experimental_detectron_detection_output_optional_params>(
                arg.get_program());

        const auto& primitive = arg.get_primitive();

        params.score_threshold = primitive->score_threshold;
        params.nms_threshold = primitive->nms_threshold;
        params.max_delta_log_wh = primitive->max_delta_log_wh;
        params.num_classes = primitive->num_classes;
        params.post_nms_count = primitive->post_nms_count;
        params.max_detections_per_image = primitive->max_detections_per_image;
        params.class_agnostic_box_regression = primitive->class_agnostic_box_regression;
        params.deltas_weights = primitive->deltas_weights;

        params.inputs.push_back(convert_data_tensor(arg.deltas().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.scores().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.image_size_info().get_output_layout()));

        params.inputs.push_back(convert_data_tensor(arg.output_classes_node().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.output_scores_node().get_output_layout()));

        const auto& kernel_selector = kernel_selector::experimental_detectron_detection_output_kernel_selector::Instance();
        const auto best_kernel = kernel_selector.get_best_kernel(params, optional_params);

        return new experimental_detectron_detection_output_impl(arg, best_kernel);
    }
};

namespace detail {
attach_experimental_detectron_detection_output_impl::attach_experimental_detectron_detection_output_impl() {
  const std::vector<data_types> types {data_types::f16, data_types::f32};
  const std::vector<format::type> formats = {format::bfyx,
                        format::b_fs_yx_fsv16,
                        format::b_fs_yx_fsv32,
                        format::bs_fs_yx_bsv16_fsv16,
                        format::bs_fs_yx_bsv32_fsv32,
                        format::bs_fs_yx_bsv32_fsv16};

  implementation_map<experimental_detectron_detection_output>::add(
      impl_types::ocl, experimental_detectron_detection_output_impl::create,
      types, formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_detection_output_impl,
                             cldnn::object_type::ACTIVATION_IMPL)