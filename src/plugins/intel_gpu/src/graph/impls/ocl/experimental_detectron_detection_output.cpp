// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "experimental_detectron_detection_output_inst.hpp"
#include "ed_do/detection_output_kernel_ref.h"
#include "ed_do/detection_output_kernel_selector.h"

namespace cldnn {
namespace ocl {
struct experimental_detectron_detection_output_impl
    : public typed_primitive_impl_ocl<experimental_detectron_detection_output> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_detection_output>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::experimental_detectron_detection_output_kernel_selector;
    using kernel_params_t = kernel_selector::experimental_detectron_detection_output_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::experimental_detectron_detection_output_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_detection_output_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<experimental_detectron_detection_output>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        if (instance.desc()->num_outputs == 1) {
            // Legacy multi-output
            args.outputs.push_back(instance.output_classes_memory());
            args.outputs.push_back(instance.output_scores_memory());
        }

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<experimental_detectron_detection_output>();
        auto params = get_default_params<kernel_selector::experimental_detectron_detection_output_params>(impl_param);

        params.score_threshold = primitive->score_threshold;
        params.nms_threshold = primitive->nms_threshold;
        params.max_delta_log_wh = primitive->max_delta_log_wh;
        params.num_classes = primitive->num_classes;
        params.post_nms_count = primitive->post_nms_count;
        params.max_detections_per_image = primitive->max_detections_per_image;
        params.class_agnostic_box_regression = primitive->class_agnostic_box_regression;
        params.deltas_weights = primitive->deltas_weights;

        if (impl_param.prog->is_new_shape_infer()) {
            const size_t num_inputs = primitive->input_size();
            for (size_t i = 1; i < num_inputs; i++) {
                params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
            }

            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[1]));
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[2]));
        } else {
            const size_t num_deps = primitive->input_size();
            OPENVINO_ASSERT(num_deps == 6, "Unexpected deps num: ", num_deps);
            const size_t num_inputs = num_deps - 2;
            for (size_t i = 1; i < num_inputs; i++) {
                params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
            }
            for (size_t i = num_inputs; i < num_deps; i++) {
                params.outputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
            }
        }

        return params;
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
        impl_types::ocl,
        typed_primitive_impl_ocl<experimental_detectron_detection_output>::create<experimental_detectron_detection_output_impl>,
        types, formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_detection_output_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::experimental_detectron_detection_output)
