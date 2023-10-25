// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "ed_rfe/roi_feature_extractor_kernel_selector.h"
#include "ed_rfe/roi_feature_extractor_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct experimental_detectron_roi_feature_extractor_impl : public typed_primitive_impl_ocl<experimental_detectron_roi_feature_extractor> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_roi_feature_extractor>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::experimental_detectron_roi_feature_extractor_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::experimental_detectron_roi_feature_extractor_params,
                                      kernel_selector::experimental_detectron_roi_feature_extractor_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::experimental_detectron_roi_feature_extractor_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_roi_feature_extractor_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const experimental_detectron_roi_feature_extractor_inst& instance) const override {
        kernel_arguments_data args;

        for (std::size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }
        args.outputs = { instance.output_memory_ptr() };

        return args;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            experimental_detectron_roi_feature_extractor_inst& instance) override {
        instance.copy_rois_input_to_second_output();
        return parent::execute_impl(events, instance);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<experimental_detectron_roi_feature_extractor>();
        auto params = get_default_params<kernel_selector::experimental_detectron_roi_feature_extractor_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::experimental_detectron_roi_feature_extractor_optional_params>(
            impl_param.get_program());

        size_t number_of_inputs = primitive->input_size() - 1;
        for (std::size_t i = 1; i < number_of_inputs; i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        params.output_dim = primitive->output_dim;
        params.pooled_height = primitive->pooled_height;
        params.pooled_width = primitive->pooled_width;
        params.pyramid_scales = primitive->pyramid_scales;
        params.sampling_ratio = primitive->sampling_ratio;
        params.aligned = primitive->aligned;
        params.number_of_inputs = number_of_inputs;

        return {params, optional_params};
    }
};

namespace detail {
attach_experimental_detectron_roi_feature_extractor_impl::attach_experimental_detectron_roi_feature_extractor_impl() {
    implementation_map<experimental_detectron_roi_feature_extractor>::add(
        impl_types::ocl,
        typed_primitive_impl_ocl<experimental_detectron_roi_feature_extractor>::create<experimental_detectron_roi_feature_extractor_impl>,
        {
            std::make_tuple(data_types::f16, format::bfyx),
            std::make_tuple(data_types::f32, format::bfyx)
        });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_roi_feature_extractor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::experimental_detectron_roi_feature_extractor)
