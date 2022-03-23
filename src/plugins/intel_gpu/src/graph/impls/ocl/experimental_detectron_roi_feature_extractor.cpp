// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"
#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "edrfe/experimental_detectron_roi_feature_extractor_kernel_selector.h"
#include "edrfe/experimental_detectron_roi_feature_extractor_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct experimental_detectron_roi_feature_extractor_impl : public typed_primitive_impl_ocl<experimental_detectron_roi_feature_extractor> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_roi_feature_extractor>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_roi_feature_extractor_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(experimental_detectron_roi_feature_extractor_inst& instance, int32_t) const override {
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
    static primitive_impl* create(const experimental_detectron_roi_feature_extractor_node& arg) {
        const auto output_layout = arg.get_output_layout();
        const auto padding_filling_value = output_layout.data_padding.filling_value();
        CLDNN_ERROR_NOT_EQUAL(arg.id(),
                              "experimental_detectron_roi_feature_extractor padding filling value",
                              padding_filling_value,
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in experimental_detectron_roi_feature_extractor.");

        auto params = get_default_params<kernel_selector::experimental_detectron_roi_feature_extractor_params>(arg);
        auto optional_params = get_default_optional_params<kernel_selector::experimental_detectron_roi_feature_extractor_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        size_t number_of_inputs = primitive->input_size() - 1;
        for (std::size_t i = 1; i < number_of_inputs; i++) {
            params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }

        params.output_dim = primitive->output_dim;
        params.pooled_height = primitive->pooled_height;
        params.pooled_width = primitive->pooled_width;
        params.pyramid_scales = primitive->pyramid_scales;
        params.sampling_ratio = primitive->sampling_ratio;
        params.aligned = primitive->aligned;
        params.number_of_inputs = number_of_inputs;

        auto& kernel_selector = kernel_selector::experimental_detectron_roi_feature_extractor_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "best_kernels.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");
        return new experimental_detectron_roi_feature_extractor_impl(arg, best_kernels.front());
    }
};

namespace detail {
attach_experimental_detectron_roi_feature_extractor_impl::attach_experimental_detectron_roi_feature_extractor_impl() {
    implementation_map<experimental_detectron_roi_feature_extractor>::add(impl_types::ocl,
                                                                            experimental_detectron_roi_feature_extractor_impl::create, {
                                                                                std::make_tuple(data_types::f16, format::bfyx),
                                                                                std::make_tuple(data_types::f32, format::bfyx)
                                                                                });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
