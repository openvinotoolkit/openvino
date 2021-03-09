// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_max_suppression_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "non_max_suppression/non_max_suppression_kernel_selector.h"
#include "non_max_suppression/non_max_suppression_kernel_ref.h"

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct non_max_suppression_impl : typed_primitive_impl_ocl<non_max_suppression> {
    using parent = typed_primitive_impl_ocl<non_max_suppression>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<non_max_suppression_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<non_max_suppression>& instance,
                                                        int32_t) const override {
        kernel_arguments_data args;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_num_select_per_class())
            args.inputs.push_back(instance.num_select_per_class_mem());

        if (instance.has_iou_threshold())
            args.inputs.push_back(instance.iou_threshold_mem());

        if (instance.has_score_threshold())
            args.inputs.push_back(instance.score_threshold_mem());

        if (instance.has_soft_nms_sigma())
            args.inputs.push_back(instance.soft_nms_sigma_mem());

        args.output = instance.output_memory_ptr();
        if (instance.has_second_output())
            args.inputs.push_back(instance.second_output_mem());
        if (instance.has_third_output())
            args.inputs.push_back(instance.third_output_mem());

        return args;
    }

public:
    static primitive_impl* create(const non_max_suppression_node& arg) {
        auto params = get_default_params<kernel_selector::non_max_suppression_params>(arg);
        auto optional_params =
            get_default_optional_params<kernel_selector::non_max_suppression_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        params.inputs.push_back(convert_data_tensor(arg.input_scores().get_output_layout()));

        if (arg.has_num_select_per_class()) {
            params.inputs.push_back(convert_data_tensor(arg.num_select_per_class_node().get_output_layout()));
            params.has_num_select_per_class = true;
        }

        if (arg.has_iou_threshold()) {
            params.inputs.push_back(convert_data_tensor(arg.iou_threshold_node().get_output_layout()));
            params.has_iou_threshold = true;
        }

        if (arg.has_score_threshold()) {
            params.inputs.push_back(convert_data_tensor(arg.score_threshold_node().get_output_layout()));
            params.has_score_threshold = true;
        }

        if (arg.has_soft_nms_sigma()) {
            params.inputs.push_back(convert_data_tensor(arg.soft_nms_sigma_node().get_output_layout()));
            params.has_soft_nms_sigma = true;
        }

        if (arg.has_second_output()) {
            params.inputs.push_back(convert_data_tensor(arg.second_output_node().get_output_layout()));
            params.has_second_output = true;
        }

        if (arg.has_second_output()) {
            params.inputs.push_back(convert_data_tensor(arg.third_output_node().get_output_layout()));
            params.has_third_output = true;
        }

        params.sort_result_descending = primitive->sort_result_descending;
        params.box_encoding = primitive->center_point_box ? 1 : 0;

        auto& kernel_selector = kernel_selector::non_max_suppression_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto non_max_suppression_node = new non_max_suppression_impl(arg, best_kernels[0]);

        return non_max_suppression_node;
    }
};

namespace detail {

attach_non_max_suppression_impl::attach_non_max_suppression_impl() {
    implementation_map<non_max_suppression>::add(impl_types::ocl, non_max_suppression_impl::create, {
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
