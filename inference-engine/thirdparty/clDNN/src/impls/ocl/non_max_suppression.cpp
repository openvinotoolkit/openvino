// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "data_inst.h"
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

        if (instance.has_num_select_per_class() && !instance.node.num_select_per_class_node().is_constant()) {
            args.inputs.push_back(instance.num_select_per_class_mem());
        }

        if (instance.has_iou_threshold() && !instance.node.iou_threshold_node().is_constant()) {
            args.inputs.push_back(instance.iou_threshold_mem());
        }

        if (instance.has_score_threshold() && !instance.node.score_threshold_node().is_constant()) {
            args.inputs.push_back(instance.score_threshold_mem());
        }

        if (instance.has_soft_nms_sigma() && !instance.node.soft_nms_sigma_node().is_constant()) {
            args.inputs.push_back(instance.soft_nms_sigma_mem());
        }

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
            cldnn::program_node& node = arg.num_select_per_class_node();
            if (node.is_constant()) {
                params.num_select_per_class_type = kernel_selector::NmsArgType::Constant;
                params.num_select_per_class = getValue<int>(node);
            } else {
                params.num_select_per_class_type = kernel_selector::NmsArgType::Input;
                params.inputs.push_back(convert_data_tensor(node.get_output_layout()));
            }
        }

        if (arg.has_iou_threshold()) {
            cldnn::program_node& node = arg.iou_threshold_node();
            if (node.is_constant()) {
                params.iou_threshold_type = kernel_selector::NmsArgType::Constant;
                params.iou_threshold = getValue<float>(node);
            } else {
                params.iou_threshold_type = kernel_selector::NmsArgType::Input;
                params.inputs.push_back(convert_data_tensor(node.get_output_layout()));
            }
        }

        if (arg.has_score_threshold()) {
            cldnn::program_node& node = arg.score_threshold_node();
            if (node.is_constant()) {
                params.score_threshold_type = kernel_selector::NmsArgType::Constant;
                params.score_threshold = getValue<float>(node);
            } else {
                params.score_threshold_type = kernel_selector::NmsArgType::Input;
                params.inputs.push_back(convert_data_tensor(node.get_output_layout()));
            }
        }

        if (arg.has_soft_nms_sigma()) {
            cldnn::program_node& node = arg.soft_nms_sigma_node();
            if (node.is_constant()) {
                params.soft_nms_sigma_type = kernel_selector::NmsArgType::Constant;
                params.soft_nms_sigma = getValue<float>(node);
            } else {
                params.soft_nms_sigma_type = kernel_selector::NmsArgType::Input;
                params.inputs.push_back(convert_data_tensor(node.get_output_layout()));
            }
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
        params.box_encoding = primitive->center_point_box ?
            kernel_selector::BoxEncodingType::BOX_ENCODING_CENTER : kernel_selector::BoxEncodingType::BOX_ENCODING_CORNER;

        auto& kernel_selector = kernel_selector::non_max_suppression_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto non_max_suppression_node = new non_max_suppression_impl(arg, best_kernels[0]);

        return non_max_suppression_node;
    }

private:
    template <class T>
    static T getValue(cldnn::program_node& node) {
        T retValue;
        auto mem = node.as<data>().get_attached_memory_ptr();
        auto& stream = node.get_program().get_stream();
        switch (mem->get_layout().data_type) {
        case data_types::f16: {
            mem_lock<half_t> lock(mem, stream);
            auto mem_value = static_cast<half_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::f32: {
            mem_lock<float> lock(mem, stream);
            auto mem_value = static_cast<float*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::i32: {
            mem_lock<int32_t> lock(mem, stream);
            auto mem_value = static_cast<int32_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::i64: {
            mem_lock<int64_t> lock(mem, stream);
            auto mem_value = static_cast<int64_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        default:
            throw std::runtime_error("Not supported data type.");
        }

        return retValue;
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
