// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "non_max_suppression.hpp"
#include "non_max_suppression_inst.h"
#include "data_inst.h"
#include "non_max_suppression/non_max_suppression_kernel_ref.h"
#include "non_max_suppression/non_max_suppression_kernel_selector.h"

namespace cldnn {
namespace ocl {
struct non_max_suppression_impl : typed_primitive_impl_ocl<non_max_suppression> {
    using parent = typed_primitive_impl_ocl<non_max_suppression>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::non_max_suppression_kernel_selector;
    using kernel_params_t = kernel_selector::non_max_suppression_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::non_max_suppression_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<non_max_suppression_impl, kernel_params_t>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<non_max_suppression>& instance) const override {
        kernel_arguments_data args;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_num_select_per_class() && !instance.num_select_per_class_inst()->is_constant()) {
            args.inputs.push_back(instance.num_select_per_class_mem());
        }

        if (instance.has_iou_threshold() && !instance.iou_threshold_inst()->is_constant()) {
            args.inputs.push_back(instance.iou_threshold_mem());
        }

        if (instance.has_score_threshold() && !instance.score_threshold_inst()->is_constant()) {
            args.inputs.push_back(instance.score_threshold_mem());
        }

        if (instance.has_soft_nms_sigma() && !instance.soft_nms_sigma_inst()->is_constant()) {
            args.inputs.push_back(instance.soft_nms_sigma_mem());
        }

        // New API for mutiple outputs support
        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }
        // // Legacy multi-output
        if (instance.has_second_output())
            args.outputs.push_back(instance.second_output_mem());
        if (instance.has_third_output())
            args.outputs.push_back(instance.third_output_mem());

        return args;
    }

public:
static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<non_max_suppression>();
        const auto& arg = impl_param.prog->get_node(impl_param.desc->id).as<non_max_suppression>();
        auto params = get_default_params<kernel_selector::non_max_suppression_params>(impl_param);

        const auto input_scores_idx = 1;
        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[input_scores_idx]));

        if (arg.has_num_select_per_class()) {
            cldnn::program_node& node = arg.num_select_per_class_node();
            if (node.is_type<data>()) {
                params.num_select_per_class_type = kernel_selector::base_params::ArgType::Constant;
                params.num_select_per_class = get_value<int>(node);
            } else {
                params.num_select_per_class_type = kernel_selector::base_params::ArgType::Input;
                params.inputs.push_back(convert_data_tensor(impl_param.get_output_layout()));
            }
        }

        if (arg.has_iou_threshold()) {
            cldnn::program_node& node = arg.iou_threshold_node();
            if (node.is_type<data>()) {
                params.iou_threshold_type = kernel_selector::base_params::ArgType::Constant;
                params.iou_threshold = get_value<float>(node);
            } else {
                params.iou_threshold_type = kernel_selector::base_params::ArgType::Input;
                params.inputs.push_back(convert_data_tensor(impl_param.get_output_layout()));
            }
        }

        if (arg.has_score_threshold()) {
            cldnn::program_node& node = arg.score_threshold_node();
            if (node.is_type<data>()) {
                params.score_threshold_type = kernel_selector::base_params::ArgType::Constant;
                params.score_threshold = get_value<float>(node);
            } else {
                params.score_threshold_type = kernel_selector::base_params::ArgType::Input;
                params.inputs.push_back(convert_data_tensor(impl_param.get_output_layout()));
            }
        }

        if (arg.has_soft_nms_sigma()) {
            cldnn::program_node& node = arg.soft_nms_sigma_node();
            if (node.is_type<data>()) {
                params.soft_nms_sigma_type = kernel_selector::base_params::ArgType::Constant;
                params.soft_nms_sigma = get_value<float>(node);
            } else {
                params.soft_nms_sigma_type = kernel_selector::base_params::ArgType::Input;
                params.inputs.push_back(convert_data_tensor(impl_param.get_output_layout()));
            }
        }

        auto get_additional_output_node_idx = [&] (bool is_third) {
            size_t offset = 2;
            offset += arg.has_num_select_per_class();
            offset += arg.has_iou_threshold();
            offset += arg.has_score_threshold();
            offset += arg.has_soft_nms_sigma();
            if (is_third)
                offset += arg.has_second_output();
            return offset;
        };

        // Legacy multi-output
        if (arg.has_second_output()) {
            params.outputs.push_back(convert_data_tensor(impl_param.input_layouts[get_additional_output_node_idx(false)]));
        }

        if (arg.has_third_output()) {
            params.outputs.push_back(convert_data_tensor(impl_param.input_layouts[get_additional_output_node_idx(true)]));
        }

        if (arg.use_multiple_outputs()) {
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[1]));
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[2]));
        }

        params.sort_result_descending = primitive->sort_result_descending;
        params.box_encoding = primitive->center_point_box ? kernel_selector::BoxEncodingType::BOX_ENCODING_CENTER
                                                          : kernel_selector::BoxEncodingType::BOX_ENCODING_CORNER;
        switch (primitive->rotation) {
            case non_max_suppression::Rotation::CLOCKWISE:
                params.rotation = kernel_selector::NMSRotationType::CLOCKWISE;
                break;
            case non_max_suppression::Rotation::COUNTERCLOCKWISE:
                params.rotation = kernel_selector::NMSRotationType::COUNTERCLOCKWISE;
                break;
            default:
                params.rotation = kernel_selector::NMSRotationType::NONE;
        }

        if (impl_param.get_program().get_node(primitive->id).is_dynamic()) {
            params.reuse_internal_buffer = true;
        }

        return params;
    }

private:
    template <class T>
    static T get_value(cldnn::program_node& node) {
        T retValue;
        auto mem = node.as<data>().get_attached_memory_ptr();
        auto& stream = node.get_program().get_stream();
        switch (mem->get_layout().data_type) {
        case data_types::f16: {
            mem_lock<ov::float16, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<ov::float16*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::f32: {
            mem_lock<float, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<float*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::i32: {
            mem_lock<int32_t, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<int32_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::i64: {
            mem_lock<int64_t, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<int64_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        default:
            throw std::runtime_error("Not supported data type.");
        }

        return retValue;
    }
};

std::unique_ptr<primitive_impl> NMSImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<non_max_suppression>());
    return typed_primitive_impl_ocl<non_max_suppression>::create<non_max_suppression_impl>(static_cast<const non_max_suppression_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::non_max_suppression_impl)
