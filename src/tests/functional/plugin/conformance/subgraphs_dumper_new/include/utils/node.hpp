// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "cache/meta/input_info.hpp"
#include "functional_test_utils/summary/op_info.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {
template <typename dType>
inline InputInfo::Range get_const_ranges(const std::shared_ptr<ov::op::v0::Constant>& node) {
    size_t elements_count = ov::shape_size(node->get_shape());
    const auto& const_values = node->cast_vector<dType>();
    auto max = *std::max_element(const_values.begin(), const_values.end());
    auto min = *std::min_element(const_values.begin(), const_values.end());
    return InputInfo::Range(static_cast<double>(min), static_cast<double>(max));
}

inline std::map<std::string, InputInfo> get_input_info_by_node(const std::shared_ptr<ov::Node>& node) {
    std::map<std::string, InputInfo> input_info;
    for (size_t port_id = 0; port_id < node->get_input_size(); ++port_id) {
        InputInfo in_info;
        std::shared_ptr<ov::Node> input_node = node->input_value(port_id).get_node_shared_ptr();
        std::string input_name = input_node->get_friendly_name();
        if (std::dynamic_pointer_cast<ov::op::v0::Constant>(input_node)) {
            auto const_node =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(input_node);
            in_info.is_const = true;
            switch (node->get_output_element_type(0)) {
            case ov::element::Type_t::boolean: {
                in_info.ranges = get_const_ranges<bool>(const_node);
                break;
            }
            case ov::element::Type_t::bf16: {
                in_info.ranges = get_const_ranges<ov::bfloat16>(const_node);
                break;
            }
            case ov::element::Type_t::f16: {
                in_info.ranges = get_const_ranges<ov::float16>(const_node);
                break;
            }
            case ov::element::Type_t::f32: {
                in_info.ranges = get_const_ranges<float>(const_node);
                break;
            }
            case ov::element::Type_t::f64: {
                in_info.ranges = get_const_ranges<double>(const_node);
                break;
            }
            case ov::element::Type_t::i8: {
                in_info.ranges = get_const_ranges<int8_t>(const_node);
                break;
            }
            case ov::element::Type_t::i16: {
                in_info.ranges = get_const_ranges<int16_t>(const_node);
                break;
            }
            case ov::element::Type_t::i32: {
                in_info.ranges = get_const_ranges<int32_t>(const_node);
                break;
            }
            case ov::element::Type_t::i64: {
                in_info.ranges = get_const_ranges<int64_t>(const_node);
                break;
            }
                // TODO cast_vector doesn't support u1 now
                //        case ov::element::Type_t::u1:
                //            return get_const_ranges<char>(const_node);
            case ov::element::Type_t::u8: {
                in_info.ranges = get_const_ranges<uint8_t>(const_node);
                break;
            }
            case ov::element::Type_t::u16: {
                in_info.ranges = get_const_ranges<uint16_t>(const_node);
                break;
            }
            case ov::element::Type_t::u32: {
                in_info.ranges = get_const_ranges<uint32_t>(const_node);
                break;
            }
            case ov::element::Type_t::u64: {
                in_info.ranges = get_const_ranges<uint64_t>(const_node);
                break;
            }
            default: {
                std::cout << "Can't get ranges.. Unsupported data type" << std::endl;
                break;
            }
            }
        }
        input_info.insert({ input_name, in_info });
    }
    return input_info;
}

// replace all input node by parameters and constants instead of non input mode types
// if `!is_save_const` replace only by parameters
inline std::shared_ptr<ov::Node> clone_node(std::shared_ptr<ov::Node> node, bool is_save_const = false) {
    bool has_parameters = false;
    ov::OutputVector inputs;
    inputs.resize(node->get_input_size());
    std::string in_name_base = ov::test::functional::get_node_version(node);
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        std::string node_name = in_name_base + "_" + std::to_string(i);
        if (is_save_const) {
            // todo: replace deprecated code
            OPENVINO_SUPPRESS_DEPRECATED_START
            const auto constant_input = ov::get_constant_from_source(node->input(i).get_source_output());
            OPENVINO_SUPPRESS_DEPRECATED_END
            if (constant_input) {
                auto in_const = std::make_shared<ov::op::v0::Constant>(constant_input->get_element_type(),
                                                                       constant_input->get_shape(),
                                                                       constant_input->get_data_ptr());
                in_const->set_friendly_name(node_name);
                inputs[i] = in_const;
                continue;
            }
        }
        has_parameters = true;
        auto param =
            std::make_shared<ov::op::v0::Parameter>(node->get_input_element_type(i), node->get_input_partial_shape(i));
        param->set_friendly_name(node_name);
        inputs[i] = param;
    }
    if (!has_parameters) {
        std::cout << "The operation: " + node->get_friendly_name() + " does not have parameters!" << std::endl;
        return nullptr;
    }
    std::shared_ptr<ov::Node> cloned_node = node->clone_with_new_inputs(inputs);
    cloned_node->set_friendly_name(in_name_base);
    return cloned_node;
}

// all inputs are defined as parameters and contains detailed info in meta
inline std::shared_ptr<ov::Model> generate_model_by_node(const std::shared_ptr<ov::Node>& node) {
    static size_t model_cnt = 0;
    auto cloned_node = clone_node(node);
    ov::OutputVector results;
    for (auto& out : cloned_node->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(out));
    }
    auto model = std::make_shared<ov::Model>(results);
    model->set_friendly_name(cloned_node->get_friendly_name() + "_" + std::to_string(model_cnt++));
    return model;
}

inline std::string get_node_type(const std::shared_ptr<ov::Node>& node) {
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        if (node->get_input_partial_shape(i).is_dynamic()) {
            return "dynamic";
        }
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return "dynamic";
        }
    }
    return "static";
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov