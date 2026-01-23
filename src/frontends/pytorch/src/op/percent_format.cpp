// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_percent_format(const NodeContext& context) {
    // aten::percentFormat(str format_string, tuple values) -> str
    
    num_inputs_check(context, 2, 2);
    
    PYTORCH_OP_CONVERSION_CHECK(context.get_input(0).get_node_shared_ptr()->get_type_info() == 
                                v0::Constant::get_type_info_static(),
                                "aten::percentFormat: format string must be a constant. "
                                "Dynamic string formatting is not supported in OpenVINO.");
    
    auto format_str = context.const_input<std::string>(0);
    auto values_input = context.get_input(1);
    const auto&& value_list = get_list_as_outputs(values_input);
    
    PYTORCH_OP_CONVERSION_CHECK(!value_list.empty(),
                                "aten::percentFormat: values tuple cannot be empty.");
    
    std::string result = format_str;
    size_t value_idx = 0;
    size_t pos = 0;
    
    while ((pos = result.find('%', pos)) != std::string::npos) {
        if (pos + 1 >= result.length()) {
            break;
        }
        
        if (result[pos + 1] == '%') {
            pos += 2;
            continue;
        }
        
        PYTORCH_OP_CONVERSION_CHECK(value_idx < value_list.size(),
                                    "aten::percentFormat: not enough arguments for format string.");
        
        auto value_node = value_list[value_idx].get_node_shared_ptr();
        PYTORCH_OP_CONVERSION_CHECK(value_node->get_type_info() == v0::Constant::get_type_info_static(),
                                    "aten::percentFormat: all format values must be constants. "
                                    "Dynamic values are not supported in OpenVINO.");
        
        auto value_const = std::dynamic_pointer_cast<v0::Constant>(value_node);
        
        size_t spec_end = pos + 1;
        while (spec_end < result.length() && 
               (std::isdigit(result[spec_end]) || result[spec_end] == '.' || result[spec_end] == '-')) {
            spec_end++;
        }
        
        if (spec_end >= result.length()) {
            break;
        }
        
        char format_type = result[spec_end];
        std::string format_spec = result.substr(pos, spec_end - pos + 1);
        std::string replacement;
        auto elem_type = value_const->get_element_type();
        
        if (format_type == 's') {
            if (elem_type == element::Type_t::string) {
                auto str_val = value_const->get_data_ptr<std::string>();
                replacement = str_val[0];
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false, 
                                          "aten::percentFormat: %s format requires string input.");
            }
        } else if (format_type == 'd' || format_type == 'i') {
            if (elem_type == element::i32) {
                auto int_val = value_const->get_data_ptr<int32_t>();
                replacement = std::to_string(int_val[0]);
            } else if (elem_type == element::i64) {
                auto int_val = value_const->get_data_ptr<int64_t>();
                replacement = std::to_string(int_val[0]);
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false,
                                          "aten::percentFormat: %d/%i format requires integer input.");
            }
        } else if (format_type == 'f' || format_type == 'F') {
            if (elem_type == element::f32) {
                auto float_val = value_const->get_data_ptr<float>();
                int precision = 6;
                size_t dot_pos = format_spec.find('.');
                if (dot_pos != std::string::npos) {
                    std::string prec_str = format_spec.substr(dot_pos + 1, spec_end - dot_pos - 1);
                    if (!prec_str.empty()) {
                        precision = std::stoi(prec_str);
                    }
                }
                
                char buffer[64];
                snprintf(buffer, sizeof(buffer), ("%." + std::to_string(precision) + "f").c_str(), float_val[0]);
                replacement = buffer;
            } else if (elem_type == element::f64) {
                auto float_val = value_const->get_data_ptr<double>();
                
                int precision = 6;
                size_t dot_pos = format_spec.find('.');
                if (dot_pos != std::string::npos) {
                    std::string prec_str = format_spec.substr(dot_pos + 1, spec_end - dot_pos - 1);
                    if (!prec_str.empty()) {
                        precision = std::stoi(prec_str);
                    }
                }
                
                char buffer[64];
                snprintf(buffer, sizeof(buffer), ("%." + std::to_string(precision) + "f").c_str(), float_val[0]);
                replacement = buffer;
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false,
                                          "aten::percentFormat: %f format requires float input.");
            }
        } else {
            PYTORCH_OP_CONVERSION_CHECK(false,
                                      "aten::percentFormat: unsupported format specifier %" + 
                                      std::string(1, format_type));
        }
        
        result.replace(pos, spec_end - pos + 1, replacement);
        pos += replacement.length();
        value_idx++;
    }
    
    PYTORCH_OP_CONVERSION_CHECK(value_idx == value_list.size(),
                                "aten::percentFormat: not all arguments were used in format string.");
    
    auto result_const = context.mark_node(v0::Constant::create(element::string, Shape{}, {result}));
    return {result_const};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
