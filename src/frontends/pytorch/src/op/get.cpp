// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"


namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;



OutputVector translate_get(const NodeContext& context) {
    // aten::get(Tensor? self, Scalar key, Tensor? default=None) -> Tensor?
    num_inputs_check(context, 2, 3);
    
    auto container = context.get_input(0);
    auto key = context.get_input(1);
    Output<ov::Node> default_value;
    
    if (context.get_input_size() == 3 && !context.input_is_none(2)) {
        default_value = context.get_input(2);
    }

    if (auto dict_construct = cast_fw_node(container.get_node_shared_ptr(), "prim::DictConstruct")) {
        const auto inputs = dict_construct->input_values();
        PYTORCH_OP_CONVERSION_CHECK(inputs.size() % 2 == 0,
                                    "prim::DictConstruct: inputs number is not divisible by 2.");

        bool key_is_string = false;
        bool key_is_int = false;
        std::string str_key;
        int64_t int_key = 0;

        if (const auto key_fw = ov::as_type_ptr<ov::op::util::FrameworkNode>(key.get_node_shared_ptr())) {
            const auto& attrs = key_fw->get_attrs();
            const auto it = attrs.find("string_value");
            if (it != attrs.end()) {
                key_is_string = true;
                str_key = it->second;
            }
        }
        if (!key_is_string) {
            if (const auto constant = ov::util::get_constant_from_source(key)) {
                if (ov::is_scalar(constant->get_shape())) {
                    switch (constant->get_element_type()) {
                    case element::i64:
                        key_is_int = true;
                        int_key = constant->cast_vector<int64_t>()[0];
                        break;
                    case element::i32:
                        key_is_int = true;
                        int_key = constant->cast_vector<int32_t>()[0];
                        break;
                    case element::u64:
                        key_is_int = true;
                        int_key = static_cast<int64_t>(constant->cast_vector<uint64_t>()[0]);
                        break;
                    case element::u32:
                        key_is_int = true;
                        int_key = static_cast<int64_t>(constant->cast_vector<uint32_t>()[0]);
                        break;
                    default:
                        break;
                    }
                }
            }
        }

        PYTORCH_OP_CONVERSION_CHECK(key_is_string || key_is_int,
                                    "aten::get supports only constant int or string keys for dict inputs.");

        for (size_t i = 0; i < inputs.size(); i += 2) {
            if (key_is_string) {
                if (const auto k_fw = ov::as_type_ptr<ov::op::util::FrameworkNode>(inputs.at(i).get_node_shared_ptr())) {
                    const auto& attrs = k_fw->get_attrs();
                    const auto it = attrs.find("string_value");
                    if (it != attrs.end() && it->second == str_key) {
                        return {inputs.at(i + 1)};
                    }
                }
            } else if (key_is_int) {
                if (const auto constant = ov::util::get_constant_from_source(inputs.at(i))) {
                    if (ov::is_scalar(constant->get_shape())) {
                        int64_t dict_key = 0;
                        switch (constant->get_element_type()) {
                        case element::i64:
                            dict_key = constant->cast_vector<int64_t>()[0];
                            break;
                        case element::i32:
                            dict_key = constant->cast_vector<int32_t>()[0];
                            break;
                        case element::u64:
                            dict_key = static_cast<int64_t>(constant->cast_vector<uint64_t>()[0]);
                            break;
                        case element::u32:
                            dict_key = static_cast<int64_t>(constant->cast_vector<uint32_t>()[0]);
                            break;
                        default:
                            continue;
                        }
                        if (dict_key == int_key) {
                            return {inputs.at(i + 1)};
                        }
                    }
                }
            }
        }

        if (default_value.get_node()) {
            return {default_value};
        }
        PYTORCH_OP_CONVERSION_CHECK(false, "Key is not present in dict for aten::get.");
    }
    
    PYTORCH_OP_CONVERSION_CHECK(false, "aten::get supports only dict inputs.");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
