// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/node_context.hpp"

#include "openvino/frontend/tensorflow/special_types.hpp"

using namespace ov::frontend::tensorflow;

ov::Any NodeContext::apply_additional_conversion_rules(const ov::Any& data, const std::type_info& type_info) const {
    if (data.is<EmptyList>()) {
        if (type_info == typeid(std::vector<int64_t>))
            return std::vector<int64_t>();
        else if (type_info == typeid(std::vector<float>))
            return std::vector<float>();
        else if (type_info == typeid(std::vector<std::string>))
            return std::vector<std::string>();
        else if (type_info == typeid(std::vector<bool>))
            return std::vector<bool>();
        else if (type_info == typeid(std::vector<ov::PartialShape>))
            return std::vector<ov::PartialShape>();
        else if (type_info == typeid(std::vector<ov::element::Type>))
            return std::vector<ov::element::Type>();
        else
            FRONT_END_GENERAL_CHECK(false,
                                    "Could not decode empty list attribute for ",
                                    get_name(),
                                    " node. Provided type is not known.");
    }
    // no conversion rules found
    return data;
}
