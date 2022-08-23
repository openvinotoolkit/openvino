// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/node_context.hpp"

using namespace ov::frontend::tensorflow;

 ov::Any NodeContext::apply_additional_conversion_rules(const ov::Any& data,
                                                       const std::type_info& type_info) const override {
    if (data.is<EmptyList>(){
        if (type_info == typeid(std::vector<int64_t>)) {
            return std::vector<int64_t>();
        } else {
            FRONT_END_GENERAL_CHECK(false,
                                    "Could not decode empty list attribute for ",
                                    get_name(),
                                    " node. Provided type is not known.");
        }
    }
    // no conversion rules found
    return data;
}
