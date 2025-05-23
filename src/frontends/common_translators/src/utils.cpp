// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs, bool allow_complex) {
    auto num_inputs = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(num_inputs >= min_inputs,
                                  "Got less inputs ",
                                  num_inputs,
                                  " than expected ",
                                  min_inputs);
    if (!allow_complex) {
        // verify that no input is complex
        for (int i = 0; i < static_cast<int>(std::min(num_inputs, max_inputs)); ++i) {
            auto input = context.get_input(i);
            auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
            FRONT_END_OP_CONVERSION_CHECK(!complex_type_mark, "The operation doesn't allow complex type.");
        }
    }
    // Check that additional inputs are all None, otherwise raise exception
    for (auto i = max_inputs; i < num_inputs; i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected: ", i + 1);
    }
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
