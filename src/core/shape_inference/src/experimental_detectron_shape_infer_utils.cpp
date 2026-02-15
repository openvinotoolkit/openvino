// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_shape_infer_utils.hpp"

namespace ov {
namespace op {
namespace detectron {
namespace validate {

/**
 * @brief Validates if all op's inputs have got same floating type and return inputs shapes and element type.
 *
 * @param op   Pointer to detector operator.
 * @return Input shapes and element type as pair.
 */
std::pair<std::vector<PartialShape>, element::Type> all_inputs_same_floating_type(const Node* const op) {
    auto shapes_and_type = std::make_pair(std::vector<ov::PartialShape>(), element::dynamic);
    auto& out_et = shapes_and_type.second;
    auto& input_shapes = shapes_and_type.first;

    const auto input_size = op->get_input_size();
    input_shapes.reserve(input_size);

    for (size_t i = 0; i < input_size; ++i) {
        const auto& input_et = op->get_input_element_type(i);
        NODE_VALIDATION_CHECK(
            op,
            element::Type::merge(out_et, out_et, input_et) && (out_et.is_dynamic() || out_et.is_real()),
            "Input[",
            i,
            "] type '",
            input_et,
            "' is not floating point or not same as others inputs.");
        input_shapes.push_back(op->get_input_partial_shape(i));
    }

    return shapes_and_type;
}
}  // namespace validate
}  // namespace detectron
}  // namespace op
}  // namespace ov
