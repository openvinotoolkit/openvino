// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_nonzero.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo StaticShapeNonZero::type_info;

StaticShapeNonZero::StaticShapeNonZero(const Output<Node>& input)
        : Op({input}) {
    constructor_validate_and_infer_types();
}

void StaticShapeNonZero::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1,
                          "StaticShapeNonZero must have only 1 input, provided: ",
                          get_input_size());

    const auto& arg_shape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, arg_shape.is_static(),
                          "StaticShapeNonZero doesn't support dynamic input shape");

    const auto& input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et.is_integral_number() || input_et.is_real(),
                          "StaticShapeNonZero input data type needs to be a numeric type. Got: ",
                          input_et);

    const auto total_dim_size = Dimension(shape_size(arg_shape.to_shape()));
    set_output_type(0, element::i64, {arg_shape.rank(), total_dim_size});
    set_output_type(1, element::i64, {Dimension(2)});
}

std::shared_ptr<Node> StaticShapeNonZero::copy_with_new_args(
        const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<StaticShapeNonZero>(new_args.at(0));
}

bool StaticShapeNonZero::visit_attributes(ngraph::AttributeVisitor& visitor) {
    return true;
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
