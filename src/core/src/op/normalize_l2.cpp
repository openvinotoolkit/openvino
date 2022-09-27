// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/normalize_l2.hpp"

#include <algorithm>
#include <iterator>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::NormalizeL2);

op::v0::NormalizeL2::NormalizeL2(const Output<Node>& data, const Output<Node>& axes, float eps, EpsMode eps_mode)
    : Op({data, axes}),
      m_eps(eps),
      m_eps_mode(eps_mode) {
    constructor_validate_and_infer_types();
}

bool op::v0::NormalizeL2::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_NormalizeL2_visit_attributes);
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("eps_mode", m_eps_mode);
    return true;
}

void op::v0::NormalizeL2::validate_and_infer_types() {
    OV_OP_SCOPE(v0_NormalizeL2_validate_and_infer_types);
    auto axes_node = input_value(1).get_node_shared_ptr();
    const auto& input_pshape = get_input_partial_shape(0);
    const auto& axes_pshape = get_input_partial_shape(1);
    const auto& input_rank = input_pshape.rank();
    const auto& axes_rank = axes_pshape.rank();

    NODE_VALIDATION_CHECK(this, has_and_set_equal_bounds(input_value(1)), "Input axes must be Constant type");

    if (axes_rank.is_static()) {
        NODE_VALIDATION_CHECK(this,
                              axes_rank.get_length() <= 1,
                              "Input axes must be scalar or have rank equal to 1 (axes rank: ",
                              axes_rank,
                              ").");

        if (input_rank.is_static()) {
            const auto reduction_axes = get_reduction_axes();
            for (auto axis : reduction_axes) {
                NODE_VALIDATION_CHECK(this,
                                      static_cast<int64_t>(axis) < input_rank.get_length(),
                                      "Reduction axis (",
                                      axis,
                                      ") is out of bounds ",
                                      "(argument shape: ",
                                      input_pshape,
                                      ")");
            }
        }
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

AxisSet op::v0::NormalizeL2::get_reduction_axes() const {
    AxisSet axes;
    if (auto const_op = get_constant_from_source(input_value(1))) {
        axes = const_op->get_axis_set_val();
    }
    return axes;
}

shared_ptr<Node> op::v0::NormalizeL2::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_NormalizeL2_clone_with_new_inputs);
    if (new_args.size() != 2) {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<op::v0::NormalizeL2>(new_args.at(0), new_args.at(1), m_eps, m_eps_mode);
}
