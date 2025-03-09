// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/equal.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace common_translators {

OutputVector translate_equal_op(const NodeContext& node) {
    num_inputs_check(node, 2, 2, true);
    auto lhs = node.get_input(0);
    auto rhs = node.get_input(1);

    auto lhs_complex = as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto rhs_complex = as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());

    auto op_type = node.get_op_type();
    FRONT_END_OP_CONVERSION_CHECK(!(lhs_complex && !rhs_complex) && !(!lhs_complex && rhs_complex),
                                  op_type + " operation expects both operands to be of the same type.");

    // both operands are of complex type
    if (lhs_complex && rhs_complex) {
        auto lhs_data = lhs_complex->get_data();
        auto rhs_data = rhs_complex->get_data();
        auto result = make_shared<v1::Equal>(lhs_data, rhs_data);
        return {result};
    }

    // both operands are real
    auto result = make_shared<v1::Equal>(lhs, rhs);
    return {result};
};

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
