// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bound_evaluate.hpp"

#include <gtest/gtest.h>

#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset10.hpp"
#include "type_prop.hpp"

using namespace ov;
using namespace ov::opset10;

class EvaluateBoundTest : public TypePropOpTest<op::util::FrameworkNode> {
protected:
    void SetUp() override {
        const auto c_inputs =
            ParameterVector(2, std::make_shared<Parameter>(element::f32, PartialShape{-1, 24, {1, -1}, {1, -1}}));
        const auto c = std::make_shared<Concat>(OutputVector(c_inputs.begin(), c_inputs.end()), 1);
        const auto s = std::make_shared<ShapeOf>(c);
        const auto s_res = std::make_shared<Result>(s);
        const auto body = std::make_shared<Model>(OutputVector{s_res}, c_inputs);

        fn_op = make_op(OutputVector{s}, 0, 2);

        auto attrs = op::util::FrameworkNodeAttrs();
        attrs.set_type_name("some_type");
        fn_op->set_attrs(attrs);
        fn_op->set_function(0, body);
        fn_op->set_function(1, body);
    }

    std::shared_ptr<op::util::FrameworkNode> fn_op;
};

// Simulate scenarios in pytorch when eval bounds for node PtFrameworkNode which inherits from FrameworkNode
TEST_F(EvaluateBoundTest, no_exception_when_node_has_output_with_dynamic_rank) {
    fn_op->set_output_type(0, element::i32, PartialShape::dynamic());
    fn_op->set_output_type(1, element::i32, PartialShape{{1, 4}});
    fn_op->validate_and_infer_types();

    EXPECT_NO_THROW(evaluate_both_bounds(fn_op));
}

TEST_F(EvaluateBoundTest, no_exception_when_node_has_output_with_dynamic_element_type) {
    fn_op->set_output_type(0, element::dynamic, PartialShape{4});
    fn_op->set_output_type(1, element::dynamic, PartialShape{4});
    fn_op->validate_and_infer_types();

    EXPECT_NO_THROW(evaluate_both_bounds(fn_op));
}
