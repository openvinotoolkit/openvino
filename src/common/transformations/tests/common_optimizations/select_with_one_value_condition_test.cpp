// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/select_with_one_value_condition.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace ov;
using namespace std;
using namespace testing;
using namespace ov::opset10;
using namespace ov::element;

namespace {
enum class SELECT_BRANCH { THEN_BRANCH, ELSE_BRANCH, NONE };

struct SelectWithOneValueConditionParams {
    PartialShape x_shape;
    PartialShape y_shape;
    Shape cond_shape;
    vector<bool> cond_values;
    PartialShape select_shape;
    SELECT_BRANCH which_branch;
};

shared_ptr<Model> gen_model(const SelectWithOneValueConditionParams& params) {
    auto x = make_shared<Parameter>(f32, params.x_shape);
    auto y = make_shared<Parameter>(f32, params.y_shape);
    auto add = make_shared<Add>(x, y);
    auto subtract = make_shared<Relu>(y);
    auto condition = make_shared<Constant>(boolean, params.cond_shape, params.cond_values);
    auto select = make_shared<Select>(condition, add, subtract);
    auto res_model = make_shared<Model>(OutputVector{select}, ParameterVector{x, y});
    return res_model;
}

shared_ptr<Model> gen_reference(const SelectWithOneValueConditionParams& params) {
    auto cond_shape = params.cond_shape;
    auto x = make_shared<Parameter>(f32, params.x_shape);
    auto y = make_shared<Parameter>(f32, params.y_shape);
    Output<Node> output;
    if (params.which_branch == SELECT_BRANCH::NONE) {
        return gen_model(params);
    } else if (params.which_branch == SELECT_BRANCH::THEN_BRANCH) {
        output = make_shared<Add>(x, y)->output(0);
    } else {
        output = make_shared<Relu>(y)->output(0);
    }

    if (!output.get_partial_shape().same_scheme(params.select_shape)) {
        vector<int32_t> select_shape_values(params.select_shape.size());
        for (size_t i = 0; i < params.select_shape.size(); ++i) {
            select_shape_values[i] = static_cast<int32_t>(params.select_shape[i].get_length());
        }

        if (select_shape_values.size() > 0) {
            auto target_shape = make_shared<Constant>(i32, Shape{select_shape_values.size()}, select_shape_values);
            output = make_shared<Broadcast>(output, target_shape)->output(0);
        }
    }

    return make_shared<Model>(OutputVector{output}, ParameterVector{x, y});
}

}  // namespace

class SelectWithOneValueConditionTest : public WithParamInterface<SelectWithOneValueConditionParams>,
                                        public TransformationTestsF {};

TEST_P(SelectWithOneValueConditionTest, SelectWithOneValueConditionTestPattern) {
    const auto& p = GetParam();
    {
        model = gen_model(p);
        manager.register_pass<pass::SelectWithOneValueCondition>();
    }

    model_ref = gen_reference(p);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

static const std::vector<SelectWithOneValueConditionParams> params = {
    SelectWithOneValueConditionParams{PartialShape{2},
                                      PartialShape{2},
                                      Shape{1},
                                      vector<bool>{false},
                                      PartialShape{2},
                                      SELECT_BRANCH::ELSE_BRANCH},
    SelectWithOneValueConditionParams{PartialShape{2},
                                      PartialShape{2},
                                      Shape{1},
                                      vector<bool>{true},
                                      PartialShape{2},
                                      SELECT_BRANCH::THEN_BRANCH},
    // with Broadcast case - Subtract result needs to be broadcasted
    SelectWithOneValueConditionParams{PartialShape{2, 2},
                                      PartialShape{2},
                                      Shape{1},
                                      vector<bool>{false},
                                      PartialShape{2, 2},
                                      SELECT_BRANCH::ELSE_BRANCH},
    // Select is not eliminated due to condition constant value
    SelectWithOneValueConditionParams{PartialShape{2, 2},
                                      PartialShape{2},
                                      Shape{2, 2},
                                      vector<bool>{false, true, false, false},
                                      PartialShape{2, 2},
                                      SELECT_BRANCH::NONE},
    // The branch is not possible to select due to dynamic output shape for Select
    SelectWithOneValueConditionParams{PartialShape{Dimension::dynamic(), 2},
                                      PartialShape{2},
                                      Shape{2},
                                      vector<bool>{true, true},
                                      PartialShape{Dimension::dynamic(), 2},
                                      SELECT_BRANCH::NONE},
    // The branch is not possible to select due to dynamic output shape for Select
    SelectWithOneValueConditionParams{PartialShape{2},
                                      PartialShape{Dimension::dynamic(), 2},
                                      Shape{2},
                                      vector<bool>{true, true},
                                      PartialShape{Dimension::dynamic(), 2},
                                      SELECT_BRANCH::NONE},
};

INSTANTIATE_TEST_SUITE_P(SelectWithOneValueConditionTest, SelectWithOneValueConditionTest, ValuesIn(params));

TEST(TransformationTests, SelectWithOneValueCondition_DontRenameParameter) {
    auto x = make_shared<Parameter>(f32, Shape{3});
    x->set_friendly_name("X");
    auto y = make_shared<Parameter>(f32, Shape{3});
    auto relu = make_shared<Relu>(y);
    auto condition = Constant::create(boolean, Shape{3}, {true, true, true});
    auto select = make_shared<Select>(condition, x, relu);
    auto abs = make_shared<Abs>(select);
    auto model = make_shared<Model>(OutputVector{abs}, ParameterVector{x, y});

    pass::Manager manager;
    manager.register_pass<pass::SelectWithOneValueCondition>();
    manager.run_passes(model);

    ASSERT_EQ(count_ops_of_type<Select>(model), 0);
    EXPECT_EQ(model->get_parameters()[0]->get_friendly_name(), "X");
}
