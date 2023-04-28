// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset10.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "test_common.hpp"
#include "tf_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;
using namespace ov::frontend::tensorflow::tests;

TEST_F(TransformationTestsF, SavedModelProgramOnly) {
    { model = convert_model("saved_model_program-only"); }
    {
        // create a reference graph
        auto x = make_shared<Constant>(element::f32, Shape{2, 3}, vector<float>{1, 2, 3, 3, 2, 1});
        auto y = make_shared<Parameter>(element::f32, Shape{1});
        auto add = make_shared<Add>(x, y);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{y});
    }
}

TEST_F(TransformationTestsF, SavedModelVariables) {
    { model = convert_model("saved_model_variables"); }
    {
        // create a reference graph
        auto x = make_shared<Parameter>(element::f32, Shape{1});
        auto y = make_shared<Constant>(element::f32, Shape{}, vector<float>{123});
        auto multiply = make_shared<Multiply>(x, y);

        model_ref = make_shared<Model>(OutputVector{multiply}, ParameterVector{x});
    }
}

TEST_F(TransformationTestsF, SavedModelWithInputIntegerType) {
    {
        model = convert_model("saved_model_with_gather",
                              nullptr,
                              {"params", "indices"},
                              {},
                              {PartialShape{10, 5}, PartialShape{3}});
    }
    {
        // create a reference graph
        auto params = make_shared<Parameter>(element::f32, Shape{10, 5});
        auto indices = make_shared<Parameter>(element::i32, Shape{3});
        auto gather_axis = make_shared<Constant>(element::i32, Shape{}, 0);
        auto gather = make_shared<Gather>(params, indices, gather_axis);

        auto const_mul = make_shared<Constant>(element::f32, Shape{}, 5);
        auto mul = make_shared<Multiply>(gather, const_mul);

        model_ref = make_shared<Model>(OutputVector{mul}, ParameterVector{params, indices});
    }
}
