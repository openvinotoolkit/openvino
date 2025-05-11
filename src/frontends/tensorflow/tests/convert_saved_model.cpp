// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"
#include "conversion_with_reference.hpp"
#include "gtest/gtest.h"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "tf_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow::tests;

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelProgramOnly) {
    {
        model = convert_model("saved_model_program-only");

        // check tensor names in the resulted model
        unordered_set<std::string> input_tensor_names = {"y"};
        unordered_set<std::string> output_tensor_names = {"z"};
        ASSERT_EQ(model->get_results().size(), 1);
        ASSERT_TRUE(model->get_results()[0]->input_value(0).get_names() == output_tensor_names);
        ASSERT_EQ(model->get_parameters().size(), 1);
        ASSERT_TRUE(model->get_parameters()[0]->output(0).get_names() == input_tensor_names);

        // check Parameter and Result node names
        ASSERT_TRUE(model->get_parameters()[0]->get_friendly_name() == "y");
        ASSERT_TRUE(model->get_results()[0]->get_friendly_name() == "z");
    }
    {
        // create a reference graph
        auto x = make_shared<v0::Constant>(element::f32, Shape{2, 3}, vector<float>{1, 2, 3, 3, 2, 1});
        auto y = make_shared<v0::Parameter>(element::f32, Shape{1});
        auto add = make_shared<v1::Add>(x, y);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelVariables) {
    { model = convert_model("saved_model_variables"); }
    {
        // create a reference graph
        auto x = make_shared<v0::Parameter>(element::f32, Shape{1});
        auto y = make_shared<v0::Constant>(element::f32, Shape{}, vector<float>{123});
        auto multiply = make_shared<v1::Multiply>(x, y);

        model_ref = make_shared<Model>(OutputVector{multiply}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelWithInputIntegerType) {
    {
        model = convert_model("saved_model_with_gather",
                              nullptr,
                              {"params", "indices"},
                              {},
                              {PartialShape{10, 5}, PartialShape{3}});

        // check tensor names in the resulted model
        unordered_set<std::string> input_tensor_name1 = {"params"};
        unordered_set<std::string> input_tensor_name2 = {"indices"};
        unordered_set<std::string> output_tensor_names = {"test_output_name"};
        ASSERT_EQ(model->get_results().size(), 1);
        ASSERT_TRUE(model->get_results()[0]->input_value(0).get_names() == output_tensor_names);
        ASSERT_EQ(model->get_parameters().size(), 2);
        ASSERT_TRUE(model->get_parameters()[0]->output(0).get_names() == input_tensor_name1);
        ASSERT_TRUE(model->get_parameters()[1]->output(0).get_names() == input_tensor_name2);

        // check Parameter and Result node names
        ASSERT_TRUE(model->get_parameters()[0]->get_friendly_name() == "params");
        ASSERT_TRUE(model->get_parameters()[1]->get_friendly_name() == "indices");
        ASSERT_TRUE(model->get_results()[0]->get_friendly_name() == "test_output_name");
    }
    {
        // create a reference graph
        auto params = make_shared<v0::Parameter>(element::f32, Shape{10, 5});
        auto indices = make_shared<v0::Parameter>(element::i32, Shape{3});
        auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather = make_shared<v8::Gather>(params, indices, gather_axis);

        auto const_mul = make_shared<v0::Constant>(element::f32, Shape{}, 5);
        auto mul = make_shared<v1::Multiply>(gather, const_mul);

        model_ref = make_shared<Model>(OutputVector{mul}, ParameterVector{params, indices});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelMultipleTensorNames) {
    // The test aims to check tensor names of input and output tensors
    // it checks that TF FE preserved user specific names for input and output tensor
    // and exclude internal names
    {
        model = convert_model("saved_model_parameter_result");

        // check tensor names in the resulted model
        unordered_set<std::string> tensor_names = {"params", "test_output_name"};
        ASSERT_EQ(model->get_results().size(), 1);
        ASSERT_TRUE(model->get_results()[0]->input_value(0).get_names() == tensor_names);
        ASSERT_EQ(model->get_parameters().size(), 1);
        ASSERT_TRUE(model->get_parameters()[0]->output(0).get_names() == tensor_names);

        // check Parameter and Result node names
        ASSERT_TRUE(model->get_parameters()[0]->get_friendly_name() == "params");
        ASSERT_TRUE(model->get_results()[0]->get_friendly_name() == "test_output_name");
    }
    {
        // create a reference graph
        auto x = make_shared<v0::Parameter>(element::f32, Shape{20, 5});
        auto result = make_shared<v0::Result>(x);
        model_ref = make_shared<Model>(OutputVector{result}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelBroadcastIssue) {
    { model = convert_model("saved_model_broadcast_issue"); }
    {
        // create a reference graph
        auto x = make_shared<v0::Constant>(element::i64, Shape{2, 2}, vector<int64_t>{1, 2, -1, -1});

        model_ref = make_shared<Model>(OutputVector{x}, ParameterVector{});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelMultiGraph) {
    // The test verifies loading of MetaGraph with empty tags as default
    // And verifies loading variables with no corresponding RestoreV2
    { model = convert_model("saved_model_multi-graph"); }
    {
        // create a reference graph
        auto x = make_shared<v0::Constant>(element::f32, Shape{2, 3}, vector<float>{1, 2, 3, 3, 2, 1});
        auto y = make_shared<v0::Parameter>(element::f32, Shape{1});
        auto add = make_shared<v1::Add>(x, y);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelWithIntermediateOutput) {
    // The test aims to check that output from intermediate layers presented in the model signature
    // must be preserved
    {
        model = convert_model("saved_model_intermediate_output");
        ASSERT_TRUE(model->get_results().size() == 2);
    }
    {
        // create a reference graph
        auto input1 = make_shared<v0::Parameter>(element::f32, Shape{2});
        auto input2 = make_shared<v0::Parameter>(element::f32, Shape{2});
        auto add = make_shared<v1::Add>(input1, input2);
        auto sub = make_shared<v1::Subtract>(input2, add);
        auto result1 = make_shared<v0::Result>(add);
        auto result2 = make_shared<v0::Result>(sub);
        model_ref = make_shared<Model>(OutputVector{result1, result2}, ParameterVector{input1, input2});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelMMAPCompare) {
    { model = convert_model("saved_model_variables"); }
    { model_ref = convert_model("saved_model_variables", nullptr, {}, {}, {}, {}, {}, true); }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelWithNumericalNames) {
    comparator.enable(FunctionsComparator::CmpValues::TENSOR_NAMES);
    // The test aims to check that model with only numerical names for operation
    // is successfully converted
    // it is a tricky case because colision between naming input and output ports may occur
    { model = convert_model("saved_model_with_numerical_names"); }
    {
        // create a reference graph
        auto x = make_shared<v0::Parameter>(element::f32, Shape{1});
        x->output(0).set_names({"0"});
        auto y = make_shared<v0::Parameter>(element::f32, Shape{1});
        y->output(0).set_names({"1"});
        auto z = make_shared<v0::Parameter>(element::f32, Shape{1});
        z->output(0).set_names({"2"});
        auto add = make_shared<v1::Add>(x, y);
        add->output(0).set_names({"3:0"});
        auto sub = make_shared<v1::Subtract>(add, z);
        sub->output(0).set_names({"4"});
        auto result = make_shared<v0::Result>(sub);
        result->output(0).set_names({"4"});
        model_ref = make_shared<Model>(ResultVector{result}, ParameterVector{x, y, z});
    }
}
