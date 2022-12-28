// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "test_common.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset10;
using namespace ov::frontend;

namespace {
shared_ptr<Model> convert_model(const string& model_path) {
    FrontEndManager fem;
    auto front_end = fem.load_by_framework(TF_FE);
    if (!front_end) {
        throw "TensorFlow Frontend is not initialized";
    }
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_MODELS_DIRNAME) + model_path);
    auto input_model = front_end->load(model_filename);
    if (!input_model) {
        throw "Input model is not read";
    }
    auto model = front_end->convert(input_model);
    if (!model) {
        throw "Model is not converted";
    }

    return model;
}
}  // namespace

TEST(FrontEndConvertTrickyModels, undefined_input_shape) {
    shared_ptr<Model> model;
    try {
        model = convert_model("undefined_input_shape/undefined_input_shape.pb");
    } catch (std::exception& ex) {
        ASSERT_TRUE(false) << ex.what();
    }

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "x") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape::dynamic()));
        } else if (node->get_friendly_name() == "y") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        } else if (node->get_friendly_name() == "z") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape::dynamic()));
        }
    }
}

TEST(FrontEndConvertTrickyModels, model_with_output_shapes) {
    shared_ptr<Model> model;
    try {
        model = convert_model("model_with_output_shapes_attr/model_with_output_shapes_attr.pb");
    } catch (std::exception& ex) {
        ASSERT_TRUE(false) << ex.what();
    }

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "x") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        } else if (node->get_friendly_name() == "relu") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(ov::PartialShape{2, 3}));
        }
    }
}

TEST_F(TransformationTestsF, AssertAndStringTensors) {
    {
        model = convert_model("string_tensors_model/string_tensors_model.pb");
        // TODO: investigate - why we have redundant nodes after the conversion
        manager.register_pass<pass::MOCTransformations>(false);
    }
    {
        auto x = make_shared<Parameter>(f32, Shape{2, 3});
        auto y = make_shared<Parameter>(f32, Shape{2, 3});
        auto cond = make_shared<Constant>(boolean, Shape{1, 1}, std::vector<bool>{true});
        auto select = make_shared<Select>(cond, x, y);

        model_ref = make_shared<Model>(OutputVector{select}, ParameterVector{x, y});
    }
}

TEST_F(TransformationTestsF, UnsortedNodes) {
    { model = convert_model("forward_edge_model_unsorted/forward_edge_model_unsorted.pb"); }
    { model_ref = convert_model("forward_edge_model/forward_edge_model.pb"); }
}

TEST_F(TransformationTestsF, ModelWithSwishF32BodyGraph) {
    {
        model = convert_model("swish_f32/swish_f32.pb");
        // need to call shape inference since body graphs can be injected with undefined shapes
        model->validate_nodes_and_infer_types();
    }
    {
        auto x = make_shared<Parameter>(f32, Shape{1, 112, 112, 32});
        auto const_add = make_shared<Constant>(f32, Shape{}, std::vector<float>{2});
        auto add = make_shared<Add>(x, const_add);
        auto sigmoid = make_shared<Sigmoid>(add);
        auto mul = make_shared<Multiply>(add, sigmoid);
        auto sigmoid2 = make_shared<Sigmoid>(mul);

        model_ref = make_shared<Model>(OutputVector{sigmoid2}, ParameterVector{x});
    }
}
