// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include "openvino/op/softmax.hpp"
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include "plugin/transformations/sink_reshape.hpp"
#include "plugin/transformations/convert_convolution.hpp"
#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

using SinkReshapeParams = std::tuple<bool,                                    // add eltwise
                                     bool,                                    // add activation
                                     bool,                                    // eligible rotation
                                     bool>;                                   // eligible reshape                                  

class SinkReshapeTests : public TransformationTestsF, public WithParamInterface<SinkReshapeParams> {
public:
    static std::string get_test_case_name(testing::TestParamInfo<SinkReshapeParams> obj) {
        std::pair<ov::PartialShape, ov::Shape> input_shapes;
        bool add_eltwise;
        bool add_activation;
        bool eligible_rotataion;
        bool eligible_reshape;
        std::tie(add_eltwise, add_activation, eligible_rotataion, eligible_reshape) = obj.param;

        std::ostringstream result;
        result << ")_add_eltwise=" << add_eltwise << "_add_activationt=" << add_activation << "_eligible_rotation=" << eligible_rotataion << "_eligible_reshape=" << eligible_reshape;
        return result.str();
    }

    static std::shared_ptr<ov::Model> init_model(const bool add_eltwise,
                                                 const bool add_activation,
                                                 const bool eligible_rotation,
                                                 const bool eligible_reshape,
                                                 const bool ref) {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{0, 0};
        ov::CoordinateDiff pads_end{0, 0};
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ 2, 3, 12, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              weights_const,
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              ov::op::PadType::EXPLICIT);
        std::shared_ptr<ov::Node> reshape_input_node = conv;
        if (add_eltwise) {
            auto sub_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {1});
            reshape_input_node = std::make_shared<ov::opset1::Subtract>(reshape_input_node, sub_const);
        }

        if (add_activation) {
            reshape_input_node = std::make_shared<ov::opset1::Sigmoid>(reshape_input_node);
        }
        std::shared_ptr<ov::Model> model = nullptr;
        if (!ref) {
            auto shape = eligible_reshape ? std::vector<int>{2, 4, 100} : std::vector<int>{2, 2, 20};
            auto reshape_const = ov::opset1::Constant::create(ov::element::i32, {3}, shape);
            auto reshape = std::make_shared<ov::opset1::Reshape>(reshape_input_node, reshape_const, true);
            auto order = eligible_rotation ? std::vector<int>{0 ,2, 1} : std::vector<int>{2, 1, 0};
            auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {3}, order);
            auto transpose = std::make_shared<ov::opset1::Transpose>(reshape, transpose_const);
        
            auto softmax = std::make_shared<ov::op::v8::Softmax>(transpose);
            model = std::make_shared<ov::Model>(ov::NodeVector{softmax}, ov::ParameterVector{input});
        } else {
            auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {4}, {0, 2, 3, 1});
            auto transpose = std::make_shared<ov::opset1::Transpose>(reshape_input_node, transpose_const);
            auto reshape_const = ov::opset1::Constant::create(ov::element::i32, {3}, {2, 100, 4});
            auto reshape = std::make_shared<ov::opset1::Reshape>(transpose, reshape_const, true);
            auto softmax = std::make_shared<ov::op::v8::Softmax>(reshape);
            model = std::make_shared<ov::Model>(ov::NodeVector{softmax}, ov::ParameterVector{input});
        }
        ov::pass::Manager manager;
        manager.register_pass<ConvertConvolutionToInternal>();
        if (!ref)
            manager.register_pass<SinkReshape>();
        manager.run_passes(model);
        return model;
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        bool add_eltwise;
        bool add_activation;
        bool eligible_rotation;
        bool eligible_reshape;
        std::tie(add_eltwise, add_activation, eligible_rotation, eligible_reshape) = this->GetParam();

        model = init_model(add_eltwise, add_activation, eligible_rotation, eligible_reshape, true);
        if (!eligible_rotation || !eligible_reshape)
            model_ref = model->clone();
        else
            model_ref = init_model(add_eltwise, add_activation, eligible_rotation, eligible_reshape, false);
    }
};

TEST_P(SinkReshapeTests, CompareFunctions) {}

const std::vector<bool> add_eltwise = {false, true};
const std::vector<bool> add_activation = {false, true};
const std::vector<bool> eligible_rotation = {false, true};
const std::vector<bool> eligible_reshape = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_TransformationTests_reshape_transpose, SinkReshapeTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(add_eltwise),
                                ::testing::ValuesIn(add_activation),
                                ::testing::ValuesIn(eligible_rotation),
                                ::testing::ValuesIn(eligible_reshape)),
                            SinkReshapeTests::get_test_case_name);

TEST_F(TransformationTestsF, SinkReshapeFalsePattern) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ 2, 3, 12, 12 });
    auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 3, 3, 3 }, { 1 });
    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                            weights_const,
                                                            strides,
                                                            pads_begin,
                                                            pads_end,
                                                            dilations,
                                                            ov::op::PadType::EXPLICIT);
    auto reshape_const = ov::opset1::Constant::create(ov::element::i32, {3}, std::vector<int>{2, 4, 100});
    auto reshape = std::make_shared<ov::opset1::Reshape>(conv, reshape_const, true);
    auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {3},  std::vector<int>{0 ,2, 1});
    auto transpose = std::make_shared<ov::opset1::Transpose>(reshape, transpose_const);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(transpose);
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 2, 100, 4 }, { 1 });
    auto matmul = std::make_shared<ov::opset1::MatMul>(reshape, input2, false, false);
    model = std::make_shared<ov::Model>(ov::NodeVector{softmax, matmul}, ov::ParameterVector{input});
    ov::pass::Manager manager;
    manager.register_pass<ConvertConvolutionToInternal>();
    manager.register_pass<SinkReshape>();
    OV_ASSERT_NO_THROW(manager.run_passes(model));
    model_ref = std::make_shared<ov::Model>(ov::NodeVector{softmax, matmul}, ov::ParameterVector{input});
    ov::pass::Manager manager_ref;
    manager_ref.register_pass<ConvertConvolutionToInternal>();
    manager.run_passes(model_ref);
}
