// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/tensor_iterator.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace ov;
using namespace test;

namespace CPULayerTestsDefinitions {

using TensorIteratorParams = typename std::tuple<
        std::vector<InputShape>,                    // Input shapes
        ngraph::op::RecurrentSequenceDirection,     // Direction
        ElementType>;                               // element type


class TensorIteratorCPUTest : public testing::WithParamInterface<TensorIteratorParams>,
                              virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TensorIteratorParams> obj) {
        std::vector<InputShape> shapes;
        ngraph::op::RecurrentSequenceDirection direction;
        ElementType inType;
        std::tie(shapes, direction, inType) = obj.param;

        std::ostringstream result;
        for (size_t i = 0; i < shapes.size(); i++) {
            result << "Input" << i << "_";
            result << "IS=" << ov::test::utils::partialShape2str({shapes[i].first}) << "_";
            result << "TS=";
            for (const auto& item : shapes[i].second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "direction=" << direction << "_";
        result << "netPRC=" << inType << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> shapes;
        ngraph::op::RecurrentSequenceDirection direction;
        ElementType inType;
        std::tie(shapes, direction, inType) = this->GetParam();

        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({shapes});

        const size_t sequence_axis = 1;
        auto tensor_iterator = std::make_shared<ngraph::opset5::TensorIterator>();
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }

        ngraph::ParameterVector body_params;
        for (size_t i = 0; i < shapes.size(); i++) {
            ngraph::PartialShape shape = shapes[i].first;
            shape[sequence_axis] = 1;
            auto paramNode = std::make_shared<ngraph::opset1::Parameter>(inType, shape);
            body_params.push_back(paramNode);
        }
        auto tanh = ngraph::builder::makeActivation(body_params[0], inType, ngraph::helpers::Tanh);
        auto relu = ngraph::builder::makeActivation(body_params[1], inType, ngraph::helpers::Relu);
        auto add = std::make_shared<ngraph::opset1::Add>(tanh, relu);

        auto body = std::make_shared<ov::Model>(ngraph::OutputVector{add}, body_params, "body");
        tensor_iterator->set_function(body);

        if (direction == ngraph::op::RecurrentSequenceDirection::FORWARD) {
            tensor_iterator->set_sliced_input(body_params[0], params[0], 0, 1, 1, -1, sequence_axis);
            tensor_iterator->set_sliced_input(body_params[1], params[1], 0, 1, 1, -1, sequence_axis);
            tensor_iterator->get_concatenated_slices(add, 0, 1, 1, -1, sequence_axis);
        } else if (direction == ngraph::op::RecurrentSequenceDirection::REVERSE) {
            tensor_iterator->set_sliced_input(body_params[0], params[0], -1, -1, 1, 0, sequence_axis);
            tensor_iterator->set_sliced_input(body_params[1], params[1], -1, -1, 1, 0, sequence_axis);
            tensor_iterator->get_concatenated_slices(add, -1, -1, 1, 0, sequence_axis);
        } else {
            NGRAPH_CHECK(false, "Bidirectional case is not supported.");
        }

        function = std::make_shared<ov::Model>(ngraph::OutputVector{tensor_iterator->output(0)}, params);
    }
};

TEST_P(TensorIteratorCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ElementType> inputPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i8
};

std::vector<ngraph::op::RecurrentSequenceDirection> direction = {ngraph::op::RecurrentSequenceDirection::FORWARD,
                                                                 ngraph::op::RecurrentSequenceDirection::REVERSE};
std::vector<std::vector<InputShape>> inputs = {
    {  //first test suit
        {   //dynamic shape for first input
            {-1, -1, -1},
            {  // target static shapes
                {10, 12, 10},
                {10, 8, 10},
                {1, 8, 2},
                {5, 3, 3}
            }
        },
        {   //dynamic shape for second input
            {-1, -1, -1},
            {  // target static shapes
                {1, 12, 1},
                {1, 8, 1},
                {5, 8, 2},
                {5, 3, 3}
            }
        },
    },

    {  //second test suit
        {   //dynamic shape for first input
            {{1, 12}, 5, {1, 12}},
            {  // target static shapes
                {1, 5, 1},
                {5, 5, 5},
                {1, 5, 1},
                {5, 5, 5}
             }
        },
        {   //dynamic shape for second input
            {{1, 12}, 5, {1, 12}},
            {  // target static shapes
                {1, 5, 1},
                {1, 5, 1},
                {5, 5, 1},
                {5, 5, 5}
            }
        },
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_TensorIteratorSimple, TensorIteratorCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputs),
                                 ::testing::ValuesIn(direction),
                                 ::testing::ValuesIn(inputPrecisions)),
                         TensorIteratorCPUTest::getTestCaseName);

}  // namespace
} // namespace CPULayerTestsDefinitions
