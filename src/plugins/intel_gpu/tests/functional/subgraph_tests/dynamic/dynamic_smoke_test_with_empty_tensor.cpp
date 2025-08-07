// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>, // input shapes
        ov::element::Type, // Network precision
        std::string // Device name
> emptyTensorTestParamsSet;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::i32,
};

class EmptyTensorDynamicGPUTest : public testing::WithParamInterface<emptyTensorTestParamsSet>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<emptyTensorTestParamsSet>& obj) {
        emptyTensorTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;

        const auto& [inputShapes, netType, targetDevice] = basicParamsSet;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "NetType=" << netType << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
        auto node = funcInputs[i].get_node_shared_ptr();
        auto tensor = ov::Tensor(node->get_element_type(), targetInputStaticShapes[i]);
        if (i == 0) {
            // All zero inputs for non_zero op
            auto tensor_ptr = static_cast<int32_t*>(tensor.data());
            for (size_t j = 0; j < ov::shape_size(targetInputStaticShapes[i]); ++j) {
                tensor_ptr[j] = 0;
            }
        } else {
            // Random inputs for concat
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 80;
            in_data.resolution = 8;
            tensor = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(), targetInputStaticShapes[i], in_data);
        }
        inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
     }

     void SetUp() override {
          emptyTensorTestParamsSet basicParamsSet = this->GetParam();

          const auto& [inputShapes, netType, _targetDevice] = basicParamsSet;
          targetDevice = _targetDevice;

          init_input_shapes(inputShapes);
          const auto AllZeroData = inputDynamicShapes[0];
          const auto ConcatInputData = inputDynamicShapes[1];
          ov::ParameterVector params;
          for (auto&& shape : {AllZeroData, ConcatInputData})
              params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));

          auto nonzeroEmptyResultOp = std::make_shared<ov::op::v3::NonZero>(params[0]);

          auto convertEmptyInputOp = std::make_shared<ov::op::v0::Convert>(nonzeroEmptyResultOp, ov::element::i32);
          auto concatPartialInputEmptyOp =
              std::make_shared<ov::op::v0::Concat>(ov::NodeVector{convertEmptyInputOp, params[1], convertEmptyInputOp},
                                          1);  // partially empty input / non empty output
          auto concatEmptyInputEmptyOutputOp =
              std::make_shared<ov::op::v0::Concat>(ov::NodeVector{convertEmptyInputOp, convertEmptyInputOp, convertEmptyInputOp},
                                          1);  // all empty input/ all empty output

          std::vector<int64_t> squeezeDims = {0};
          auto squeezeDimsConst = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, squeezeDims);

          auto squeezeEmptyInputOp = std::make_shared<ov::op::v0::Squeeze>(nonzeroEmptyResultOp, squeezeDimsConst);

          auto axisNode = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape({1}), std::vector<int>{0});
          auto gatherEmptyIndicesOp = std::make_shared<ov::op::v7::Gather>(params[0], squeezeEmptyInputOp, axisNode, 0);

          auto shapeofEmptyInputOp = std::make_shared<ov::op::v3::ShapeOf>(gatherEmptyIndicesOp, ov::element::i32);

          ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(shapeofEmptyInputOp),
                                      std::make_shared<ov::op::v0::Result>(concatPartialInputEmptyOp),
                                      std::make_shared<ov::op::v0::Result>(concatEmptyInputEmptyOutputOp)};
          function = std::make_shared<ov::Model>(results, params, "result");

          auto nonzero = std::make_shared<ov::op::v3::NonZero>(params[0]);
     }
};

TEST_P(EmptyTensorDynamicGPUTest, Inference) {
    run();
}

const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    {
        // Input for NonZero
        {{ov::Dimension::dynamic()}, {{30}, {40}, {50}, {10}, {7}}},
        // Input for Concat
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 0}, {1, 8}, {1, 0}, {1, 3}, {1, 20}}}
    },
};

const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(netPrecisions), // netprec
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_empty_tensor, EmptyTensorDynamicGPUTest,
                         testParams_smoke, EmptyTensorDynamicGPUTest::getTestCaseName);
} // namespace
