// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/convolution.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"


namespace {
using ov::test::InputShape;

typedef std::tuple<
        InputShape,                 // input shape
        ov::element::Type,          // Network precision
        std::string                 // Device name
> convReshapeFullyConnectedDynamicGPUTestDynamicParamsSet;


class ConvReshapeFullyConnectedDynamicGPUTestDynamic : public testing::WithParamInterface<convReshapeFullyConnectedDynamicGPUTestDynamicParamsSet>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convReshapeFullyConnectedDynamicGPUTestDynamicParamsSet>& obj) {
        InputShape inputShape;
        ov::element::Type model_type;
        std::string targetDevice;

        convReshapeFullyConnectedDynamicGPUTestDynamicParamsSet basicParamsSet = obj.param;
        std::tie(inputShape, model_type, targetDevice) = basicParamsSet;

        std::ostringstream result;
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        for (const auto& actual_shape : inputShape.second) {
            result << ov::test::utils::partialShape2str({actual_shape}) << "_";
        }
        result << "model_type=" << model_type << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        InputShape inputShape;
        ov::element::Type model_type;
        convReshapeFullyConnectedDynamicGPUTestDynamicParamsSet basicParamsSet = this->GetParam();
        std::tie(inputShape, model_type, targetDevice) = basicParamsSet;

        init_input_shapes({inputShape});

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto convolutionOp = ov::test::utils::make_convolution(inputParams.front(), model_type, {3, 3, 3}, {1, 1, 1}, {1, 1, 1},
                                                                 {1, 1, 1}, {1, 1, 1}, ov::op::PadType::EXPLICIT, 64);

        convolutionOp->set_friendly_name("convolution");

        std::vector<int> shape_pattern = {1, -1, 64, 1};
        auto shapePatternsNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({4}), shape_pattern);
        auto reshapeOp = std::make_shared<ov::op::v1::Reshape>(convolutionOp, shapePatternsNode, false);
        reshapeOp->set_friendly_name("reshape");

        std::vector<int> transpose_order = {0, 1, 3, 2};
        auto transposeOrderNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({4}), transpose_order);
        auto transposeOp = std::make_shared<ov::op::v1::Transpose>(reshapeOp, transposeOrderNode);
        transposeOp->set_friendly_name("transpose");

        auto convertOp1 = std::make_shared<ov::op::v0::Convert>(transposeOp, ov::element::f32);
        convertOp1->set_friendly_name("convert1");

        std::shared_ptr<ov::Node> fakequantizeOp1 =
            ov::test::utils::make_fake_quantize(convertOp1, ov::element::f32, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        fakequantizeOp1->set_friendly_name("fakequantize1");

        ov::PartialShape inShapeB = {64, 10};
        auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::i8, inShapeB.to_shape());
        std::shared_ptr<ov::Node> constantOp = std::make_shared<ov::op::v0::Constant>(tensor);
        constantOp->set_friendly_name("constant");

        auto convertOp2 = std::make_shared<ov::op::v0::Convert>(constantOp, ov::element::f32);
        convertOp2->set_friendly_name("convert2");

        std::shared_ptr<ov::Node> fakequantizeOp2 =
            ov::test::utils::make_fake_quantize(convertOp2, ov::element::f32, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
        fakequantizeOp2->set_friendly_name("fakequantize2");

        auto fullyConnectedOp = std::make_shared<ov::op::v0::MatMul>(fakequantizeOp1, fakequantizeOp2, false, false);
        auto makeFunction = [](const ov::element::Type &ngPrc, ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "fullyconnected");
        };
        function = makeFunction(model_type, inputParams, fullyConnectedOp);
    }
};

TEST_P(ConvReshapeFullyConnectedDynamicGPUTestDynamic, Inference) {
    run();
}

const std::vector<ov::test::InputShape> dynInputShapes3D = {
    {
        {ov::Dimension::dynamic(), 64, 1, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 64, 1, 1, 1}}
    },
};

const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes3D),
                                                   ::testing::Values(ov::element::f16),
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_conv_reshape_fullyconnected, ConvReshapeFullyConnectedDynamicGPUTestDynamic,
                         testParams_smoke, ConvReshapeFullyConnectedDynamicGPUTestDynamic::getTestCaseName);
}  // namespace

