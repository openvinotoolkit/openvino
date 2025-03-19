// Copyright (C) 2024 Intel Corporation
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
        std::vector<InputShape>,    // input shape
        ov::element::Type,          // Network precision
        std::string                 // Device name
> convStaticConcatDynamicGPUTestDynamicParamsSet;
class ConvStaticConcatDynamicGPUTestDynamic : public testing::WithParamInterface<convStaticConcatDynamicGPUTestDynamicParamsSet>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convStaticConcatDynamicGPUTestDynamicParamsSet>& obj) {
        std::vector<InputShape> inputShape;
        ov::element::Type model_type;
        std::string targetDevice;

        convStaticConcatDynamicGPUTestDynamicParamsSet basicParamsSet = obj.param;
        std::tie(inputShape, model_type, targetDevice) = basicParamsSet;

        std::ostringstream result;
        result << "IS_Dynamic=";
        result << ov::test::utils::partialShape2str({inputShape[0].first}) << "_";
        for (const auto& actual_shape : {inputShape[0].second}) {
            result << ov::test::utils::partialShape2str({actual_shape[0]}) << "_";
        }
        result << "IS_Static=";
        result << ov::test::utils::partialShape2str({inputShape[1].first}) << "_";
        for (const auto& actual_shape : {inputShape[1].second}) {
            result << ov::test::utils::partialShape2str({actual_shape[0]}) << "_";
        }
        result << "model_type=" << model_type << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inputShape;
        ov::element::Type model_type;
        convStaticConcatDynamicGPUTestDynamicParamsSet basicParamsSet = this->GetParam();
        std::tie(inputShape, model_type, targetDevice) = basicParamsSet;

        init_input_shapes(inputShape);

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        // Constant weight
        auto sh0 = inputShape[0].first[1].get_length();
        auto sh1 = inputShape[1].first[1].get_length();
        ov::PartialShape inShape1 = {sh0, sh1, 1};
        auto tensor1 = ov::test::utils::create_and_fill_tensor(model_type, inShape1.to_shape());
        std::shared_ptr<ov::Node> constantWeightOp = std::make_shared<ov::op::v0::Constant>(tensor1);
        constantWeightOp->set_friendly_name("constantWeight");

        // Static convolution
        auto convolutionOp = ov::test::utils::make_convolution(inputParams[1], constantWeightOp, model_type,
                                                                {3}, {1}, {0}, {0}, {1}, ov::op::PadType::EXPLICIT, 1);
        convolutionOp->set_friendly_name("convolution");

        // Dynamic Concat
        const auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({inputParams[0], convolutionOp}), 2);

        // Function
        auto makeFunction = [](const ov::element::Type &ngPrc, ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "Concat");
        };
        function = makeFunction(model_type, inputParams, concat);
    }
};

TEST_P(ConvStaticConcatDynamicGPUTestDynamic, Inference) {
    run();
}

const std::vector<std::vector<ov::test::InputShape>> dynInputShapes1D = {
    {
        {{1, 192, ov::Dimension::dynamic()}, {{1, 192, 1}}},
        {{1, 256, 1}, {{1, 256, 1}}},
    },
    {
        {{1, 32, ov::Dimension::dynamic()}, {{1, 32, 1}}},
        {{1, 48, 1}, {{1, 48, 1}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_static_conv_n_dynamic_concat, ConvStaticConcatDynamicGPUTestDynamic,
                        ::testing::Combine(::testing::ValuesIn(dynInputShapes1D),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvStaticConcatDynamicGPUTestDynamic::getTestCaseName);

}  // namespace
