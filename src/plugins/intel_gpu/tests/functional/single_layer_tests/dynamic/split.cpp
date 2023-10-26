// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/select.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include <string>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        size_t,                    // Num splits
        int64_t,                   // Axis
        ElementType,               // Net precision
        InputShape,                // Input shapes
        std::vector<size_t>        // Used outputs indices
> splitDynamicGPUTestParams;

class SplitLayerGPUDynamicTest : public testing::WithParamInterface<splitDynamicGPUTestParams>,
                          virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<splitDynamicGPUTestParams> obj) {
        std::ostringstream result;
        size_t numSplits;
        int64_t axis;
        ElementType netPrecision;
        InputShape inputShape;
        std::vector<size_t> outIndices;
        std::tie(numSplits, axis, netPrecision, inputShape, outIndices) = obj.param;

        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "numSplits=" << numSplits << "_";
        result << "axis=" << axis << "_";
        if (!outIndices.empty()) {
            result << "outIndices" << ov::test::utils::vec2str(outIndices) << "_";
        }
        result << "netPRC=" << netPrecision << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        int64_t axis;
        size_t numSplits;
        InputShape inputShape;
        std::vector<size_t> outIndices;
        ElementType netPrecision;
        std::tie(numSplits, axis, netPrecision, inputShape, outIndices) = this->GetParam();
        if (outIndices.empty()) {
            for (size_t i = 0; i < numSplits; ++i) {
                outIndices.push_back(i);
            }
        }
        init_input_shapes({inputShape});
        ov::ParameterVector dyn_params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};
        auto paramOuts =
            ngraph::helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(dyn_params));
        auto split = std::dynamic_pointer_cast<ngraph::opset5::Split>(
                     ngraph::builder::makeSplit(paramOuts[0], netPrecision, numSplits, axis));
        ngraph::ResultVector results;
        for (size_t i = 0; i < outIndices.size(); i++) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(outIndices[i])));
        }
        function = std::make_shared<ngraph::Function>(results, dyn_params, "split");
    }
};

TEST_P(SplitLayerGPUDynamicTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

const std::vector<InputShape> inputShapes4d = {
        {
            {-1, -1, -1, -1}, {{1, 4, 5, 7}, {3, 8, 5, 9}, {5, 16, 1, 8}}
        }
};

const std::vector<InputShape> inputShapes5d = {
        {
            {-1, -1, -1, -1, -1}, {{10, 20, 30, 40, 10}, {5, 18, 3, 10, 10}, {3, 10, 6, 2, 4}}
        }
};

const std::vector<InputShape> inputShapes6d = {
        {
            {-1, -1, -1, -1, -1, -1}, {{10, 32, 3, 4, 12, 24}, {5, 2, 3, 1, 32, 12}, {3, 1, 6, 2, 4, 18}}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_SplitsCheck4Dr, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(2),                                       // nSplits
                                ::testing::Values(1),                                       // axes
                                ::testing::Values(ElementType::f16),                         // netPrec
                                ::testing::ValuesIn(inputShapes4d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // outIndices
                        SplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SplitsCheck5D, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(3),                                       // nSplits
                                ::testing::Values(2),                                       // axes
                                ::testing::Values(ElementType::f32),                         // netPrec
                                ::testing::ValuesIn(inputShapes5d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // outIndices
                        SplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SplitsCheck6D, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(4),                                       // nSplits
                                ::testing::Values(4),                                       // axes
                                ::testing::Values(ElementType::i8),                         // netPrec
                                ::testing::ValuesIn(inputShapes6d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // outIndices
                        SplitLayerGPUDynamicTest::getTestCaseName);

typedef std::tuple<
        int64_t,                            // Axis
        std::vector<int32_t>,               // SplitLength
        ElementType,                        // Net precision
        InputShape,                         // Input shapes
        ngraph::helpers::InputLayerType     // input type of splitLength
> varSplitDynamicGPUTestParams;

class VariadicSplitLayerGPUDynamicTest : public testing::WithParamInterface<varSplitDynamicGPUTestParams>,
                          virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<varSplitDynamicGPUTestParams> obj) {
        std::ostringstream result;
        int64_t axis;
        std::vector<int32_t> splitLength;
        ElementType netPrecision;
        InputShape inputShape;
        ngraph::helpers::InputLayerType inputType;
        std::tie(axis, splitLength, netPrecision, inputShape, inputType) = obj.param;

        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "SplitLen=" << ov::test::utils::vec2str(splitLength) << "_";
        result << "axis=" << axis << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "restInputType=" << inputType << "_";
        return result.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 1) {
                tensor = ov::Tensor(ov::element::i64, targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                for (size_t i = 0; i < splitLength_vec.size(); i++) {
                    dataPtr[i] = splitLength_vec[i];
                }
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

protected:
    std::vector<int32_t> splitLength_vec;
    size_t inferRequestNum = 0;
    ElementType netPrecision;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        int64_t axis;
        InputShape inputShape;
        std::vector<int32_t> splitLength;
        ngraph::helpers::InputLayerType inputType;
        std::tie(axis, splitLength, netPrecision, inputShape, inputType) = this->GetParam();

        splitLength_vec = splitLength;

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(inputShape);
        if (inputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(splitLength.size())},
                                  std::vector<ov::Shape>(inputShape.second.size(), {splitLength.size()})));
        }
        init_input_shapes(inputShapes);

        ov::ParameterVector dyn_params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};
        auto paramOuts = ngraph::helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(dyn_params));

        auto splitAxisOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{}, std::vector<int64_t>{static_cast<int64_t>(axis)});

        std::shared_ptr<ov::Node> splitLengthOp;
        if (inputType == ngraph::helpers::InputLayerType::PARAMETER) {
            auto splitLengthNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::i64, ov::Shape{splitLength.size()});
            dyn_params.push_back(splitLengthNode);
            splitLengthOp = splitLengthNode;
        } else {
            splitLengthOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{splitLength.size()}, splitLength);
        }

        auto varSplit = std::make_shared<ngraph::opset3::VariadicSplit>(paramOuts[0], splitAxisOp, splitLengthOp);
        ngraph::ResultVector results;
        for (size_t i = 0; i < splitLength.size(); i++) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(varSplit->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, dyn_params, "varSplit");
    }
};

TEST_P(VariadicSplitLayerGPUDynamicTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

const std::vector<ngraph::helpers::InputLayerType> restInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplitsCheck4D, VariadicSplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(1),                                       // axes
                                ::testing::Values(std::vector<int32_t>{2, 1, -1}),          // splitLength
                                ::testing::Values(ElementType::f16),                        // netPrec
                                ::testing::ValuesIn(inputShapes4d),                         // inShapes
                                ::testing::ValuesIn(restInputTypes)),                       // input type of splitLength
                        VariadicSplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplitsCheck5D, VariadicSplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(2),                                       // axes
                                ::testing::Values(std::vector<int32_t>{2, -1}),             // splitLength
                                ::testing::Values(ElementType::f32),                        // netPrec
                                ::testing::ValuesIn(inputShapes5d),                         // inShapes
                                ::testing::ValuesIn(restInputTypes)),                       // input type of splitLength
                        VariadicSplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplitsCheck6D, VariadicSplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(5),                                       // nSplits
                                ::testing::Values(std::vector<int32_t>{2, 3, 2, -1}),       // splitLength
                                ::testing::Values(ElementType::i8),                         // netPrec
                                ::testing::ValuesIn(inputShapes6d),                         // inShapes
                                ::testing::ValuesIn(restInputTypes)),                       // input type of splitLength
                        VariadicSplitLayerGPUDynamicTest::getTestCaseName);


const std::vector<InputShape> inputShapes4d_static = {
        {
            {5, 16, 10, 8}, {{5, 16, 10, 8}, }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplitsCheck4D_static_input_dyn_output, VariadicSplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(1),                                       // axes
                                ::testing::Values(std::vector<int32_t>{2, 1, -1}),          // splitLength
                                ::testing::Values(ElementType::f16),                        // netPrec
                                ::testing::ValuesIn(inputShapes4d_static),                         // inShapes
                                ::testing::ValuesIn(restInputTypes)),                       // input type of splitLength
                        VariadicSplitLayerGPUDynamicTest::getTestCaseName);

} // namespace GPULayerTestsDefinitions
