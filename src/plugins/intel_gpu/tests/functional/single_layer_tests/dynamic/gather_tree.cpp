// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gather_tree.hpp"
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
    InputShape,                         // Input tensors shape
    ngraph::helpers::InputLayerType,    // Secondary input type
    ov::element::Type_t,                // Network precision
    std::string                         // Device name
> GatherTreeGPUTestParams;

class GatherTreeLayerGPUTest : public testing::WithParamInterface<GatherTreeGPUTestParams>,
                               virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTreeGPUTestParams> &obj) {
        InputShape inputShape;
        ov::element::Type_t netPrecision;
        ngraph::helpers::InputLayerType secondaryInputType;
        std::string targetName;

        std::tie(inputShape, secondaryInputType, netPrecision, targetName) = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& item : inputShape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "secondaryInputType=" << secondaryInputType << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "trgDev=" << targetName;

        return result.str();
    }

protected:
    void SetUp() override {
        InputShape inputShape;
        ov::element::Type netPrecision;
        ngraph::helpers::InputLayerType secondaryInputType;

        std::tie(inputShape, secondaryInputType, netPrecision, targetDevice) = this->GetParam();
        InputShape parentShape{inputShape};
        InputShape::first_type maxSeqLenFirst;
        if (inputShape.first.is_dynamic()) {
            maxSeqLenFirst = {inputShape.first[1]};
        }
        InputShape::second_type maxSeqLenSecond;
        maxSeqLenSecond.reserve(inputShape.second.size());
        for (const auto& item : inputShape.second) {
            maxSeqLenSecond.emplace_back(std::initializer_list<size_t>{item[1]});
        }
        InputShape maxSeqLenShape{std::move(maxSeqLenFirst), std::move(maxSeqLenSecond)};

        init_input_shapes({inputShape, parentShape, maxSeqLenShape});

        // initialization of scalar input as it cannot be done properly in init_input_shapes
        inputDynamicShapes.push_back({});
        for (auto& shape : targetStaticShapes) {
            shape.push_back({});
        }

        std::shared_ptr<ngraph::Node> inp2;
        std::shared_ptr<ngraph::Node> inp3;
        std::shared_ptr<ngraph::Node> inp4;

        ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};
        if (ngraph::helpers::InputLayerType::PARAMETER == secondaryInputType) {
            auto param2 = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[1]);
            auto param3 = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[2]);
            auto param4 = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[3]);
            inp2 = param2;
            inp3 = param3;
            inp4 = param4;

            paramsIn.push_back(param2);
            paramsIn.push_back(param3);
            paramsIn.push_back(param4);
        } else if (ngraph::helpers::InputLayerType::CONSTANT == secondaryInputType) {
            auto maxBeamIndex = inputShape.second.front().at(2) - 1;

            inp2 = ngraph::builder::makeConstant<float>(netPrecision, inputShape.second.front(), {}, true, maxBeamIndex);
            inp3 = ngraph::builder::makeConstant<float>(netPrecision, {inputShape.second.front().at(1)}, {}, true, maxBeamIndex);
            inp4 = ngraph::builder::makeConstant<float>(netPrecision, {}, {}, true, maxBeamIndex);
        } else {
            throw std::runtime_error("Unsupported inputType");
        }

        auto operationResult = std::make_shared<ngraph::opset4::GatherTree>(paramsIn.front(), inp2, inp3, inp4);

        ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(operationResult)};
        function = std::make_shared<ngraph::Function>(results, paramsIn, "GatherTree");
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto maxBeamIndex = targetInputStaticShapes.front().at(2) - 1;
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            auto tensor =
                ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                        targetInputStaticShapes[i],
                                                        maxBeamIndex,
                                                        (i == 2 || i == 3) ? maxBeamIndex / 2 : 0);
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GatherTreeLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ov::element::Type_t> netPrecisions = {
    ov::element::f32,
    ov::element::i32
};

const std::vector<InputShape> inputDynamicShapesParameter = {
    {
        {-1, 1, -1}, {{7, 1, 10}, {8, 1, 20}}
    },
    {
        {-1, 1, {5, 10}}, {{2, 1, 7}, {5, 1, 8}}
    },
    {
        {-1, {1, 5}, 10}, {{20, 1, 10}, {17, 2, 10}}
    },
    {
        {-1, -1, -1}, {{20, 20, 15}, {30, 30, 10}}
    }
};

const std::vector<InputShape> inputDynamicShapesConstant = {
    {
        {-1, 1, -1}, {{7, 1, 10}}
    },
    {
        {-1, 1, {5, 10}}, {{2, 1, 7}}
    },
    {
        {-1, {1, 5}, 10}, {{20, 1, 10}}
    },
    {
        {-1, -1, -1}, {{20, 20, 15}}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_gathertree_parameter_compareWithRefs_dynamic, GatherTreeLayerGPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputDynamicShapesParameter),
                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GatherTreeLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_gathertree_constant_compareWithRefs_dynamic, GatherTreeLayerGPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputDynamicShapesConstant),
                            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GatherTreeLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions

