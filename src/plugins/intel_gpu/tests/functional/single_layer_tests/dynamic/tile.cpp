// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/tile.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using TileLayerTestParamsSet = typename std::tuple<
        std::vector<ov::test::InputShape>,     // Input shapes
        std::vector<int64_t>,                  // Repeats
        ov::element::Type_t,                   // Network precision
        bool,                                  // Is Repeats input constant
        std::string>;                          // Device name


class TileLayerGPUTest : public testing::WithParamInterface<TileLayerTestParamsSet>,
                         public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TileLayerTestParamsSet> obj) {
        TileLayerTestParamsSet basicParamsSet = obj.param;

        std::vector<ov::test::InputShape> inputShapes;
        std::vector<int64_t> repeats;
        ov::element::Type_t netPrecision;
        bool isRepeatsConst;
        std::string deviceName;
        std::tie(inputShapes, repeats, netPrecision, isRepeatsConst, deviceName) = basicParamsSet;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "Repeats=" << ov::test::utils::vec2str(repeats)  << "_";
        result << "netPrec=" << netPrecision << "_";
        result << "constRepeats=" << (isRepeatsConst ? "True" : "False") << "_";
        result << "trgDev=" << deviceName;

        return result.str();
    }

protected:
    void SetUp() override {
        TileLayerTestParamsSet basicParamsSet = this->GetParam();

        std::vector<ov::test::InputShape> inputShapes;
        ov::element::Type_t netPrecision;
        bool isRepeatsConst;
        std::tie(inputShapes, repeatsData, netPrecision, isRepeatsConst, targetDevice) = basicParamsSet;

        if (inputShapes.front().first.rank() != 0) {
            inputDynamicShapes.push_back(inputShapes.front().first);
            if (!isRepeatsConst) {
                inputDynamicShapes.push_back({ static_cast<int64_t>(repeatsData.size()) });
            }
        }
        const size_t targetStaticShapeSize = inputShapes.front().second.size();
        targetStaticShapes.resize(targetStaticShapeSize);
        for (size_t i = 0lu; i < targetStaticShapeSize; ++i) {
            targetStaticShapes[i].push_back(inputShapes.front().second[i]);
            if (!isRepeatsConst)
                targetStaticShapes[i].push_back({ repeatsData.size() });
        }

        ov::ParameterVector functionParams;
        if (inputDynamicShapes.empty()) {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, targetStaticShapes.front().front()));
        } else {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes.front()));
            if (!isRepeatsConst) {
                functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]));
                functionParams.back()->set_friendly_name("repeats");
            }
        }
        functionParams.front()->set_friendly_name("data");

        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(functionParams));
        std::shared_ptr<ov::Node> tileNode;
        if (isRepeatsConst) {
            tileNode = std::make_shared<ov::op::v0::Tile>(paramOuts[0],
                    ov::op::v0::Constant::create(ov::element::i64, { repeatsData.size() }, repeatsData));
        } else {
            tileNode = std::make_shared<ov::op::v0::Tile>(paramOuts[0], paramOuts[1]);
        }

        ngraph::ResultVector results;
        for (size_t i = 0; i < tileNode->get_output_size(); i++) {
            results.push_back(std::make_shared<ngraph::opset4::Result>(tileNode->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, functionParams, "Tile");
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_node()->get_friendly_name() == "repeats") {
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i]};
                auto data = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                for (size_t i = 0lu; i < repeatsData.size(); i++) {
                    data[i] = repeatsData[i];
                }
            } else {
                if (funcInput.get_element_type().is_real()) {
                    tensor = ov::test::utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[i], 10, 0, 1000);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    std::vector<int64_t> repeatsData;
};

TEST_P(TileLayerGPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ov::element::Type_t> netPrecisions = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes4D = {
    {
        { // Origin dynamic shapes
            {ov::Dimension(1, 20), ov::Dimension(10, 20), ov::Dimension(1, 20), ov::Dimension(1, 20)},
            { // Dynamic shapes instances
                {2, 16, 3, 4},
                {1, 16, 1, 1},
                {1, 16, 2, 3}
            }
         }
    },
    {
        { // Origin dynamic shapes
            {-1, -1, -1, -1},
            { // Dynamic shapes instances
                {3, 15, 5, 7},
                {4, 55, 8, 24}
            }
        }
    }
};

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes5D = {
    {
        { // Origin dynamic shapes
            {ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 20), ov::Dimension(1, 70)},
            { // Dynamic shapes instances
                {2, 16, 2, 3, 4},
                {1, 16, 8, 5, 4},
                {8, 1, 2, 3, 64}
            }
        }
    },
    {
        { // Origin dynamic shapes
            {-1, -1, -1, -1, -1},
            { // Dynamic shapes instances
                {2, 16, 2, 3, 4},
                {1, 16, 8, 5, 4},
                {8, 1, 2, 3, 64}
            }
        }
    }
};

const std::vector<std::vector<int64_t>> repeats4D = {
        {2, 3},
        {1, 2, 3},
        {1, 1, 1, 1},
        {1, 1, 2, 3},
        {1, 2, 1, 3},
        {2, 1, 1, 1},
        {2, 3, 1, 1}
};
const std::vector<std::vector<int64_t>> repeats5D = {
        {1, 2, 3},
        {1, 1, 2, 3},
        {1, 1, 1, 2, 3},
        {1, 2, 1, 1, 3},
        {2, 1, 1, 1, 1},
        {2, 3, 1, 1, 1}
};

INSTANTIATE_TEST_CASE_P(DynamicShape4D, TileLayerGPUTest,
                                ::testing::Combine(
                                        ::testing::ValuesIn(dynamicInputShapes4D),
                                        ::testing::ValuesIn(repeats4D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(true, false),
                                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        TileLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(DynamicShape5D, TileLayerGPUTest,
                                ::testing::Combine(
                                        ::testing::ValuesIn(dynamicInputShapes5D),
                                        ::testing::ValuesIn(repeats5D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(true, false),
                                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        TileLayerGPUTest::getTestCaseName);

} // namespace

} // namespace GPULayerTestsDefinitions
