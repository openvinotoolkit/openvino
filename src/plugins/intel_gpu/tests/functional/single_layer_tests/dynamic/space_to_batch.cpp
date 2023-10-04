// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/space_to_batch.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

struct SpaceToBatchParams {
    std::vector<int64_t> block;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
};

typedef std::tuple<
        InputShape,                        // Input shapes
        SpaceToBatchParams,
        ElementType,                       // Element type
        ngraph::helpers::InputLayerType,   // block/begin/end input type
        std::map<std::string, std::string> // Additional network configuration
> SpaceToBatchParamsLayerParamSet;

class SpaceToBatchLayerGPUTest : public testing::WithParamInterface<SpaceToBatchParamsLayerParamSet>,
                                 virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SpaceToBatchParamsLayerParamSet>& obj) {
        InputShape shapes;
        SpaceToBatchParams params;
        ElementType elementType;
        ngraph::helpers::InputLayerType restInputType;
        TargetDevice targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapes, params, elementType, restInputType, additionalConfig) = obj.param;

        std::ostringstream results;
        results << "IS=" <<  ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "netPRC=" << elementType << "_";
        results << "block=" << ov::test::utils::vec2str(params.block) << "_";
        results << "begin=" << ov::test::utils::vec2str(params.begin) << "_";
        results << "end=" << ov::test::utils::vec2str(params.end) << "_";
        results << "restInputType=" << restInputType << "_";
        results << "config=(";
        for (const auto& configEntry : additionalConfig) {
            results << configEntry.first << ", " << configEntry.second << ":";
        }
        results << ")";

        return results.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<float>();
                for (size_t i = 0; i < block.size(); i++) {
                    dataPtr[i] = static_cast<float>(block[i]);
                }
            } else  if (i == 2) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<float>();
                for (size_t i = 0; i < begin.size(); i++) {
                    dataPtr[i] = static_cast<float>(begin[i]);
                }
            } else if (i == 3) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<float>();
                for (size_t i = 0; i < end.size(); i++) {
                    dataPtr[i] = static_cast<float>(end[i]);
                }
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

protected:
    std::vector<int64_t> block;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    size_t inferRequestNum = 0;

    void SetUp() override {
        InputShape shapes;
        SpaceToBatchParams ssParams;
        ngraph::helpers::InputLayerType restInputType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapes, ssParams, inType, restInputType, additionalConfig) = this->GetParam();

        block = ssParams.block;
        begin = ssParams.begin;
        end = ssParams.end;

        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(shapes);
        if (restInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(block.size())}, std::vector<ov::Shape>(shapes.second.size(), {block.size()})));
            inputShapes.push_back(InputShape({static_cast<int64_t>(begin.size())}, std::vector<ov::Shape>(shapes.second.size(), {begin.size()})));
            inputShapes.push_back(InputShape({static_cast<int64_t>(end.size())}, std::vector<ov::Shape>(shapes.second.size(), {end.size()})));
        }

        init_input_shapes(inputShapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};
        std::shared_ptr<ov::Node> blockInput, beginInput, endInput;
        if (restInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            auto blockNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::i64, ov::Shape{block.size()});
            auto beginNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::i64, ov::Shape{begin.size()});
            auto endNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::i64, ov::Shape{end.size()});

            params.push_back(blockNode);
            params.push_back(beginNode);
            params.push_back(endNode);

            blockInput = blockNode;
            beginInput = beginNode;
            endInput = endNode;
        } else {
            blockInput = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ov::Shape{block.size()}, block);
            beginInput = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ov::Shape{begin.size()}, begin);
            endInput = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ov::Shape{end.size()}, end);
        }
        auto ss = std::make_shared<ngraph::op::v1::SpaceToBatch>(params[0], blockInput, beginInput, endInput);

        ngraph::ResultVector results;
        for (size_t i = 0; i < ss->get_output_size(); i++) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(ss->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, params, "SpaceToBatchFuncTest");
    }
};

TEST_P(SpaceToBatchLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

std::map<std::string, std::string> emptyAdditionalConfig;

const std::vector<ElementType> inputPrecisions = {
        ElementType::f32
};

const std::vector<ngraph::helpers::InputLayerType> restInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER
};

const std::vector<InputShape> inputShapesDynamic3D = {
        {{-1, -1, -1}, {{2, 3, 6}}},
};

const std::vector<SpaceToBatchParams> paramsPlain3D = {
        SpaceToBatchParams{ { 1, 2, 3 }, { 0, 2, 2 }, { 0, 3, 1 } },
        SpaceToBatchParams{ { 1, 4, 5 }, { 0, 4, 5 }, { 0, 9, 4 } },
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Dynamic3D, SpaceToBatchLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic3D),
                             ::testing::ValuesIn(paramsPlain3D),
                             ::testing::ValuesIn(inputPrecisions),
                             ::testing::ValuesIn(restInputTypes),
                             ::testing::Values(emptyAdditionalConfig)),
                         SpaceToBatchLayerGPUTest::getTestCaseName);


const std::vector<InputShape> inputShapesDynamic4D = {
        {{-1, -1, -1, -1}, {{2, 3, 6, 5}}},
};

const std::vector<SpaceToBatchParams> paramsPlain4D = {
        SpaceToBatchParams{ { 1, 1, 2, 3 }, { 0, 2, 2, 2 }, { 0, 3, 4, 5 } },
        SpaceToBatchParams{ { 1, 1, 4, 5 }, { 0, 2, 4, 5 }, { 0, 3, 2, 5 } },
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Dynamic4D, SpaceToBatchLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic4D),
                             ::testing::ValuesIn(paramsPlain4D),
                             ::testing::ValuesIn(inputPrecisions),
                             ::testing::ValuesIn(restInputTypes),
                             ::testing::Values(emptyAdditionalConfig)),
                         SpaceToBatchLayerGPUTest::getTestCaseName);

const std::vector<InputShape> inputShapesDynamic5D = {
        {{-1, -1, -1, -1, -1}, {{2, 3, 6, 5, 7}}},
};

const std::vector<SpaceToBatchParams> paramsPlain5D = {
        SpaceToBatchParams{ { 1, 1, 2, 3, 7 }, { 0, 2, 2, 2, 4 }, { 0, 3, 4, 5, 3 } },
        SpaceToBatchParams{ { 1, 1, 4, 5, 8 }, { 0, 2, 4, 5, 5 }, { 0, 3, 2, 5, 4 } },
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Dynamic5D, SpaceToBatchLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic5D),
                             ::testing::ValuesIn(paramsPlain5D),
                             ::testing::ValuesIn(inputPrecisions),
                             ::testing::ValuesIn(restInputTypes),
                             ::testing::Values(emptyAdditionalConfig)),
                         SpaceToBatchLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
