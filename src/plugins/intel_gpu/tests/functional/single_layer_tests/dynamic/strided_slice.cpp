// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/strided_slice.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

struct StridedSliceParams {
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> stride;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisAxisMask;
};

typedef std::tuple<
        InputShape,                        // Input shapes
        StridedSliceParams,
        ElementType,                       // Element type
        ngraph::helpers::InputLayerType,   // begin/end/stride input type
        std::map<std::string, std::string> // Additional network configuration
> StridedSliceLayerParamSet;

class StridedSliceLayerGPUTest : public testing::WithParamInterface<StridedSliceLayerParamSet>,
                                 virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceLayerParamSet>& obj) {
        InputShape shapes;
        StridedSliceParams params;
        ElementType elementType;
        ngraph::helpers::InputLayerType restInputType;
        TargetDevice targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapes, params, elementType, restInputType, additionalConfig) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "netPRC=" << elementType << "_";
        results << "begin=" << ov::test::utils::vec2str(params.begin) << "_";
        results << "end=" << ov::test::utils::vec2str(params.end) << "_";
        results << "stride=" << ov::test::utils::vec2str(params.stride) << "_";
        results << "begin_m=" << ov::test::utils::vec2str(params.beginMask) << "_";
        results << "end_m=" << ov::test::utils::vec2str(params.endMask) << "_";
        results << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.newAxisMask)) << "_";
        results << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.shrinkAxisMask)) << "_";
        results << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.ellipsisAxisMask)) << "_";
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
                for (size_t i = 0; i < begin.size(); i++) {
                    dataPtr[i] = static_cast<float>(begin[i]);
                }
            } else if (i == 2) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<float>();
                for (size_t i = 0; i < end.size(); i++) {
                    dataPtr[i] = static_cast<float>(end[i]);
                }
            } else if (i == 3) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<float>();
                for (size_t i = 0; i < stride.size(); i++) {
                    dataPtr[i] = static_cast<float>(stride[i]);
                }
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

protected:
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> stride;
    size_t inferRequestNum = 0;

    void SetUp() override {
        InputShape shapes;
        StridedSliceParams ssParams;
        ngraph::helpers::InputLayerType restInputType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapes, ssParams, inType, restInputType, additionalConfig) = this->GetParam();

        begin = ssParams.begin;
        end = ssParams.end;
        stride = ssParams.stride;

        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(shapes);
        if (restInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(begin.size())}, std::vector<ov::Shape>(shapes.second.size(), {begin.size()})));
            inputShapes.push_back(InputShape({static_cast<int64_t>(end.size())}, std::vector<ov::Shape>(shapes.second.size(), {end.size()})));
            inputShapes.push_back(InputShape({static_cast<int64_t>(stride.size())}, std::vector<ov::Shape>(shapes.second.size(), {stride.size()})));
        }

        init_input_shapes(inputShapes);

        auto params = ngraph::builder::makeDynamicParams(inType, {inputDynamicShapes.front()});
        // auto paramNode = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(shape));
        std::shared_ptr<ov::Node> beginInput, endInput, strideInput;
        if (restInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            auto beginNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::i64, ov::Shape{begin.size()});
            auto endNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::i64, ov::Shape{end.size()});
            auto strideNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::i64, ov::Shape{stride.size()});
            params.push_back(beginNode);
            params.push_back(endNode);
            params.push_back(strideNode);
            beginInput = beginNode;
            endInput = endNode;
            strideInput = strideNode;
        } else {
            beginInput = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ov::Shape{begin.size()}, begin);
            endInput = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ov::Shape{end.size()}, end);
            strideInput = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ov::Shape{stride.size()}, stride);
        }
        auto ss = std::make_shared<ngraph::op::v1::StridedSlice>(params[0], beginInput, endInput, strideInput, ssParams.beginMask, ssParams.endMask,
                                                                 ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);

        ngraph::ResultVector results;
        for (size_t i = 0; i < ss->get_output_size(); i++) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(ss->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, params, "StridedSlice");
    }
};

TEST_P(StridedSliceLayerGPUTest, CompareWithRefs) {
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

const std::vector<InputShape> inputShapesDynamic2D = {
        {{-1, -1},
         {{32, 20}, {16, 16}, {24, 16}}},

        {{-1, 16},
         {{16, 16}, {20, 16}, {32, 16}}},

        {{{16, 32}, {16, 32}},
         {{16, 32}, {32, 16}, {24, 24}}},
};

const std::vector<StridedSliceParams> paramsPlain2D = {
        StridedSliceParams{ { 0, 10 }, { 16, 16 }, { 1, 1 }, { 0, 0 }, { 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 2, 5 }, { 16, 8 }, { 1, 1 }, { 0, 0 }, { 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 2, 5 }, { 16, 16 }, { 1, 2 }, { 0, 1 }, { 1, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0 }, { 16, 16 }, { 2, 1 }, { 0, 0 }, { 1, 0 },  { },  { },  { } },
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Static_2D, StridedSliceLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(static_shapes_to_test_representation({{32, 20}})),
                             ::testing::ValuesIn(paramsPlain2D),
                             ::testing::ValuesIn(inputPrecisions),
                             ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                             ::testing::Values(emptyAdditionalConfig)),
                         StridedSliceLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Dynamic_2D, StridedSliceLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic2D),
                             ::testing::ValuesIn(paramsPlain2D),
                             ::testing::ValuesIn(inputPrecisions),
                             ::testing::ValuesIn(restInputTypes),
                             ::testing::Values(emptyAdditionalConfig)),
                         StridedSliceLayerGPUTest::getTestCaseName);

const std::vector<StridedSliceParams> testCasesCommon4D = {
        StridedSliceParams{ { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 1, 0, 0 }, { 1, 3, 32, 20 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 0 }, { 1, 3, 20, 20 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 1, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 20, 20 }, { 1, 5, 25, 26 }, { 1, 1, 1, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 20 }, { 1, 2, 30, 30 }, { 1, 1, 2, 1 }, { 0, 0, 0, 1 }, { 0, 1, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10 }, { 1, 3, 32, 20 }, { 1, 1, 1, 1 }, { 0, 0, 1, 1 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 1, 0, 10 }, { 1, 5, 32, 30 }, { 1, 1, 1, 1 }, { 0, 1, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 1, 2, 10 }, { 1, 5, 32, 18 }, { 1, 1, 1, 2 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10 }, { 1, 8, 32, 18 }, { 1, 2, 1, 2 },  { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
};

const std::vector<InputShape> inputShapesDynamic4D = {
        {{-1, -1, -1, -1},
         {{ 1, 5, 32, 32 }, { 2, 5, 32, 32 }, { 1, 5, 64, 64 }}},

        {{-1, 5, -1, -1},
         {{ 1, 5, 32, 32 }, { 2, 5, 32, 32 }, { 3, 5, 32, 36 }}},

        {{{1, 5}, 5, {32, 64}, {32, 64}},
         {{ 2, 5, 32, 32 }, { 1, 5, 48, 32 }, { 5, 5, 32, 32 }}},
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D, StridedSliceLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic4D),
                             ::testing::ValuesIn(testCasesCommon4D),
                             ::testing::ValuesIn(inputPrecisions),
                             ::testing::ValuesIn(restInputTypes),
                             ::testing::Values(emptyAdditionalConfig)),
                         StridedSliceLayerGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
