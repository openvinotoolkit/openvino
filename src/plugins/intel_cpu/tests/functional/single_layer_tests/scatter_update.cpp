// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
using ScatterUpdateShapes = std::vector<InputShape>;
using IndicesDescription = std::pair<ov::Shape, std::vector<std::int64_t>>;
using Axis = std::int64_t;

struct ScatterUpdateLayerParams {
    ScatterUpdateShapes inputShapes; // shapes for "data" and "updates" inputs
    IndicesDescription indicesDescriprion; // indices shapes and values
    Axis axis;
};

using scatterUpdateParams = std::tuple<
    ScatterUpdateLayerParams,
    ElementType,        // input precision
    ElementType>;       // indices precision

class ScatterUpdateLayerCPUTest : public testing::WithParamInterface<scatterUpdateParams>, public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<scatterUpdateParams> obj) {
        ScatterUpdateLayerParams scatterParams;
        ElementType inputPrecision;
        ElementType idxPrecision;
        std::tie(scatterParams, inputPrecision, idxPrecision) = obj.param;
        const auto inputShapes = scatterParams.inputShapes;
        const auto indicesDescr = scatterParams.indicesDescriprion;
        const auto axis = scatterParams.axis;

        std::ostringstream result;
        result << inputPrecision << "_IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({ shape.first }) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            for (const auto& targetShape : shape.second) {
                result << ov::test::utils::vec2str(targetShape) << "_";
            }
            result << ")_";
        }
        result << "indices_shape=" << indicesDescr.first << "_indices_values=" << ov::test::utils::vec2str(indicesDescr.second)
               << "axis=" << axis << "_idx_precision=" << idxPrecision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ScatterUpdateLayerParams scatterParams;
        ElementType inputPrecision;
        ElementType idxPrecision;
        std::tie(scatterParams, inputPrecision, idxPrecision) = this->GetParam();
        const auto inputShapes = scatterParams.inputShapes;
        const auto indicesDescr = scatterParams.indicesDescriprion;
        const auto axis = scatterParams.axis;

        init_input_shapes(inputShapes);
        selectedType = makeSelectedTypeStr("unknown", inputPrecision);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inputPrecision, shape));
        }
        auto indicesNode = ngraph::opset1::Constant::create(idxPrecision, indicesDescr.first, indicesDescr.second);
        auto axis_node = ngraph::opset1::Constant::create(idxPrecision, {}, { axis });
        auto scatter = std::make_shared<ngraph::opset3::ScatterUpdate>(params[0], indicesNode, params[1], axis_node);

        function = makeNgraphFunction(inputPrecision, params, scatter, "ScatterUpdateLayerCPUTest");
    }
};

TEST_P(ScatterUpdateLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ScatterUpdate");
}

const std::vector<ScatterUpdateLayerParams> scatterParams = {
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1}, {{4, 12, 3, 11}, {7, 11, 2, 3}, {3, 9, 4, 10}}},
            {{-1, -1, -1, -1}, {{4, 8, 3, 11}, {7, 8, 2, 3}, {3, 8, 4, 10}}}
        },
        IndicesDescription{{8}, {0, 2, 4, 6, 1, 3, 5, 7}},
        Axis{1}
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1}, {{4, 12, 3, 11}, {7, 9, 1, 12}, {3, 2, 1, 9}}},
            {{-1, -1, -1, -1}, {{4, 12, 3, 8}, {7, 9, 1, 8}, {3, 2, 1, 8}}}
        },
        IndicesDescription{{8}, {0, 2, 4, 6, 1, 3, 5, 7}},
        Axis{3}
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1, -1}, {{5, 9, 10, 3, 4}, {7, 8, 11, 2, 2}, {11, 3, 12, 2, 2}}},
            {{-1, -1, -1, -1, -1, -1}, {{5, 9, 4, 2, 3, 4}, {7, 8, 4, 2, 2, 2}, {11, 3, 4, 2, 2, 2}}}
        },
        IndicesDescription{{ 4, 2 }, { 0, 2, 4, 6, 1, 3, 5, 7 }},
        Axis{2}
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{-1, -1, -1, -1, -1}, {{8, 9, 10, 3, 4}, {11, 3, 4, 3, 4}, {12, 9, 11, 2, 2}}},
            {{-1, -1, -1, -1, -1, -1}, {{4, 2, 9, 10, 3, 4}, {4, 2, 3, 4, 3, 4}, {4, 2, 9, 11, 2, 2}}}
        },
        IndicesDescription{{ 4, 2 }, { 0, 2, 4, 6, 1, 3, 5, 7 }},
        Axis{0}
    },
    ScatterUpdateLayerParams{
        ScatterUpdateShapes{
            {{{8, 12}, {3, 9}, {4, 11}, {2, 3}, {2, 4}}, {{8, 9, 10, 3, 4}, {11, 3, 4, 3, 4}, {12, 9, 11, 2, 2}}},
            {{4, 2, {3, 9}, {4, 11}, {2, 3}, {2, 4}}, {{4, 2, 9, 10, 3, 4}, {4, 2, 3, 4, 3, 4}, {4, 2, 9, 11, 2, 2}}}
        },
        IndicesDescription{{ 4, 2 }, { 0, 2, 4, 6, 1, 3, 5, 7 }},
        Axis{0}
    },
};

const std::vector<ElementType> inputPrecisions = {
    ElementType::f32,
    ElementType::i32,
};

const std::vector<ElementType> constantPrecisions = {
    ElementType::i32,
    ElementType::i64,
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, ScatterUpdateLayerCPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(scatterParams),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(constantPrecisions)),
    ScatterUpdateLayerCPUTest::getTestCaseName);
} // namespace CPULayerTestsDefinitions
