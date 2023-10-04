// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gather.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include <string>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {
struct GatherShapeParams {
    InputShape inputShapes;
    InputShape targetShapes;
    int axis;
    int batch_dims;
};

typedef std::tuple<
        GatherShapeParams,
        ElementType,                     // Network precision
        bool,                            // Is const Indices
        bool                             // Is const Axis
> GatherGPUTestParams;


class GatherGPUTest : public testing::WithParamInterface<GatherGPUTestParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherGPUTestParams> obj) {
        GatherShapeParams Shapes;
        ElementType netPrecision;
        bool isIndicesConstant;
        bool isAxisConstant;

        std::tie(Shapes, netPrecision, isIndicesConstant, isAxisConstant) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({Shapes.inputShapes.first}) << "_";
        for (size_t i = 0lu; i < Shapes.inputShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(Shapes.inputShapes.second[i]) << "_";
            result << "}_";
        }
        result << "TS=(";
        result << ov::test::utils::partialShape2str({Shapes.targetShapes.first}) << "_";
        for (size_t i = 0lu; i < Shapes.targetShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(Shapes.targetShapes.second[i]) << "_";
            result << "}_";
        }
        result << "axis=" << Shapes.axis << "_";
        result << "batchDims=" << Shapes.batch_dims << "_";
        result << "netPrc=" << netPrecision << "_";
        result << "constIdx=" << (isIndicesConstant ? "True" : "False") << "_";
        result << "constAx=" << (isAxisConstant ? "True" : "False") << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        GatherShapeParams Shapes;
        ElementType netPrecision;
        bool isAxisConstant;
        bool isIndicesConstant;
        const ElementType intInputsPrecision = ElementType::i32;

        std::tie(Shapes, netPrecision, isIndicesConstant, isAxisConstant) = this->GetParam();
        const int axis = Shapes.axis;
        const int batchDims = Shapes.batch_dims;
        targetDevice = ov::test::utils::DEVICE_GPU;
        std::shared_ptr<ov::Node> indicesNode;
        std::shared_ptr<ov::Node> gatherNode;
        std::shared_ptr<ov::Node> axisNode;

        if (isIndicesConstant) {
            init_input_shapes({Shapes.inputShapes});
        } else { // Not being tested because currently parameter targetshape is not supported
            init_input_shapes({Shapes.inputShapes, Shapes.targetShapes});
        }

        ngraph::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};
        params.back()->set_friendly_name("data");

        if (isIndicesConstant) {
            auto dimsize = Shapes.inputShapes.second[0].size();
            int64_t idx_range = INT64_MAX;
            auto axis_norm = axis < 0 ? axis + dimsize : axis;
            for (size_t i = 0; i < Shapes.inputShapes.second.size(); ++i) {
                idx_range = std::min(static_cast<int64_t>(Shapes.inputShapes.second[i][axis_norm]), idx_range);
            }
            indicesNode = ngraph::builder::makeConstant<int64_t>(
                ngraph::element::i64,
                Shapes.targetShapes.second[0],
                {},
                true,
                idx_range - 1,
                0);
        } else {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(intInputsPrecision, inputDynamicShapes[1]));
            params.back()->set_friendly_name("indices");
        }

        if (isAxisConstant) {
            axisNode = ngraph::builder::makeConstant<int64_t>(intInputsPrecision, ov::Shape({1}), {axis});
        } else {
            inputDynamicShapes.push_back({1});
            for (size_t i = 0lu; i < targetStaticShapes.size(); i++) {
                targetStaticShapes[i].push_back({1});
            }
            params.push_back(std::make_shared<ov::op::v0::Parameter>(intInputsPrecision, inputDynamicShapes[2]));
            params.back()->set_friendly_name("axis");
        }

        auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));

        gatherNode = std::make_shared<ov::op::v7::Gather>(paramOuts[0],
                                                          isIndicesConstant ? indicesNode : paramOuts[1],
                                                          isAxisConstant    ? axisNode
                                                                            : isIndicesConstant ? paramOuts[1]
                                                                            : paramOuts[2],
                                                          batchDims);
        ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(gatherNode)};
        function = std::make_shared<ngraph::Function>(results, params, "Gather");
    }
};

TEST_P(GatherGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ov::element::Type_t> netPrecisions = {
    ov::element::f32,
    ov::element::i32,
    ov::element::i64,
    ov::element::i8
};

const std::vector<GatherShapeParams> dynamicInputShapeConstTargetShape = {
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1}), {{2, 3, 4}, {3, 4, 5}}),
        ov::test::InputShape(ov::PartialShape({}), {{2}}),
        1, 0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{1, 2, 3, 4}, {1, 3, 4, 5}}),
        ov::test::InputShape(ov::PartialShape({}), {{}}),
        2, 0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{2, 7, 8, 9}, {2, 7, 4, 8}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        2, 1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{2, 1, 3, 3}, {2, 1, 10, 11}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        3, 2
    },
    {
        ov::test::InputShape(ov::PartialShape({8, -1, -1, 2}), {{8, 2, 3, 2}, {8, 4, 5, 2}}),
        ov::test::InputShape(ov::PartialShape({}), {{}}),
        0, 0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{2, 6, 7, 8, 9}, {2, 6, 9, 1, 2}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 6}}),
        3, 1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{2, 4, 2, 2, 3}, {2, 4, 8, 9, 10}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 4}}),
        2, 1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{3, 4, 2, 2, 3}, {3, 4, 8, 9, 10}}),
        ov::test::InputShape(ov::PartialShape({}), {{3, 4, 3}}),
        3, 2
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{2, 4, 2, 2, 3}, {2, 4, 8, 9, 10}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 4}}),
        2, 2
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1, -1}), {{2, 4, 2, 3, 1, 3}, {2, 4, 7, 8, 9, 10}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 4}}),
        2, 2
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_input_shapes_const_target_shapes, GatherGPUTest,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapeConstTargetShape),    // input shapes
                    ::testing::ValuesIn(netPrecisions),          // network precision
                    ::testing::Values(true),                     // is const indices
                    ::testing::Values(true)),                    // is const axis
                GatherGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
