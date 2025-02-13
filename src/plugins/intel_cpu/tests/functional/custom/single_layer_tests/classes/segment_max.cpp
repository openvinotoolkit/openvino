// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "segment_max.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace SegmentMax {
std::string SegmentMaxLayerCPUTest::getTestCaseName(testing::TestParamInfo<SegmentMaxLayerCPUTestParamsSet> obj) {
        SegmentMaxLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        SegmentMaxSpecificParams SegmentMaxPar;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType dataPrecision;
        ElementType segmentIdsPrecision;
        std::tie(SegmentMaxPar, dataPrecision, segmentIdsPrecision, td) = basicParamsSet;

        InputShape dataShape;
        std::vector<int64_t> segmentIdsValues;
        ov::op::FillMode fillMode;
        std::tie(dataShape, segmentIdsValues, fillMode) = SegmentMaxPar;
        std::ostringstream result;

        result << ov::test::utils::partialShape2str({ dataShape.first }) << "_";
        result << "TS=";
        result << "(";
        for (const auto& targetShape : dataShape.second) {
            result << ov::test::utils::vec2str(targetShape) << "_";
        }
        result << ")";
        result << "_segmentIds=" << ov::test::utils::vec2str(segmentIdsValues);
        result << "_dataPrecision=" << dataPrecision;
        result << "_segmentIdsPrecision=" << segmentIdsPrecision;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

void SegmentMaxLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const auto dataType = funcInputs[0].get_element_type();
        const auto& dataShape = targetInputStaticShapes[0];
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 10;
        const auto dataTensor = ov::test::utils::create_and_fill_tensor(dataType, dataShape, in_data);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), dataTensor});
}

void SegmentMaxLayerCPUTest::SetUp() {
        SegmentMaxLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        SegmentMaxSpecificParams SegmentMaxParams;
        ElementType inputPrecision;
        ElementType segmentIdsPrecision;
        std::tie(SegmentMaxParams, inputPrecision, segmentIdsPrecision, targetDevice) = basicParamsSet;

        InputShape dataShape;
        std::vector<int64_t> segmentIdsValues;
        ov::op::FillMode fillMode;
        std::tie(dataShape, segmentIdsValues, fillMode) = SegmentMaxParams;
        const ov::test::InputShape segmentIdsShape = {
            ov::PartialShape{static_cast<ov::Dimension::value_type>(segmentIdsValues.size())}, std::vector<ov::Shape>{segmentIdsValues.size()}
            };

        std::vector<ov::test::InputShape> input_shapes = { dataShape, segmentIdsShape };
        init_input_shapes(input_shapes);
        auto dataParameter = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[0]);
        auto segmentIdsConst = std::make_shared<ov::op::v0::Constant>(segmentIdsPrecision, ov::Shape{segmentIdsValues.size()}, segmentIdsValues);
        auto segmentMax = std::make_shared<ov::op::v16::SegmentMax>(dataParameter, segmentIdsConst, fillMode);

        ov::ParameterVector params{ dataParameter };
        function = makeNgraphFunction(inputPrecision, params, segmentMax, "SegmentMax");
}

TEST_P(SegmentMaxLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "SegmentMax");
}

const std::vector<SegmentMaxSpecificParams> SegmentMaxParamsVector = {
    SegmentMaxSpecificParams {
        InputShape{{}, {{4}}},
        std::vector<int64_t>{0, 0, 3, 3},
        ov::op::FillMode::ZERO
    },
};

}  // namespace SegmentMax
}  // namespace test
}  // namespace ov
