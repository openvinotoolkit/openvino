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
        bool useNumSegments;
        ov::test::utils::InputLayerType secondaryInputType;
        std::tie(SegmentMaxPar, dataPrecision, useNumSegments, secondaryInputType, td) = basicParamsSet;

        InputShape dataShape;
        std::vector<int64_t> segmentIdsValues;
        int64_t numSegments;
        ov::op::FillMode fillMode;
        std::tie(dataShape, segmentIdsValues, numSegments, fillMode) = SegmentMaxPar;
        std::ostringstream result;

        result << ov::test::utils::partialShape2str({ dataShape.first }) << "_";
        result << "TS=";
        result << "(";
        for (const auto& targetShape : dataShape.second) {
            result << ov::test::utils::vec2str(targetShape);
        }
        result << ")";
        result << "_segmentIds=" << ov::test::utils::vec2str(segmentIdsValues);
        if (useNumSegments) {
            result << "_numSegments=" << numSegments;
        }
        result << "_dataPrecision=" << dataPrecision;
        result << "_secondaryInputType=" << secondaryInputType;
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

    const auto secondaryInputType = std::get<3>(std::get<0>(this->GetParam()));
    if (secondaryInputType == ov::test::utils::InputLayerType::PARAMETER) {
            const auto segmentIdsTensor = ov::test::utils::create_and_fill_tensor(
                ov::element::i32,
                {dataShape[0]},
                0,
                20);
            inputs.insert({funcInputs[1].get_node_shared_ptr(), segmentIdsTensor});

        const auto useNumSegments = std::get<2>(std::get<0>(this->GetParam()));
        if (useNumSegments) {
            const auto numSegmentsValue = std::get<2>(std::get<0>(std::get<0>(this->GetParam())));
            const auto numSegmentsTensor = ov::test::utils::create_and_fill_tensor(
                funcInputs[2].get_element_type(),
                {},
                static_cast<unsigned int>(numSegmentsValue));
            inputs.insert({funcInputs[2].get_node_shared_ptr(), numSegmentsTensor});
        }
    }
}

void SegmentMaxLayerCPUTest::SetUp() {
        SegmentMaxLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        SegmentMaxSpecificParams SegmentMaxParams;
        ElementType inputPrecision;
        bool useNumSegments;
        ov::test::utils::InputLayerType secondaryInputType;
        std::tie(SegmentMaxParams, inputPrecision, useNumSegments, secondaryInputType, targetDevice) = basicParamsSet;

        InputShape dataShape;
        std::vector<int64_t> segmentIdsValues;
        int64_t numSegmentsValue;
        ov::op::FillMode fillMode;
        std::tie(dataShape, segmentIdsValues, numSegmentsValue, fillMode) = SegmentMaxParams;
        const ov::test::InputShape segmentIdsShape = {
            ov::PartialShape{static_cast<ov::Dimension::value_type>(segmentIdsValues.size())}, std::vector<ov::Shape>{segmentIdsValues.size()}
            };

        std::vector<ov::test::InputShape> input_shapes = { dataShape, segmentIdsShape };
        if (useNumSegments) {
            const ov::test::InputShape numSegmentsShape = {
                ov::PartialShape{}, std::vector<ov::Shape>{ov::Shape{}}
            };
            input_shapes.emplace_back(numSegmentsShape);
        }
        init_input_shapes(input_shapes);
        auto dataParameter = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[0]);
        ov::ParameterVector params{ dataParameter };
        std::shared_ptr<ov::Node> segmentMax;
        if (secondaryInputType == ov::test::utils::InputLayerType::CONSTANT) {
            auto segmentIdsConst = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{segmentIdsValues.size()}, segmentIdsValues);
            if (useNumSegments) {
                auto numSegmentsConst = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, numSegmentsValue);
                segmentMax = std::make_shared<ov::op::v16::SegmentMax>(dataParameter, segmentIdsConst, numSegmentsConst, fillMode);
            } else {
                segmentMax = std::make_shared<ov::op::v16::SegmentMax>(dataParameter, segmentIdsConst, fillMode);
            }

        } else {
            auto segmentIdsParameter = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{segmentIdsValues.size()});
            params.push_back(segmentIdsParameter);
            if (useNumSegments) {
                auto numSegmentsParameter = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
                segmentMax = std::make_shared<ov::op::v16::SegmentMax>(dataParameter, segmentIdsParameter, numSegmentsParameter, fillMode);
                params.push_back(numSegmentsParameter);
            } else {
                segmentMax = std::make_shared<ov::op::v16::SegmentMax>(dataParameter, segmentIdsParameter, fillMode);
            }
        }
        function = makeNgraphFunction(inputPrecision, params, segmentMax, "SegmentMax");
}

TEST_P(SegmentMaxLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "SegmentMax");
}

const std::vector<ov::test::utils::InputLayerType> secondaryInputTypes = {ov::test::utils::InputLayerType::CONSTANT,
                                                                          ov::test::utils::InputLayerType::PARAMETER};

const std::vector<SegmentMaxSpecificParams> SegmentMaxParamsVector = {
    // Simple test case
    SegmentMaxSpecificParams {
        InputShape{{}, {{4}}},
        std::vector<int64_t>{0, 0, 3, 3},
        5,
        ov::op::FillMode::ZERO
    },
    // 2D data
    SegmentMaxSpecificParams {
        InputShape{{}, {{6, 6}}},
        std::vector<int64_t>{0, 0, 3, 3, 9, 10},
        3,
        ov::op::FillMode::LOWEST
    },
    // 5D data
    SegmentMaxSpecificParams {
        InputShape{{}, {{6, 6, 3, 1, 1}}},
        std::vector<int64_t>{0, 0, 3, 3, 9, 10},
        3,
        ov::op::FillMode::ZERO
    },
    // Empty data tensor
    SegmentMaxSpecificParams {
        InputShape{{}, {{5, 0}}},
        std::vector<int64_t>{0, 3, 3, 9, 10},
        4,
        ov::op::FillMode::LOWEST
    },
    // single segment
    SegmentMaxSpecificParams {
        InputShape{{}, {{5, 3}}},
        std::vector<int64_t>{0, 0, 0, 0, 0},
        4,
        ov::op::FillMode::ZERO
    },
    // numSegments = 0
    SegmentMaxSpecificParams {
        InputShape{{}, {{5, 7}}},
        std::vector<int64_t>{0, 3, 3, 9, 10},
        0,
        ov::op::FillMode::LOWEST
    },
    // Sequential dynamic inputs
    SegmentMaxSpecificParams {
        InputShape{{-1, -1}, {{5, 7}, {5, 15}, {5, 0}}},
        std::vector<int64_t>{0, 3, 3, 9, 10},
        12,
        ov::op::FillMode::ZERO
    },
    // Sequential dynamic inputs with ranges
    SegmentMaxSpecificParams {
        InputShape{{{0, 10}, {-1, 16}}, {{5, 7}, {5, 15}, {5, 0}}},
        std::vector<int64_t>{0, 3, 3, 9, 10},
        12,
        ov::op::FillMode::LOWEST
    },
};

}  // namespace SegmentMax
}  // namespace test
}  // namespace ov
