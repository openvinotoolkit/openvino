// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/segment_max.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/segment_max.hpp"

namespace ov {
namespace test {

std::string SegmentMaxLayerTest::getTestCaseName(const testing::TestParamInfo<SegmentMaxLayerTestParams>& obj) {
    const auto& [segmentMaxParams, inputPrecision, targetDevice] = obj.param;
    const auto& [dataInputShape, segmentIdsValues, numSegments, fillMode] = segmentMaxParams;

    std::ostringstream result;
    result << inputPrecision << "_IS=";
    result << ov::test::utils::partialShape2str({dataInputShape.first}) << "_";
    result << "TS=";
    result << "(";
    for (const auto& targetShape : dataInputShape.second) {
        result << ov::test::utils::vec2str(targetShape) << "_";
    }
    result << ")_";
    result << "segIds=" << ov::test::utils::vec2str(segmentIdsValues) << "_";
    result << "numSeg=" << numSegments << "_";
    result << "fillMode=" << static_cast<int>(fillMode) << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void SegmentMaxLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    const auto dataPrecision = funcInputs[0].get_element_type();

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = -10;
    in_data.range = 20;
    const auto dataTensor = ov::test::utils::create_and_fill_tensor(dataPrecision, targetInputStaticShapes[0], in_data);
    inputs.insert({funcInputs[0].get_node_shared_ptr(), dataTensor});
}

void SegmentMaxLayerTest::SetUp() {
    const auto& [segmentMaxParams, inputPrecision, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    const auto& [dataInputShape, segmentIdsValues, numSegments, fillMode] = segmentMaxParams;

    init_input_shapes({dataInputShape});

    auto dataParam = std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[0]);
    ov::ParameterVector params{dataParam};

    // segment_ids is always a constant in these tests
    auto segmentIdsConst = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{segmentIdsValues.size()}, segmentIdsValues);

    std::shared_ptr<ov::op::v16::SegmentMax> segmentMax;
    if (numSegments >= 0) {
        auto numSegmentsConst = std::make_shared<ov::op::v0::Constant>(
            ov::element::i32, ov::Shape{}, numSegments);
        segmentMax = std::make_shared<ov::op::v16::SegmentMax>(dataParam, segmentIdsConst, numSegmentsConst, fillMode);
    } else {
        segmentMax = std::make_shared<ov::op::v16::SegmentMax>(dataParam, segmentIdsConst, fillMode);
    }

    function = std::make_shared<ov::Model>(segmentMax->outputs(), params, "SegmentMax");
}

const std::vector<SegmentMaxSpecificParams> SegmentMaxLayerTest::GenerateParams() {
    const std::vector<SegmentMaxSpecificParams> params = {
        // 1D simple, FillMode::ZERO
        SegmentMaxSpecificParams{InputShape{{}, {{4}}},
                                 {0, 0, 2, 2},
                                 3,
                                 ov::op::FillMode::ZERO},
        // 1D simple, FillMode::LOWEST
        SegmentMaxSpecificParams{InputShape{{}, {{4}}},
                                 {0, 0, 2, 2},
                                 3,
                                 ov::op::FillMode::LOWEST},
        // 2D data
        SegmentMaxSpecificParams{InputShape{{}, {{4, 3}}},
                                 {0, 0, 1, 1},
                                 -1,  // no num_segments
                                 ov::op::FillMode::ZERO},
        // 2D data with empty segments
        SegmentMaxSpecificParams{InputShape{{}, {{6, 4}}},
                                 {0, 0, 3, 3, 5, 5},
                                 7,
                                 ov::op::FillMode::LOWEST},
        // Single element per segment
        SegmentMaxSpecificParams{InputShape{{}, {{4, 2}}},
                                 {0, 1, 2, 3},
                                 4,
                                 ov::op::FillMode::ZERO},
        // All elements in one segment
        SegmentMaxSpecificParams{InputShape{{}, {{5}}},
                                 {0, 0, 0, 0, 0},
                                 -1,
                                 ov::op::FillMode::ZERO},
        // numSegments > max(segment_ids) + 1
        SegmentMaxSpecificParams{InputShape{{}, {{4}}},
                                 {0, 1, 2, 3},
                                 8,
                                 ov::op::FillMode::LOWEST},
        // 3D data
        SegmentMaxSpecificParams{InputShape{{}, {{4, 2, 3}}},
                                 {0, 0, 1, 1},
                                 2,
                                 ov::op::FillMode::ZERO},
    };
    return params;
}

}  // namespace test
}  // namespace ov
