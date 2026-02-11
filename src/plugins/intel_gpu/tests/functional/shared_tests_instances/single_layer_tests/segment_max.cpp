// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/segment_max.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {
namespace {

// Workaround: core_configuration() in core_config.cpp sets inference_precision via
// core->set_property(DEVICE_GPU, f32), but this global property does not propagate
// to dGPU (GPU.1) when --device_suffix=1 is used.  As a result, f32 SegmentMax models
// are silently compiled with f16 precision on dGPU, causing LOWEST fill values
// (-FLT_MAX) to be truncated to -HALF_MAX (-65504) and comparison failures.
// To work around this, explicitly pass inference_precision=f32 in the per-model
// configuration for f32 test cases.
class SegmentMaxLayerGPUTest : public SegmentMaxLayerTest {
    void SetUp() override {
        SegmentMaxLayerTest::SetUp();
        const auto& [segmentMaxParams, inputPrecision, targetDevice] = this->GetParam();
        if (inputPrecision == ov::element::f32) {
            configuration.insert({ov::hint::inference_precision(ov::element::f32)});
        }
    }
};

TEST_P(SegmentMaxLayerGPUTest, Inference) {
    run();
}

// 10 params × 3 precisions = 30 tests
// Coverage: Rank 1-4, ZERO/LOWEST, contiguous/gap, padding/truncation, single-element
const std::vector<SegmentMaxSpecificParams> segmentMaxParams = {
    // Rank-1: contiguous segments, ZERO
    {InputShape{{}, {{6}}}, {0, 0, 1, 1, 2, 2}, -1, ov::op::FillMode::ZERO},
    // Rank-1: discontinuous gap (segs 1,2 empty), LOWEST
    {InputShape{{}, {{4}}}, {0, 0, 3, 3}, -1, ov::op::FillMode::LOWEST},
    // Rank-1: large gaps + num_segments padding (7 > max+1=6), ZERO
    {InputShape{{}, {{6}}}, {0, 0, 0, 3, 5, 5}, 7, ov::op::FillMode::ZERO},
    // Rank-1: single element
    {InputShape{{}, {{1}}}, {0}, -1, ov::op::FillMode::ZERO},
    // Rank-1: num_segments truncation (2 < max+1=4), ZERO
    {InputShape{{}, {{5}}}, {0, 0, 2, 3, 3}, 2, ov::op::FillMode::ZERO},
    // Rank-2: gap (segment 1 empty), LOWEST
    {InputShape{{}, {{3, 4}}}, {0, 2, 2}, -1, ov::op::FillMode::LOWEST},
    // Rank-2: per-element segments + num_segments padding, LOWEST
    {InputShape{{}, {{4, 2}}}, {0, 1, 2, 3}, 6, ov::op::FillMode::LOWEST},
    // Rank-3: contiguous segments, LOWEST
    {InputShape{{}, {{4, 2, 3}}}, {0, 0, 1, 1}, 2, ov::op::FillMode::LOWEST},
    // Rank-4: gap (segments 1,2 empty), ZERO
    {InputShape{{}, {{3, 2, 2, 3}}}, {0, 3, 3}, -1, ov::op::FillMode::ZERO},
    // Rank-4: per-element segments + num_segments padding, LOWEST
    {InputShape{{}, {{3, 2, 2, 2}}}, {0, 1, 2}, 6, ov::op::FillMode::LOWEST},
};

const auto precisions = testing::Values(ElementType::f32, ElementType::f16, ElementType::i32);
const auto targetDevice = testing::Values(ov::test::utils::DEVICE_GPU);

INSTANTIATE_TEST_SUITE_P(smoke_SegmentMax,
                         SegmentMaxLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(segmentMaxParams),
                                            precisions,
                                            targetDevice),
                         SegmentMaxLayerTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
