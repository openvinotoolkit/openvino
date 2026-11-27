// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// Multi-scale deformable attention geometry: feature level sizes {h, w},
// batch, queries, heads, channels per head and sampling points per level.
struct MSDAShapes {
    std::vector<std::pair<size_t, size_t>> levels;
    size_t batch;
    size_t queries;
    size_t heads;
    size_t embed;
    size_t points;
};

// How the exported graph takes the keys and the locations of a level:
// VariadicSplit outputs of value and a scalar Gather index (Deformable-DETR,
// GroundingDINO, RT-DETR exports), or StridedSlice slices of value and a [1]
// shaped Gather index followed by Squeeze.
enum class MSDAForm { VariadicSplit, StridedSlice };

// Shapes, form, inference precision and target device.
using MSDAPatternParams = std::tuple<MSDAShapes, MSDAForm, ov::element::Type, std::string>;

// GridSample based multi-scale deformable attention that the plugin fuses into
// a single MSDA primitive.
class MSDAPattern : public SubgraphBaseTest, public testing::WithParamInterface<MSDAPatternParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MSDAPatternParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void validate() override;
};

}  // namespace test
}  // namespace ov
