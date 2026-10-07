// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <map>
#include <memory>
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

std::string msda_shapes_to_string(const MSDAShapes& shapes);

// Fills value and weights with values in [-1, 1] and the sampling locations
// with values in (0, 1), so every sample lands inside its feature level.
void msda_generate_inputs(const std::shared_ptr<ov::Model>& model,
                          const std::vector<ov::Shape>& shapes,
                          std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs);

using MSDAPatternParams = std::tuple<MSDAShapes, std::string>;

// StridedSlice and Gather based formulation of multi-scale deformable
// attention: every level slices its keys from value with StridedSlice and
// takes its locations with a [1] shaped Gather index followed by Squeeze.
class MSDAPattern : public SubgraphBaseTest, public testing::WithParamInterface<MSDAPatternParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MSDAPatternParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov
