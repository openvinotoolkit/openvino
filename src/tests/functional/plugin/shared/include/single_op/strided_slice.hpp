// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
struct StridedSliceSpecificParams {
    std::vector<InputShape> input_shape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    std::vector<int64_t> begin_mask;
    std::vector<int64_t> end_mask;
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_axis_mask;
};

using StridedSliceParams = std::tuple<
        StridedSliceSpecificParams,
        ov::element::Type,              // Model type
        ov::test::TargetDevice          // Device name
>;

class StridedSliceLayerTest : public testing::WithParamInterface<StridedSliceParams>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
