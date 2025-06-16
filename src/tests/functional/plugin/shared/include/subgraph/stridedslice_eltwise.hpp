// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct StridedSliceEltwiseSpecificParams {
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

using StridedSliceEltwiseParamsTuple = typename std::tuple<
        StridedSliceEltwiseSpecificParams, // strided_slice params
        ov::element::Type,          // Network precision
        std::string>;               // Device name

class StridedSliceEltwiseTest: public testing::WithParamInterface<StridedSliceEltwiseParamsTuple>,
                         virtual public ov::test::SubgraphBaseStaticTest{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceEltwiseParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
