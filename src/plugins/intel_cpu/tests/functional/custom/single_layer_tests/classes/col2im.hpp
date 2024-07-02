// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Col2Im {
using Col2ImSpecificParams =  std::tuple<
        InputShape,                                         // data shape
        std::vector<int64_t>,                               // output size values
        std::vector<int64_t>,                               // kernel size values
        ov::Strides,                                        // strides
        ov::Strides,                                        // dilations
        ov::Shape,                                          // pads_begin
        ov::Shape                                           // pads_end
>;

using Col2ImLayerTestParams = std::tuple<
        Col2ImSpecificParams,
        ElementType,                                        // data precision
        ElementType,                                        // index precision
        ov::test::TargetDevice                              // device name
>;

using Col2ImLayerCPUTestParamsSet = std::tuple<
        Col2ImLayerTestParams,
        CPUSpecificParams>;

class Col2ImLayerCPUTest : public testing::WithParamInterface<Col2ImLayerCPUTestParamsSet>,
                             public SubgraphBaseTest, public CPUTestsBase {
protected:
   void SetUp() override;
   void generate_inputs();
};

const std::vector<ElementType> indexPrecisions = {
        ElementType::i32,
        ElementType::i64
};

const std::vector<Col2ImSpecificParams> col2ImParamsVector = {
    Col2ImSpecificParams {
        InputShape{{}, {{1, 12, 9}}},
        std::vector<int64_t>{4, 4},
        std::vector<int64_t>{2, 2},
        ov::Strides{1, 1},
        ov::Strides{1, 1},
        ov::Shape{0, 0},
        ov::Shape{0, 0}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{3, 12, 81}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{2, 2},
        ov::Shape{2, 2}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{12, 81}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{2, 2},
        ov::Shape{2, 2}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{3, 12, 225}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 2},
        ov::Strides{1, 1},
        ov::Strides{1, 1},
        ov::Shape{0, 0},
        ov::Shape{0, 0}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{1, 27, 49}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{3, 3},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{1, 1},
        ov::Shape{1, 1}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{1, 18, 104}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 3},
        ov::Strides{2, 1},
        ov::Strides{2, 2},
        ov::Shape{1, 0},
        ov::Shape{0, 1}
    },
    Col2ImSpecificParams {
        InputShape{{-1, -1, -1}, {{1, 12, 120}, {3, 12, 120}}},
        std::vector<int64_t>{16, 16},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 1},
        ov::Strides{2, 2},
        ov::Shape{1, 0},
        ov::Shape{0, 1}
    },
    Col2ImSpecificParams {
        InputShape{{}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{-1, 12, 324}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{-1, -1, -1}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{12, -1, -1}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{12, 12, -1}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{12, -1, 324}, {{12, 12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    },
    Col2ImSpecificParams {
        InputShape{{-1, -1}, {{12, 324}}},
        std::vector<int64_t>{32, 32},
        std::vector<int64_t>{2, 2},
        ov::Strides{2, 2},
        ov::Strides{2, 2},
        ov::Shape{3, 3},
        ov::Shape{3, 3}
    }
};
}  // namespace Col2Im
}  // namespace test
}  // namespace ov