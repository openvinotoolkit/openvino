// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>
// to suppress deprecated definition errors
#define IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#include "layers/gna_split_layer.hpp"

namespace {

using GetAlignedSplitSizesData = std::tuple<
    uint32_t,               // total size
    uint32_t,               // maximum split size
    uint32_t,               // alignment
    std::vector<uint32_t>   // expected sizes
>;

const std::vector<GetAlignedSplitSizesData> data = {
    GetAlignedSplitSizesData{1024, 100, 64, std::vector<uint32_t>(16, 64)},
    GetAlignedSplitSizesData{151, 100, 64, std::vector<uint32_t>{64, 64, 23}},
    GetAlignedSplitSizesData{151, 65, 32, std::vector<uint32_t>{64, 64, 23}},
    GetAlignedSplitSizesData{151, 65, 1, std::vector<uint32_t>{65, 65, 21}}
};

TEST(GetAlignedSplitSizesTest, testAlignedSplitSizes) {
    for (const auto &dataItem : data) {
        auto sizes = GNAPluginNS::GetAlignedSplitSizes(std::get<0>(dataItem), std::get<1>(dataItem),
                                                       std::get<2>(dataItem));
        ASSERT_EQ(sizes, std::get<3>(dataItem));
    }
}

} // namespace