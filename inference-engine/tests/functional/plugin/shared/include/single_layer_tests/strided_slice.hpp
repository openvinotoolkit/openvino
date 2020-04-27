// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using stridedSliceParamsTuple = typename std::tuple<
        InferenceEngine::SizeVector,       // Input shape
        std::vector<int64_t>,              // Begin
        std::vector<int64_t>,              // End
        std::vector<int64_t>,              // Stride
        std::vector<int64_t>,              // Begin mask
        std::vector<int64_t>,              // End mask
        std::vector<int64_t>,              // New axis mask
        std::vector<int64_t>,              // Shrink axis mask
        std::vector<int64_t>,              // Ellipsis axis mask
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name>;

class StridedSliceLayerTest : public testing::WithParamInterface<stridedSliceParamsTuple>,
                              public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<stridedSliceParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
