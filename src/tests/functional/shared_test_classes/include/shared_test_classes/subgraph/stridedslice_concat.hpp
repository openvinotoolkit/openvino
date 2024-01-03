// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<int64_t>,       // Input shape
        std::vector<int64_t>,       // Begin
        std::vector<int64_t>,       // End
        std::vector<int64_t>,       // Strides
        std::vector<int64_t>,       // Begin mask
        std::vector<int64_t>        // End mask
> StridedSliceParams;

typedef std::tuple<
        InferenceEngine::Precision,          // Network Precision
        std::string,                         // Target Device
        std::map<std::string, std::string>,  // Configuration
        StridedSliceParams                   // StridedSlice parameters
> SliceConcatParams;

class SliceConcatTest : public testing::WithParamInterface<SliceConcatParams>,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SliceConcatParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
