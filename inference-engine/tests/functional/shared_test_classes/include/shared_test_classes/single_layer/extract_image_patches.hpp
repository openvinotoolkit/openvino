// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using extractImagePatchesTuple = typename std::tuple<
        std::vector<size_t>,               // input shape
        std::vector<size_t>,               // kernel size
        std::vector<size_t>,               // strides
        std::vector<size_t>,               // rates
        ngraph::op::PadType,               // pad type
        InferenceEngine::Precision,        // Network precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        LayerTestsUtils::TargetDevice>;                      // Device name

class ExtractImagePatchesTest : public testing::WithParamInterface<extractImagePatchesTuple>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<extractImagePatchesTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
