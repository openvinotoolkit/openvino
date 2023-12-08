// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {

using regionYoloParamsTuple = std::tuple<
        ngraph::Shape,                  // Input Shape
        size_t,                         // classes
        size_t,                         // coordinates
        size_t,                         // num regions
        bool,                           // do softmax
        std::vector<int64_t>,           // mask
        int,                            // start axis
        int,                            // end axis
        InferenceEngine::Precision,     // Network precision
        std::string>;                   // Device name

class RegionYoloLayerTest : public testing::WithParamInterface<regionYoloParamsTuple>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<regionYoloParamsTuple> &obj);

protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions
