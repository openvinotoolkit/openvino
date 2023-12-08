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

using ReorgYoloParamsTuple = typename std::tuple<
        ngraph::Shape,                  // Input Shape
        size_t,                         // stride
        InferenceEngine::Precision,     // Network precision
        std::string>;                   // Device name

class ReorgYoloLayerTest : public testing::WithParamInterface<ReorgYoloParamsTuple>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReorgYoloParamsTuple> &obj);

protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions
