// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
    InferenceEngine::SizeVector,       // Input shapes
    size_t,                            // Axis
    std::vector<size_t>,               // Split number
    std::vector<size_t>,               // Index connected layer
    std::vector<int64_t>,              // Pad begin
    std::vector<int64_t>,              // Pad end
    ngraph::helpers::PadMode,          // Pad mode
    InferenceEngine::Precision,        // Network precision
    std::string                        // Device name
> SplitPadTuple;


class VariadicSplitPad: public testing::WithParamInterface<SplitPadTuple>,
                        virtual public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SplitPadTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace SubgraphTestsDefinitions
