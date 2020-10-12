// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <memory>
#include <set>
#include <functional>
#include <gtest/gtest.h>


#include "ie_core.hpp"
#include "ie_precision.hpp"
#include "details/ie_exception.hpp"

#include "ngraph/opsets/opset1.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"


namespace LayerTestsDefinitions {
typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::Precision,    // Input precision
    InferenceEngine::Precision,    // Output precision
    InferenceEngine::Layout,       // Input layout
    InferenceEngine::Layout,       // Output layout
    InferenceEngine::SizeVector,
    bool,
    std::string> ctcGreedyDecoderParams;

class CTCGreedyDecoderLayerTest
    :  public testing::WithParamInterface<ctcGreedyDecoderParams>,
       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderParams>& obj);

protected:
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::SizeVector sequenceLengths;
    bool mergeRepeated;

    std::vector<std::vector<std::uint8_t>> CalculateRefs() override;
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
