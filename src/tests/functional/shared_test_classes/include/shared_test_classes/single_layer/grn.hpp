// Copyright (C) 2018-2023 Intel Corporation
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

#include "ngraph/opsets/opset1.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"


namespace LayerTestsDefinitions {
typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::Precision,    // Input precision
    InferenceEngine::Precision,    // Output precision
    InferenceEngine::Layout,       // Input layout
    InferenceEngine::Layout,       // Output layout
    InferenceEngine::SizeVector,
    float,
    std::string> grnParams;

class GrnLayerTest
    : public testing::WithParamInterface<grnParams>,
      virtual public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<grnParams>& obj);

protected:
    InferenceEngine::SizeVector inputShapes;
    float bias;

    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
