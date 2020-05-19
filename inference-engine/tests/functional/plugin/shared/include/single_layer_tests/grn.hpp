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
#include "ngraph/runtime/reference/relu.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"


namespace LayerTestsDefinitions {
typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    float,
    std::string> grnParams;

class GrnLayerTest
    : public testing::WithParamInterface<grnParams>,
      public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<grnParams>& obj);

protected:
    InferenceEngine::SizeVector inputShapes;
    float bias;

    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
