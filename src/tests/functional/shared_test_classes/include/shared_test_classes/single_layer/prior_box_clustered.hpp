// Copyright (C) 2018-2021 Intel Corporation
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

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<float>,  // widths
        std::vector<float>,  // heights
        bool,                // clip
        float,               // step_width
        float,               // step_height
        float,               // step
        float,               // offset
        std::vector<float>> priorBoxClusteredSpecificParams;

typedef std::tuple<
        priorBoxClusteredSpecificParams,
        ov::test::ElementType,        // net precision
        ov::test::ElementType,        // Input precision
        ov::test::ElementType,        // Output precision
        InferenceEngine::Layout,      // Input layout
        InferenceEngine::Layout,      // Output layout
        ov::test::InputShape,         // input shape
        ov::test::InputShape,         // image shape
        std::string> priorBoxClusteredLayerParams;

class PriorBoxClusteredLayerTest
    : public testing::WithParamInterface<priorBoxClusteredLayerParams>,
      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj);

protected:
    ov::test::InputShape inputShapes;
    ov::test::InputShape imageShapes;
    ov::test::ElementType netPrecision;
    std::vector<float> widths;
    std::vector<float> heights;
    std::vector<float> variances;
    float step_width;
    float step_height;
    float step;
    float offset;
    bool clip;

    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
