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

namespace LayerTestDefinitions {
using priorBoxSpecificParams =  std::tuple<
        std::vector<float>, // min_size
        std::vector<float>, // max_size
        std::vector<float>, // aspect_ratio
        std::vector<float>, // density
        std::vector<float>, // fixed_ratio
        std::vector<float>, // fixed_size
        bool,               // clip
        bool,               // flip
        float,              // step
        float,              // offset
        std::vector<float>, // variance
        bool,               // scale_all_sizes
        bool>;              // min_max_aspect_ratios_order

typedef std::tuple<
        priorBoxSpecificParams,
        ov::test::ElementType,        // net precision
        ov::test::ElementType,        // Input precision
        ov::test::ElementType,        // Output precision
        InferenceEngine::Layout,      // Input layout
        InferenceEngine::Layout,      // Output layout
        ov::test::InputShape,         // input shape
        ov::test::InputShape,         // image shape
        std::string> priorBoxLayerParams;

class PriorBoxLayerTest
    : public testing::WithParamInterface<priorBoxLayerParams>,
      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<priorBoxLayerParams>& obj);
protected:
    ov::test::InputShape inputShapes;
    ov::test::InputShape imageShapes;
    ov::test::ElementType netPrecision;
    std::vector<float> min_size;
    std::vector<float> max_size;
    std::vector<float> aspect_ratio;
    std::vector<float> density;
    std::vector<float> fixed_ratio;
    std::vector<float> fixed_size;
    std::vector<float> variance;
    float step;
    float offset;
    bool clip;
    bool flip;
    bool scale_all_sizes;
    bool min_max_aspect_ratios_order;

    void SetUp() override;
};

} // namespace LayerTestDefinitions
