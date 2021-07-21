// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include "ngraph_functions/builders.hpp"
#include <tuple>

#include "comparison.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using ComparisonTypes = ngraph::helpers::ComparisonTypes;

namespace ComparisonOpsRefTestDefinitions {
void ReferenceComparisonLayerTest::SetUp() {
    auto params = GetParam();
    function = CreateFunction(params.comparisonType, params.pshape1, params.pshape2, params.inType, params.outType);
    inputData = {params.inputData1, params.inputData2};
    refOutData = {params.refData};
}
} // namespace ComparisonOpsRefTestDefinitions
