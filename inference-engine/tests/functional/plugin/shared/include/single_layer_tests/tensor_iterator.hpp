// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ngraph/op/util/attr_types.hpp>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using TensorIteratorParams = typename std::tuple<
        bool,                                     // using unroll tensor iterator transformation
        size_t,                                   // seq_lengths
        size_t,                                   // batch
        size_t,                                   // hidden size
        size_t,                                   // input size
        float,                                    // clip
        ngraph::helpers::TensorIteratorBody,      // body type
        ngraph::op::RecurrentSequenceDirection,   // direction
        InferenceEngine::Precision,               // Network precision
        std::string>;                             // Device name

class TensorIteratorTest : public testing::WithParamInterface<TensorIteratorParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TensorIteratorParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
