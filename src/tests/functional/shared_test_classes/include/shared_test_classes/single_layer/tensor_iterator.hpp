// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ngraph/op/util/attr_types.hpp>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {

using TensorIteratorParams = typename std::tuple<
        bool,                                     // using unroll tensor iterator transformation
        size_t,                                   // seq_lengths
        size_t,                                   // batch
        size_t,                                   // hidden size
        // todo: fix. input size hardcoded to 10 due to limitation (10 args) of gtests Combine() func.
        //size_t,                                 // input size
        size_t,                                   // sequence axis
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
