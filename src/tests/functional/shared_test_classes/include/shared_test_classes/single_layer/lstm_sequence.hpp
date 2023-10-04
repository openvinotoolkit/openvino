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

using LSTMSequenceParams = typename std::tuple<
        ngraph::helpers::SequenceTestsMode,       // pure Sequence or TensorIterator
        size_t,                                   // seq_lengths
        size_t,                                   // batch
        size_t,                                   // hidden size
        size_t,                                   // input size
        std::vector<std::string>,                 // activations
        float,                                    // clip
        ngraph::op::RecurrentSequenceDirection,   // direction
        ngraph::helpers::InputLayerType,          // WRB input type (Constant or Parameter)
        InferenceEngine::Precision,               // Network precision
        std::string>;                             // Device name


class LSTMSequenceTest : public testing::WithParamInterface<LSTMSequenceParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSequenceParams> &obj);
protected:
    void GenerateInputs() override;
    void SetUp() override;

    ngraph::helpers::SequenceTestsMode m_mode;
    int64_t m_max_seq_len = 0;
};

}  // namespace LayerTestsDefinitions
