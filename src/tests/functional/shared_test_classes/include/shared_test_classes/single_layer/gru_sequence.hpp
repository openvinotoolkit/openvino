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
#include "common_test_utils/test_enums.hpp"

namespace LayerTestsDefinitions {

using GRUSequenceParams = typename std::tuple<
        ngraph::helpers::SequenceTestsMode,       // pure Sequence or TensorIterator
        size_t,                                   // seq_lengths
        size_t,                                   // batch
        size_t,                                   // hidden size
        // todo: fix. input size hardcoded to 10 due to limitation (10 args) of gtests Combine() func.
        //size_t,                                 // input size
        std::vector<std::string>,                 // activations
        float,                                    // clip
        bool,                                     // linear_before_reset
        ngraph::op::RecurrentSequenceDirection,   // direction
        ngraph::helpers::InputLayerType,          // WRB input type (Constant or Parameter)
        InferenceEngine::Precision,               // Network precision
        std::string>;                             // Device name

class GRUSequenceTest : public testing::WithParamInterface<GRUSequenceParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GRUSequenceParams> &obj);

protected:
    void SetUp() override;
    void GenerateInputs() override;
    ngraph::helpers::SequenceTestsMode m_mode;
    int64_t m_max_seq_len = 0;
};

}  // namespace LayerTestsDefinitions
