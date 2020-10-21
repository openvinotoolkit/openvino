// Copyright (C) 2019 Intel Corporation
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
enum LOOP_IN_TYPE {
    INVARIANT,
    MERGED
};

using LoopParams = typename std::tuple<
        bool,                                                              // ExecuteFirstIteration
        bool,                                                              // BodyCondition is a constant?
        bool,                                                              // BodyCondition value, if it is a Const
        int64_t,                                                           // TripCount, -1 means infinity
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>>,         // inputs
        InferenceEngine::Precision,                                        // Network precision
        std::string>;                                                      // Device name

class LoopTest : public testing::WithParamInterface<LoopParams>,
                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LoopParams> &obj);

protected:
    void SetUp() override;
};


using StaticShapeLoopParams = typename std::tuple<
        bool,
        std::tuple<
            bool,
            int64_t,
            int64_t,
            int64_t
            >,
        int64_t,
        InferenceEngine::SizeVector,
        InferenceEngine::Precision,
        std::string
        >;

/**
 * Test case with static SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class StaticShapeLoopTest : public testing::WithParamInterface<StaticShapeLoopParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StaticShapeLoopParams> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    std::vector<std::vector<std::uint8_t>> CalculateRefs() override;

private:
    bool static_iter_num;       // trip count provided by constant node
    bool static_continue_cond;  // initial_cond provided by constant node
    int64_t max_iter_num;       // -1 means infinity loop (expected dynamic exit condition in body)
    int64_t dynamic_exit;       // -1 means always true
    int64_t axis;               // -1 means no auto concatenation
    int64_t start_value;
    InferenceEngine::SizeVector data_shape;
    InferenceEngine::Precision data_prc;

    int64_t actual_n_iter();

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
