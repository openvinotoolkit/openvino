// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
enum LOOP_IN_TYPE {
    INVARIANT,
    MERGED
};

using LoopParams = typename std::tuple<
        bool,                            // ExecuteFirstIteration
        bool,                            // BodyCondition is a constant?
        bool,                            // BodyCondition value, if it is a Const
        int64_t,                         // TripCount, -1 means infinity
        std::vector<InputShape>,         // input shapes
        std::vector<LOOP_IN_TYPE>,       // input types. Vector size have to be equal to input shapes vector size
        ov::element::Type,               // Model type
        std::string>;                    // Device name

class LoopLayerTest : public testing::WithParamInterface<LoopParams>,
                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LoopParams> &obj);

protected:
    void SetUp() override;
};

using StaticShapeLoopParams = typename std::tuple<
    bool,
    bool,
    std::tuple<
        bool,
        int64_t,
        int64_t,
        int64_t
        >,
    int64_t,
    ov::Shape,
    ov::element::Type,
    std::string>;

/**
 * Test case with static SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class StaticShapeLoopLayerTest : public testing::WithParamInterface<StaticShapeLoopParams>,
                            virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StaticShapeLoopParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
