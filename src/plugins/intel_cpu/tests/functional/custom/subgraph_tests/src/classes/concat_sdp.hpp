// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <set>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// Subgraph:
/*                            Parameter
 *                                |
 *       Parameter    ReadValue   |    ReadValue  Parameter
 *           \           /        |       \          /
 *         Gather       /               Gather      /
 *             \       /          |         \      /
 *               Concat           |          Concat
 *                / \             |            / \
 *               /   \            |           /   \
 *              /     \           |          /     \
 *          Assign     ScaledDotProductAttention  Assign
 *                                |
 *                               Add
 *                                |
 *                              Result
 */
// (inType, inputShapes, cacheCfg, hasShapeOf, head_num_q, head_num_kv)
// cacheCfg carries KEY_CACHE_* / VALUE_CACHE_* / KV_CACHE_* properties; empty = default.
typedef std::tuple<ElementType,
                   std::vector<InputShape>,
                   ov::AnyMap,
                   bool,
                   int64_t,
                   int64_t>
    ConcatSDPTestParams;

class ConcatSDPTest : public testing::WithParamInterface<ConcatSDPTestParams>,
                      virtual public ov::test::SubgraphBaseTest,
                      public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSDPTestParams>& obj);

protected:
    void SetUp() override;
    void run() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    // Compile+run all iters. Updates `compiledModel` so post-checks see last run.
    std::vector<std::vector<ov::Tensor>>
    run_test(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& cfg);

    ov::AnyMap m_cacheCfg;
    bool m_hasShapeOf = false;
    int64_t m_headNumQ = 8;
    int64_t m_headNumKV = 8;
    int m_iter = 0;
    size_t m_accum_L_q = 0;
    // Iteration indices at which the KV-cache states are reset before the infer, i.e. the
    // first prompt of a new conversation. Iteration 0 is implicitly a conversation start
    std::set<size_t> m_resetBefore;
};

// Same subgraph as ConcatSDPTest, but the iteration list is split into several conversations
// by `reset_state` calls, with a different batch in each conversation
class ConcatSDPResetStateTest : public ConcatSDPTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSDPTestParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
