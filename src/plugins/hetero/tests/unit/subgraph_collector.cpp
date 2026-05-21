// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_collector.hpp"

#include <gtest/gtest.h>

#include <algorithm>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "op/device_subgraph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"

using namespace ov::hetero;

namespace {
// input -> add -> sub -> reshape -> result
std::shared_ptr<ov::Model> create_linear_chain_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::op::v1::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(subtract, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
// input -> add1 -> add2 -> sub -> add3 -> reshape -> result
//                  └──────────────────→ add3 (diamond topology)
std::shared_ptr<ov::Model> create_diamond_chain_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add1 = std::make_shared<ov::op::v1::Add>(param, const_value);
    add1->set_friendly_name("add1");
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, const_value);
    add2->set_friendly_name("add2");
    auto subtract = std::make_shared<ov::op::v1::Subtract>(add2, const_value);
    subtract->set_friendly_name("sub");
    auto add3 = std::make_shared<ov::op::v1::Add>(add1, subtract);
    add3->set_friendly_name("add3");
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(add3, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
// Subgraph: param -> add -> result (output: add)
std::shared_ptr<ov::Model> create_subgraph_add() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param, const_value);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::OutputVector{add->output(0)}, ov::ParameterVector{param});
}
// Subgraph: input -> add1 -> add2 (outputs: add2, add1)
std::shared_ptr<ov::Model> create_subgraph_add_add() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add1 = std::make_shared<ov::op::v1::Add>(param, const_value);
    add1->set_friendly_name("add1");
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, const_value);
    add2->set_friendly_name("add2");
    return std::make_shared<ov::Model>(ov::OutputVector{add2->output(0), add1->output(0)}, ov::ParameterVector{param});
}
// Subgraph: input -> sub (output: sub)
std::shared_ptr<ov::Model> create_subgraph_sub() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto sub = std::make_shared<ov::op::v1::Subtract>(param, const_value);
    sub->set_friendly_name("sub");
    return std::make_shared<ov::Model>(ov::OutputVector{sub->output(0)}, ov::ParameterVector{param});
}
// Subgraph: (param0, param1) -> add -> reshape -> result
std::shared_ptr<ov::Model> create_subgraph_add_reshape() {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    auto add = std::make_shared<ov::op::v1::Add>(param0, param1);
    add->set_friendly_name("add_second");
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(add, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0, param1});
}
// Subgraph: input -> reshape -> result
std::shared_ptr<ov::Model> create_subgraph_reshape() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(param, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::op::v0::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
// Model: param → A → B → C → D → result (linear chain for alternating device tests)
std::shared_ptr<ov::Model> create_alternating_chain_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    param->set_friendly_name("input");
    auto c1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    c1->set_friendly_name("c1");
    auto a = std::make_shared<ov::op::v1::Add>(param, c1);
    a->set_friendly_name("A");
    auto b = std::make_shared<ov::op::v1::Subtract>(a, c1);
    b->set_friendly_name("B");
    auto c = std::make_shared<ov::op::v1::Add>(b, c1);
    c->set_friendly_name("C");
    auto d = std::make_shared<ov::op::v1::Subtract>(c, c1);
    d->set_friendly_name("D");
    auto result = std::make_shared<ov::op::v0::Result>(d);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
// Model: param1 → add1 → res1, param2 → add2 → res2 (two independent paths)
std::shared_ptr<ov::Model> create_independent_paths_model() {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2});
    param1->set_friendly_name("input1");
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2});
    param2->set_friendly_name("input2");
    auto c1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    c1->set_friendly_name("c1");
    auto c2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    c2->set_friendly_name("c2");
    auto add1 = std::make_shared<ov::op::v1::Add>(param1, c1);
    add1->set_friendly_name("add1");
    auto add2 = std::make_shared<ov::op::v1::Add>(param2, c2);
    add2->set_friendly_name("add2");
    auto result1 = std::make_shared<ov::op::v0::Result>(add1);
    result1->set_friendly_name("res1");
    auto result2 = std::make_shared<ov::op::v0::Result>(add2);
    result2->set_friendly_name("res2");
    return std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
}
// Model: param1 → add1(+c1) → add2(+c2) → res, param2 unused (for merge_independent test)
std::shared_ptr<ov::Model> create_merge_independent_model() {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    param1->set_friendly_name("input1");
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    param2->set_friendly_name("input2");
    auto const_value1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {1});
    const_value1->set_friendly_name("const_val1");
    auto add1 = std::make_shared<ov::op::v1::Add>(param1, const_value1);
    add1->set_friendly_name("add1");
    auto const_value2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {1});
    const_value2->set_friendly_name("const_val2");
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, const_value2);
    add2->set_friendly_name("add2");
    auto result = std::make_shared<ov::op::v0::Result>(add2);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});
}
// Model: param → add → sub1 → res1, add → sub2 → res2 (diverging paths from shared node)
std::shared_ptr<ov::Model> create_diverging_paths_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2});
    param->set_friendly_name("input");
    auto c1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    c1->set_friendly_name("c1");
    auto add = std::make_shared<ov::op::v1::Add>(param, c1);
    add->set_friendly_name("add");
    auto sub1 = std::make_shared<ov::op::v1::Subtract>(add, c1);
    sub1->set_friendly_name("sub1");
    auto sub2 = std::make_shared<ov::op::v1::Subtract>(add, c1);
    sub2->set_friendly_name("sub2");
    auto result1 = std::make_shared<ov::op::v0::Result>(sub1);
    result1->set_friendly_name("res1");
    auto result2 = std::make_shared<ov::op::v0::Result>(sub2);
    result2->set_friendly_name("res2");
    return std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param});
}

// Model: input → a1 → b1 → a2 → b2 → a3 → res, with a1 also feeding a3 (skip edge).
// Designed to stress SubgraphCollector::split_cyclic_dependencies() with nested cycles:
// when {a1, a2, a3} are on MOCK.0 and {b1, b2} on MOCK.1, the initial M0 group contains
// two stacked cyclic dependencies (via b1 and via b2) that require multiple split iterations
// of the fixed-point loop to resolve.
std::shared_ptr<ov::Model> create_nested_cyclic_chain_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    param->set_friendly_name("input");
    auto c1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    c1->set_friendly_name("c1");
    auto a1 = std::make_shared<ov::op::v1::Add>(param, c1);
    a1->set_friendly_name("a1");
    auto b1 = std::make_shared<ov::op::v1::Subtract>(a1, c1);
    b1->set_friendly_name("b1");
    auto a2 = std::make_shared<ov::op::v1::Add>(b1, c1);
    a2->set_friendly_name("a2");
    auto b2 = std::make_shared<ov::op::v1::Subtract>(a2, c1);
    b2->set_friendly_name("b2");
    auto a3 = std::make_shared<ov::op::v1::Add>(b2, a1);
    a3->set_friendly_name("a3");
    auto result = std::make_shared<ov::op::v0::Result>(a3);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

// Model: shared constant bridges disjoint paths into one subgraph via indirect Union-Find merge.
//
// Topology: param(M0) → A(M0,+other_const) → B(M1) → C(M0,+shared_const) → res1
//                                            A → X(M0,+shared_const) → res2
//
// Union-Find merges {param, other_const, A, X, shared_const, C, res1, res2} into one subgraph
// because X (reachable via res2) connects A to shared_const, and shared_const connects to C.
// This subgraph depends on B (via C←B) and B depends on the same subgraph (via B←A) → cycle.
// split_cyclic_dependencies() must detect the cycle re-entry point (C←B) and promote
// C.input(1) (from shared_const) as a boundary to break the cycle.
std::shared_ptr<ov::Model> create_shared_const_indirect_bridge_cycle_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    param->set_friendly_name("input");
    auto shared_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    shared_const->set_friendly_name("shared_const");
    auto other_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f});
    other_const->set_friendly_name("other_const");
    // A: DEV0, uses param + other_const (NOT shared_const!)
    auto a = std::make_shared<ov::op::v1::Add>(param, other_const);
    a->set_friendly_name("A");
    // X: DEV0, uses A + shared_const — bridges shared_const into A's subgraph via Union-Find.
    // Crucially, X leads to res2, so it IS in get_ordered_ops().
    auto x = std::make_shared<ov::op::v1::Add>(a, shared_const);
    x->set_friendly_name("X");
    // B: DEV1, uses A (cross-device boundary, unary op to minimize noise)
    auto b = std::make_shared<ov::op::v0::Abs>(a);
    b->set_friendly_name("B");
    // C: DEV0, uses B + shared_const (after DEV1 detour, merges with A's sg via shared_const)
    auto c = std::make_shared<ov::op::v1::Add>(b, shared_const);
    c->set_friendly_name("C");
    auto res1 = std::make_shared<ov::op::v0::Result>(c);
    res1->set_friendly_name("res1");
    // res2 makes X reachable — the key to triggering the bug
    auto res2 = std::make_shared<ov::op::v0::Result>(x);
    res2->set_friendly_name("res2");
    return std::make_shared<ov::Model>(ov::ResultVector{res1, res2}, ov::ParameterVector{param});
}

// Model: shared constant IS the cycle source — it directly feeds a cross-device op AND merges
// with the cycle re-entry node through a purely internal chain.
// Topology: param(DEV0) → A(DEV0) → B(DEV1, +shared_const) → C(DEV0) → res1
//           shared_const(DEV0) → X(DEV0) → C(DEV0)
//           X → res2
//
// A→B: cross-device (DEV0→DEV1) → boundary.
// shared_const→B: cross-device (DEV0→DEV1) → boundary (constant IS the cycle source!).
// B→C: cross-device (DEV1→DEV0) → boundary.
// shared_const→X: same device → NOT boundary → merge via Union-Find.
// X→C: same device → NOT boundary → merge via Union-Find.
//
// Union-Find: SG0={param,A}, SG1={B}, SG2={shared_const,X,C,res1,res2}.
// Cycle: SG2→SG1 (shared_const→B) and SG1→SG2 (B→C).
// split_cyclic_dependencies() must promote C.input(1) (from X) as a boundary,
// yielding 4 subgraphs: {param,A}(DEV0), {B}(DEV1), {shared_const,X,res2}(DEV0), {C,res1}(DEV0).
std::shared_ptr<ov::Model> create_shared_const_as_cycle_source_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    param->set_friendly_name("param");
    auto shared_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    shared_const->set_friendly_name("shared_const");
    // A: DEV0, uses param only
    auto a = std::make_shared<ov::op::v0::Abs>(param);
    a->set_friendly_name("A");
    // B: DEV1, uses A + shared_const. shared_const→B is the KEY cross-device edge making
    // shared_const the cycle source (SG2 produces output to SG1).
    auto b = std::make_shared<ov::op::v1::Add>(a, shared_const);
    b->set_friendly_name("B");
    // X: DEV0, uses shared_const. Merges with shared_const (same device, internal).
    auto x = std::make_shared<ov::op::v0::Abs>(shared_const);
    x->set_friendly_name("X");
    // C: DEV0, uses B (boundary, re-entry from SG1) + X (internal, merged with shared_const).
    // Without the fix, the intersection check fails because cyclic_inputs_dependencies
    // doesn't include shared_const's self-boundary bit.
    auto c = std::make_shared<ov::op::v1::Add>(b, x);
    c->set_friendly_name("C");
    auto res1 = std::make_shared<ov::op::v0::Result>(c);
    res1->set_friendly_name("res1");
    auto res2 = std::make_shared<ov::op::v0::Result>(x);
    res2->set_friendly_name("res2");
    return std::make_shared<ov::Model>(ov::ResultVector{res1, res2}, ov::ParameterVector{param});
}

// Model: shared constant fans out to three consumers across two devices.
// Topology: param(GPU) → Node_A(GPU, +shared_const) → Node_B(CPU, +shared_const) → Node_C(GPU, +shared_const) → res
//
// shared_const(GPU) → Node_A(GPU): same affinity → NOT boundary, merged via Union-Find.
// shared_const(GPU) → Node_B(CPU): different affinity → IS boundary (cross-device edge).
// shared_const(GPU) → Node_C(GPU): same affinity → NOT boundary, merged via Union-Find.
//
// Union-Find produces one GPU subgraph {param, shared_const, Node_A, Node_C, res} + one CPU {Node_B}.
// GPU depends on Node_B (Node_C←Node_B) and Node_B depends on GPU (Node_B←Node_A) → cycle.
// split_cyclic_dependencies() must promote Node_C.input(0) (from Node_B) as a boundary,
// yielding 3 subgraphs: {param, shared_const, Node_A}(GPU), {Node_B}(CPU), {Node_C, res}(GPU).
std::shared_ptr<ov::Model> create_shared_const_cross_device_fanout_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    param->set_friendly_name("param");
    auto shared_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    shared_const->set_friendly_name("shared_const");
    // Node_A: GPU, consumes param + shared_const (same device → merged with shared_const)
    auto node_a = std::make_shared<ov::op::v1::Add>(param, shared_const);
    node_a->set_friendly_name("Node_A");
    // Node_B: CPU, consumes Node_A + shared_const
    //   Node_A→Node_B is boundary (GPU→CPU)
    //   shared_const→Node_B is boundary (GPU→CPU) — this is the KEY edge
    auto node_b = std::make_shared<ov::op::v1::Add>(node_a, shared_const);
    node_b->set_friendly_name("Node_B");
    // Node_C: GPU, consumes Node_B + shared_const
    //   Node_B→Node_C is boundary (CPU→GPU)
    //   shared_const→Node_C is NOT boundary (GPU→GPU) — merges Node_C into shared_const's subgraph
    auto node_c = std::make_shared<ov::op::v1::Add>(node_b, shared_const);
    node_c->set_friendly_name("Node_C");
    auto res = std::make_shared<ov::op::v0::Result>(node_c);
    res->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param});
}

// Stateful model: param → read_value → add(+c1) → {result, assign(sink)}.
// Single-device by design — exercises Subgraph::_sinks wire-through and
// create_submodel_from_collected_subgraph()'s sink-preserving construction without
// depending on the (currently unspecified) cross-affinity Assign/ReadValue contract.
std::shared_ptr<ov::Model> create_stateful_single_device_model() {
    const ov::op::util::VariableInfo variable_info{ov::PartialShape{4}, ov::element::f32, "var0"};
    auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    param->set_friendly_name("input");
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(param, variable);
    read_value->set_friendly_name("read_value");
    auto c1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
    c1->set_friendly_name("c1");
    auto add = std::make_shared<ov::op::v1::Add>(read_value, c1);
    add->set_friendly_name("add");
    auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    assign->set_friendly_name("assign");
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::SinkVector{assign}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> create_submodel_from_collected_subgraph(const ov::hetero::Subgraph& sg) {
    return std::make_shared<ov::Model>(sg._results, sg._sinks, sg._parameters);
}
}  // namespace

class SubgraphCollectorTest : public testing::Test {
    void SetUp() override {
        m_model = create_linear_chain_model();
        m_model_ref = m_model->clone();
        m_submodels = {};
        m_submodels.push_back(create_subgraph_add());
        m_submodels.push_back(create_subgraph_sub());
        m_submodels.push_back(create_subgraph_reshape());
    }

    void TearDown() override {}

protected:
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ov::Model> m_model_ref;
    std::vector<std::shared_ptr<ov::Model>> m_submodels;
};

TEST_F(SubgraphCollectorTest, mask_ops_single_device) {
    const std::map<std::string, std::string> supported_ops_mock0 = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");

        auto device_subgraph0 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{param->output(0)},
                                                                                 m_submodels.at(0),
                                                                                 "MOCK.0");
        device_subgraph0->set_friendly_name("device_subgraph0");

        auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
        const_value->set_friendly_name("const_val");
        auto subtract = std::make_shared<ov::op::v1::Subtract>(device_subgraph0->output(0), const_value);
        subtract->set_friendly_name("sub");

        auto device_subgraph1 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{subtract->output(0)},
                                                                                 m_submodels.at(2),
                                                                                 "MOCK.0");
        device_subgraph1->set_friendly_name("device_subgraph1");

        auto result = std::make_shared<ov::op::v0::Result>(device_subgraph1->output(0));
        result->set_friendly_name("res");
        m_model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    auto supported_ops = supported_ops_mock0;
    auto actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST");
    auto res = compare_functions(m_model_ref, m_model);
    ASSERT_TRUE(res.first) << res.second;

    // Only first device was replace, mapping is unavailable
    ASSERT_TRUE(actual_mapping_info.empty());
}

TEST_F(SubgraphCollectorTest, mask_ops_all_devices) {
    const std::map<std::string, std::string> supported_ops_mock0 = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    const std::map<std::string, std::string> supported_ops_mock1 = {
        {"sub", "MOCK.1"},
    };
    const SubgraphsMappingInfo exptected_mapping_info = {{// inputs to submodel inputs
                                                          NodeInfo{0, 0}},
                                                         {// outputs to submodel outpts
                                                          NodeInfo{2, 0}},
                                                         {// submodel input to previous output
                                                          {NodeInfo{1, 0}, NodeInfo{0, 0}},
                                                          {NodeInfo{2, 0}, NodeInfo{1, 0}}}};
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");

        auto device_subgraph0 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{param->output(0)},
                                                                                 m_submodels.at(0),
                                                                                 "MOCK.0");
        device_subgraph0->set_friendly_name("device_subgraph0");

        auto device_subgraph1 =
            std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{device_subgraph0->output(0)},
                                                             m_submodels.at(1),
                                                             "MOCK.1");
        device_subgraph1->set_friendly_name("device_subgraph1");

        auto device_subgraph2 =
            std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{device_subgraph1->output(0)},
                                                             m_submodels.at(2),
                                                             "MOCK.0");
        device_subgraph2->set_friendly_name("device_subgraph2");

        auto result = std::make_shared<ov::op::v0::Result>(device_subgraph2->output(0));
        result->set_friendly_name("res");
        m_model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    ov::hetero::SubgraphsMappingInfo actual_mapping_info;
    auto supported_ops = supported_ops_mock0;
    ASSERT_EQ(6, supported_ops.size());
    actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST");
    // + 2 subgraphs + 2 params + 2 results
    ASSERT_EQ(12, supported_ops.size());
    // Adds mock1 supported operations
    supported_ops.insert(supported_ops_mock1.begin(), supported_ops_mock1.end());
    ASSERT_EQ(13, supported_ops.size());
    actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false);
    // + 1 subgraphs + 1 param + 1 result
    ASSERT_EQ(16, supported_ops.size());
    auto res = compare_functions(m_model_ref, m_model);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_EQ(exptected_mapping_info._inputs_to_submodels_inputs, actual_mapping_info._inputs_to_submodels_inputs);
    ASSERT_EQ(exptected_mapping_info._outputs_to_submodels_outputs, actual_mapping_info._outputs_to_submodels_outputs);
    ASSERT_EQ(exptected_mapping_info._submodels_input_to_prev_output,
              actual_mapping_info._submodels_input_to_prev_output);
}

class SubgraphCollectorTest2 : public SubgraphCollectorTest {
    void SetUp() override {
        m_model = create_diamond_chain_model();
        m_model_ref = m_model->clone();
        m_submodels = {};
        m_submodels.push_back(create_subgraph_add_add());
        m_submodels.push_back(create_subgraph_sub());
        m_submodels.push_back(create_subgraph_add_reshape());
    }
};

TEST_F(SubgraphCollectorTest2, mask_ops_single_device) {
    const std::map<std::string, std::string> supported_ops_mock0 = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add1", "MOCK.0"},
        {"add2", "MOCK.0"},
        {"add3", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");

        auto device_subgraph0 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{param->output(0)},
                                                                                 m_submodels.at(0),
                                                                                 "MOCK.0");
        device_subgraph0->set_friendly_name("device_subgraph0");

        auto const_value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
        const_value->set_friendly_name("const_val");
        auto subtract = std::make_shared<ov::op::v1::Subtract>(device_subgraph0->output(0), const_value);
        subtract->set_friendly_name("sub");

        auto device_subgraph2 = std::make_shared<ov::hetero::op::DeviceSubgraph>(
            ov::OutputVector{device_subgraph0->output(0), subtract->output(0)},
            m_submodels.at(2),
            "MOCK.0");
        device_subgraph2->set_friendly_name("device_subgraph2");

        auto result = std::make_shared<ov::op::v0::Result>(device_subgraph2->output(0));
        result->set_friendly_name("res");
        m_model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    auto supported_ops = supported_ops_mock0;
    ASSERT_EQ(8, supported_ops.size());
    auto actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST");
    // + 2 subgraphs + 3 params + 3 results
    ASSERT_EQ(16, supported_ops.size());
    auto res = compare_functions(m_model_ref, m_model);
    ASSERT_TRUE(res.first) << res.second;

    // Only first device was replace, mapping is unavailable
    ASSERT_TRUE(actual_mapping_info.empty());
}

TEST_F(SubgraphCollectorTest2, mask_ops_all_devices) {
    const std::map<std::string, std::string> supported_ops_mock0 = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add1", "MOCK.0"},
        {"add2", "MOCK.0"},
        {"add3", "MOCK.0"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    const std::map<std::string, std::string> supported_ops_mock1 = {
        {"sub", "MOCK.1"},
    };
    const SubgraphsMappingInfo exptected_mapping_info = {{// inputs to submodel inputs
                                                          NodeInfo{0, 0}},
                                                         {// outputs to submodel outpts
                                                          NodeInfo{2, 0}},
                                                         {// submodel input to previous output
                                                          {NodeInfo{1, 0}, NodeInfo{0, 0}},
                                                          {NodeInfo{2, 0}, NodeInfo{0, 1}},
                                                          {NodeInfo{2, 1}, NodeInfo{1, 0}}}};
    {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("input");

        auto device_subgraph0 = std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{param->output(0)},
                                                                                 m_submodels.at(0),
                                                                                 "MOCK.0");
        device_subgraph0->set_friendly_name("device_subgraph0");

        auto device_subgraph1 =
            std::make_shared<ov::hetero::op::DeviceSubgraph>(ov::OutputVector{device_subgraph0->output(1)},
                                                             m_submodels.at(1),
                                                             "MOCK.1");
        device_subgraph1->set_friendly_name("device_subgraph1");

        auto device_subgraph2 = std::make_shared<ov::hetero::op::DeviceSubgraph>(
            ov::OutputVector{device_subgraph0->output(0), device_subgraph1->output(0)},
            m_submodels.at(2),
            "MOCK.0");
        device_subgraph2->set_friendly_name("device_subgraph2");

        auto result = std::make_shared<ov::op::v0::Result>(device_subgraph2->output(0));
        result->set_friendly_name("res");
        m_model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
    ov::hetero::SubgraphsMappingInfo actual_mapping_info;
    auto supported_ops = supported_ops_mock0;
    ASSERT_EQ(8, supported_ops.size());
    actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST");
    // + 2 subgraphs + 3 params + 3 results
    ASSERT_EQ(16, supported_ops.size());
    // Adds mock1 supported operations
    supported_ops.insert(supported_ops_mock1.begin(), supported_ops_mock1.end());
    ASSERT_EQ(17, supported_ops.size());
    actual_mapping_info = ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false);
    // + 1 subgraphs + 1 param + 1 result
    ASSERT_EQ(20, supported_ops.size());
    auto res = compare_functions(m_model_ref, m_model);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_EQ(exptected_mapping_info._inputs_to_submodels_inputs, actual_mapping_info._inputs_to_submodels_inputs);
    ASSERT_EQ(exptected_mapping_info._outputs_to_submodels_outputs, actual_mapping_info._outputs_to_submodels_outputs);
    ASSERT_EQ(exptected_mapping_info._submodels_input_to_prev_output,
              actual_mapping_info._submodels_input_to_prev_output);
}

TEST_F(SubgraphCollectorTest, mask_ops_mixed_affinity_no_throw) {
    const std::map<std::string, std::string> supported_ops_with_affinity = {
        {"input", "MOCK.0"},
        {"const_val", "MOCK.0"},
        {"add", "MOCK.1"},
        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.0"},
        {"res", "MOCK.0"},
    };
    auto supported_ops = supported_ops_with_affinity;
    OV_ASSERT_NO_THROW(ov::hetero::mask_model_subgraphs_by_ops(m_model, supported_ops, false, "TEST"));
}

TEST_F(SubgraphCollectorTest, constant_subgraphs_follow_consumer_affinity) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{1, 3, 16, 16});
    input->set_friendly_name("input");
    auto convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    convert->set_friendly_name("convert");

    auto constant1 = ov::op::v0::Constant::create(ov::element::f32, {}, {2.f});
    constant1->set_friendly_name("constant1");

    auto mul = std::make_shared<ov::op::v1::Multiply>(convert, constant1);
    mul->set_friendly_name("mul");

    auto constant2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
    constant2->set_friendly_name("constant2");
    auto transpose = std::make_shared<ov::op::v1::Transpose>(mul, constant2);
    transpose->set_friendly_name("transpose");
    auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(transpose);
    shapeOf->set_friendly_name("shapeOf");

    auto reshape_val = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::op::v1::Reshape>(shapeOf, reshape_val, true);
    reshape->set_friendly_name("reshape");

    auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    zero->set_friendly_name("zero");
    auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    one->set_friendly_name("one");
    auto gather = std::make_shared<ov::op::v8::Gather>(reshape, one, zero);
    gather->set_friendly_name("gather");

    auto result = std::make_shared<ov::op::v0::Result>(gather);
    result->set_friendly_name("result");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    const std::map<std::string, std::string> supported_ops_with_affinity = {
        {"input", "MOCK.0"},
        {"convert", "MOCK.0"},
        {"mul", "MOCK.0"},
        {"constant1", "MOCK.0"},
        {"constant2", "MOCK.0"},
        {"transpose", "MOCK.1"},
        {"shapeOf", "MOCK.0"},

        {"reshape_val", "MOCK.0"},
        {"reshape", "MOCK.1"},
        {"zero", "MOCK.0"},
        {"one", "MOCK.0"},
        {"gather", "MOCK.1"},
        {"result", "MOCK.0"},
    };

    auto supported_ops = supported_ops_with_affinity;

    const auto& [ordered_subgraphs, actual_mapping_info] = get_model_subgraphs(model, supported_ops, true, false);
    for (const auto& subgraph : ordered_subgraphs) {
        std::set<std::string> node_set;
        auto sub_model = create_submodel_from_collected_subgraph(subgraph);
        for (auto& node : sub_model->get_ordered_ops()) {
            node_set.insert(node->get_friendly_name());
        }
        ASSERT_EQ(node_set.count("transpose"), node_set.count("constant2"));
        ASSERT_EQ(node_set.count("reshape"), node_set.count("reshape_val"));
        ASSERT_EQ(node_set.count("gather"), node_set.count("zero"));
        ASSERT_EQ(node_set.count("gather"), node_set.count("one"));
        if (node_set.count("transpose") || node_set.count("reshape") || node_set.count("gather")) {
            ASSERT_EQ(subgraph._affinity, "MOCK.1");
        }
    }
}

// --- Parameterized SubgraphCollector tests ---
// Unified test for all scenarios that directly use SubgraphCollector:
// split_and_merge, merge_independent, and affinity-based subgraph counting.
// Each param specifies model, affinities, and expected results. Optional fields control extra checks.
//
// Sink coverage: the single-device stateful_assign_readvalue case below exercises
// Subgraph::_sinks wire-through (collection → create_submodel_from_collected_subgraph →
// merge_submodels). The cross-affinity case — where paired Assign/ReadValue land in different
// subgraphs — is intentionally not covered here: the expected SubgraphCollector contract for
// splitting stateful variable pairs is not yet specified; that case should be added in a
// follow-up once the contract is clarified.

struct SubgraphCollectorTestParam {
    using ModelFactory = std::function<std::shared_ptr<ov::Model>()>;
    // --- required fields ---
    std::string test_name;
    ModelFactory create_model;                        // factory to build the model under test
    std::map<std::string, std::string> affinity_map;  // node_name → device; empty = broadcast default
    std::string default_affinity;                     // used when affinity_map is empty
    size_t expected_subgraph_count;                   // number of subgraphs from run()
    // --- optional checks (a default-constructed/empty/false value disables the check) ---
    std::vector<std::string> expected_affinities = {};                       // sorted affinity list per subgraph
    std::map<std::string, SubgraphCollector::SubgraphId> expected_ids = {};  // node_name → expected subgraph ID
    std::vector<ModelFactory> expected_submodel_factories = {};              // reference submodel per subgraph
    SubgraphsMappingInfo expected_mapping = {};                              // expected mapping info from run()
    size_t expected_total_sinks = 0;      // sum of sg._sinks.size() across subgraphs (0 = no check)
    bool verify_merge_roundtrip = false;  // merge submodels back and check size == 1
    bool verify_merge_compare = false;    // compare_functions(original, merged)
};

class SubgraphCollectorParamTest : public testing::TestWithParam<SubgraphCollectorTestParam> {};

TEST_P(SubgraphCollectorParamTest, split_by_affinity) {
    const auto& param = GetParam();
    auto model = param.create_model();
    auto model_ref = model->clone();

    // Test fixture guard: friendly_name uniqueness is not an ov::Model invariant, but every
    // factory above relies on it to key affinity_map / expected_ids by name. A duplicate would
    // silently corrupt those lookups, so fail fast with a clear message instead.
    {
        std::set<std::string> seen_names;
        for (const auto& node : model->get_ordered_ops()) {
            ASSERT_TRUE(seen_names.insert(node->get_friendly_name()).second)
                << "Test fixture bug: duplicate friendly_name '" << node->get_friendly_name() << "'";
        }
    }

    SubgraphCollector::AffinitiesMap affinities;
    for (const auto& node : model->get_ordered_ops()) {
        if (param.affinity_map.empty()) {
            affinities[node] = param.default_affinity;
        } else {
            const auto& name = node->get_friendly_name();
            const auto it = param.affinity_map.find(name);
            ASSERT_TRUE(it != param.affinity_map.end()) << "Missing affinity for node '" << name << "'";
            affinities[node] = it->second;
        }
    }

    SubgraphCollector collector(model, affinities);

    // Check subgraph IDs if expected
    if (!param.expected_ids.empty()) {
        auto actual_ids = collector.get_subgraph_ids();
        ASSERT_EQ(param.expected_ids.size(), actual_ids.size());
        for (const auto& [node, actual_id] : actual_ids) {
            const auto& name = node->get_friendly_name();
            auto it = param.expected_ids.find(name);
            ASSERT_TRUE(it != param.expected_ids.end()) << "No expected ID for node: " << name;
            ASSERT_EQ(it->second, actual_id) << "ID mismatch for node: " << name;
        }
    }

    const auto& [subgraphs, mapping] = collector.run();

    ASSERT_EQ(param.expected_subgraph_count, subgraphs.size());

    // Check affinities (sorted comparison)
    if (!param.expected_affinities.empty()) {
        std::vector<std::string> actual_affinities;
        for (const auto& sg : subgraphs) {
            actual_affinities.push_back(sg._affinity);
        }
        std::sort(actual_affinities.begin(), actual_affinities.end());
        auto expected_sorted = param.expected_affinities;
        std::sort(expected_sorted.begin(), expected_sorted.end());
        ASSERT_EQ(expected_sorted, actual_affinities);
    }

    // Check submodel structure if reference factories provided
    if (!param.expected_submodel_factories.empty()) {
        std::vector<std::shared_ptr<ov::Model>> actual_submodels;
        for (const auto& sg : subgraphs) {
            actual_submodels.push_back(create_submodel_from_collected_subgraph(sg));
        }
        ASSERT_EQ(param.expected_submodel_factories.size(), actual_submodels.size());

        auto unmatched_actual_submodels = actual_submodels;
        for (size_t i = 0; i < param.expected_submodel_factories.size(); i++) {
            auto expected_submodel = param.expected_submodel_factories[i]();
            bool matched = false;
            std::string mismatch_details;

            // Order-independent match: scan remaining actuals, take the first that compares equal,
            // and remove it from the pool so each expected pairs with a distinct actual submodel.
            for (auto it = unmatched_actual_submodels.begin(); it != unmatched_actual_submodels.end(); ++it) {
                auto res = compare_functions(expected_submodel, *it);
                if (res.first) {
                    unmatched_actual_submodels.erase(it);
                    matched = true;
                    break;
                }
                if (mismatch_details.empty()) {
                    mismatch_details = res.second;
                }
            }

            ASSERT_TRUE(matched) << "Failed to find a matching actual submodel for expected submodel at index " << i
                                 << (mismatch_details.empty() ? "" : ". Example mismatch: ") << mismatch_details;
        }
    }

    // Check mapping info if provided
    const bool check_mapping = !param.expected_mapping._inputs_to_submodels_inputs.empty() ||
                               !param.expected_mapping._outputs_to_submodels_outputs.empty() ||
                               !param.expected_mapping._submodels_input_to_prev_output.empty();
    if (check_mapping) {
        ASSERT_EQ(param.expected_mapping._inputs_to_submodels_inputs, mapping._inputs_to_submodels_inputs);
        ASSERT_EQ(param.expected_mapping._outputs_to_submodels_outputs, mapping._outputs_to_submodels_outputs);
        ASSERT_EQ(param.expected_mapping._submodels_input_to_prev_output, mapping._submodels_input_to_prev_output);
    }

    // Check total sink count across subgraphs if expected
    if (param.expected_total_sinks > 0) {
        size_t actual_total_sinks = 0;
        for (const auto& sg : subgraphs) {
            actual_total_sinks += sg._sinks.size();
        }
        ASSERT_EQ(param.expected_total_sinks, actual_total_sinks);
    }

    // Optional stronger validation: after the baseline partition checks above,
    // verify that merging submodels reconstructs a structurally equivalent model.
    if (param.verify_merge_roundtrip) {
        std::vector<std::shared_ptr<ov::Model>> submodels;
        for (const auto& sg : subgraphs) {
            submodels.push_back(create_submodel_from_collected_subgraph(sg));
        }
        OV_ASSERT_NO_THROW(ov::hetero::merge_submodels(submodels, mapping._submodels_input_to_prev_output));
        ASSERT_EQ(1, submodels.size());
        if (param.verify_merge_compare) {
            // This compare is intentionally optional; some cases only check that merge succeeds.
            auto res = compare_functions(model_ref, submodels[0]);
            ASSERT_TRUE(res.first) << res.second;
        }
    }
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    SubgraphCollector,
    SubgraphCollectorParamTest,
    testing::Values(
        // --- split_and_merge: linear chain model (input → add → sub → reshape → result) ---
        SubgraphCollectorTestParam{
            "split_and_merge_linear_chain_model",
            create_linear_chain_model,
            {{"input", "MOCK.0"}, {"const_val", "MOCK.0"}, {"add", "MOCK.0"},
             {"sub", "MOCK.1"}, {"reshape_val", "MOCK.0"}, {"reshape", "MOCK.0"}, {"res", "MOCK.0"}},
            "",
            3,
            {"MOCK.0", "MOCK.1", "MOCK.0"},
            {{"input", 0}, {"const_val", 0}, {"add", 0},
             {"sub", 2}, {"reshape_val", 3}, {"reshape", 3}, {"res", 3}},
            {create_subgraph_add, create_subgraph_sub, create_subgraph_reshape},
            {{NodeInfo{0, 0}},
             {NodeInfo{2, 0}},
             {{NodeInfo{1, 0}, NodeInfo{0, 0}},
              {NodeInfo{2, 0}, NodeInfo{1, 0}}}},
            0,
            true,
            true,
        },
        // --- split_and_merge: diamond chain with subgraph-level cycle.
        // Topology: input → add1(MOCK.0) ─┬→ add2(MOCK.0) → sub(MOCK.1) ─┐
        //                                 └──────────────────────────────┴→ add3(MOCK.0) → reshape → res
        // Without cycle splitting, all MOCK.0 nodes (add1/add2/add3/reshape/res) would merge into one
        // subgraph that depends on sub(MOCK.1) which itself depends on the same subgraph → cycle.
        // Exercises SubgraphCollector::split_cyclic_dependencies(): MOCK.0 must be split into two
        // subgraphs ({add1, add2} and {add3, reshape, res}) yielding 3 subgraphs total.
        SubgraphCollectorTestParam{
            "split_cyclic_dependency_diamond",
            create_diamond_chain_model,
            {{"input", "MOCK.0"}, {"const_val", "MOCK.0"}, {"add1", "MOCK.0"},
             {"add2", "MOCK.0"}, {"sub", "MOCK.1"}, {"add3", "MOCK.0"},
             {"reshape_val", "MOCK.0"}, {"reshape", "MOCK.0"}, {"res", "MOCK.0"}},
            "",
            3,
            {"MOCK.0", "MOCK.1", "MOCK.0"},
            {},
            {create_subgraph_add_add, create_subgraph_sub, create_subgraph_add_reshape},
            {{NodeInfo{0, 0}},
             {NodeInfo{2, 0}},
             {{NodeInfo{1, 0}, NodeInfo{0, 0}},
              {NodeInfo{2, 0}, NodeInfo{0, 1}},
              {NodeInfo{2, 1}, NodeInfo{1, 0}}}},
            0,
            true,
            true,
        },
        // --- split_cyclic: nested cycles requiring multi-iteration fixed-point splitting.
        // Topology: input → a1(M0) → b1(M1) → a2(M0) → b2(M1) → a3(M0) → res(M0)
        //                    └────────────────────────────────────┘  (a1 also feeds a3)
        // Initial M0 group {a1,a2,a3} contains two stacked cyclic deps (via b1 and b2).
        // A single split iteration cannot resolve both; the outer fixed-point loop in
        // split_cyclic_dependencies() must run >=2 iterations. Expected: 5 subgraphs
        // alternating MOCK.0 / MOCK.1 / MOCK.0 / MOCK.1 / MOCK.0 (3×M0 + 2×M1).
        SubgraphCollectorTestParam{
            "split_cyclic_dependency_nested",
            create_nested_cyclic_chain_model,
            {{"input", "MOCK.0"}, {"c1", "MOCK.0"}, {"a1", "MOCK.0"},
             {"b1", "MOCK.1"}, {"a2", "MOCK.0"}, {"b2", "MOCK.1"},
             {"a3", "MOCK.0"}, {"res", "MOCK.0"}},
            "",
            5,
            {"MOCK.0", "MOCK.0", "MOCK.0", "MOCK.1", "MOCK.1"},
            // expected_ids: locks in the partition produced by collect_subgraphs_ids() after
            // split_cyclic_dependencies() promotes boundaries for {a2 ← c1} and {a3 ← a1}.
            // Union-find keeps-first merging causes the c1 placeholder id (1) to be absorbed
            // into the input/a1 group (id 0), so allocated ids are {0, 2, 3, 4, 5} — non-contiguous
            // by design (same pattern as split_and_merge_linear_chain_model).
            {{"input", 0}, {"c1", 0}, {"a1", 0},
             {"b1", 2}, {"a2", 3}, {"b2", 4},
             {"a3", 5}, {"res", 5}},
            {},
            {},
            0,
            true,
            true,
        },
        // --- merge_independent: param1 → add1(MOCK.0) → add2(MOCK.1) → res, plus unused param2.
        // Splits into 3 subgraphs (M0 chain, M1 chain, and an isolated subgraph for the unused
        // param2). The merge roundtrip is verified to not throw and to collapse back to a single
        // model, but structural equality (verify_merge_compare) is intentionally skipped:
        // merge_submodels does not guarantee preservation of the original Parameter ordering when
        // an isolated/unused-input subgraph is involved, so compare_functions would report a
        // benign ordering mismatch rather than a real regression.
        SubgraphCollectorTestParam{
            "merge_independent_submodel",
            create_merge_independent_model,
            {{"input1", "MOCK.0"}, {"input2", "MOCK.0"}, {"const_val1", "MOCK.0"},
             {"add1", "MOCK.0"}, {"const_val2", "MOCK.1"}, {"add2", "MOCK.1"}, {"res", "MOCK.1"}},
            "",
            3,
            {},
            {},
            {},
            {},
            0,
            true,
            false,
        },
        // --- Affinity: all same device → single subgraph ---
        SubgraphCollectorTestParam{
            "all_same_affinity",
            create_linear_chain_model,
            {},
            "MOCK.0",
            1,
            {"MOCK.0"},
            {},
            {},
            {},
            0,
            false,
            false,
        },
        // --- Affinity: each computational node on different device ---
        SubgraphCollectorTestParam{
            "all_different_affinities",
            create_linear_chain_model,
            {{"input", "MOCK.0"}, {"const_val", "MOCK.0"}, {"add", "MOCK.1"},
             {"sub", "MOCK.2"}, {"reshape_val", "MOCK.3"}, {"reshape", "MOCK.3"}, {"res", "MOCK.3"}},
            "",
            4,
            {"MOCK.0", "MOCK.1", "MOCK.2", "MOCK.3"},
            {},
            {},
            {},
            0,
            false,
            false,
        },
        // --- Affinity: result inherits affinity from its input ---
        SubgraphCollectorTestParam{
            "result_inherits_input_affinity",
            create_linear_chain_model,
            {{"input", "MOCK.0"}, {"const_val", "MOCK.0"}, {"add", "MOCK.0"},
             {"sub", "MOCK.0"}, {"reshape_val", "MOCK.0"}, {"reshape", "MOCK.0"}, {"res", "MOCK.1"}},
            "",
            1,
            {"MOCK.0"},
            {},
            {},
            {},
            0,
            false,
            false,
        },
        // --- Affinity: alternating devices A→B→C→D → 4 subgraphs ---
        SubgraphCollectorTestParam{
            "alternating_devices",
            create_alternating_chain_model,
            {{"input", "MOCK.0"}, {"c1", "MOCK.0"}, {"A", "MOCK.0"},
             {"B", "MOCK.1"}, {"C", "MOCK.0"}, {"D", "MOCK.1"}, {"res", "MOCK.1"}},
            "",
            4,
            {"MOCK.0", "MOCK.0", "MOCK.1", "MOCK.1"},
            {},
            {},
            {},
            0,
            false,
            false,
        },
        // --- Affinity: two disconnected paths, same device → 2 subgraphs ---
        SubgraphCollectorTestParam{
            "independent_paths_same_affinity",
            create_independent_paths_model,
            {},
            "MOCK.0",
            2,
            {"MOCK.0", "MOCK.0"},
            {},
            {},
            {},
            0,
            false,
            false,
        },
        // --- Affinity: diverging paths from shared node → 1 subgraph ---
        SubgraphCollectorTestParam{
            "diverging_paths_no_split",
            create_diverging_paths_model,
            {},
            "MOCK.0",
            1,
            {"MOCK.0"},
            {},
            {},
            {},
            0,
            false,
            false,
        },
        // --- Sink coverage: single-device stateful model (ReadValue/Assign on MOCK.0).
        // Verifies that Subgraph::_sinks is populated, survives create_submodel_from_collected_subgraph,
        // and is preserved by merge_submodels (compare_functions checks sink-set equality).
        SubgraphCollectorTestParam{
            "stateful_assign_readvalue_single_device",
            create_stateful_single_device_model,
            {},
            "MOCK.0",
            1,
            {"MOCK.0"},
            {},
            {},
            {},
            1,
            true,
            true,
        },
        // --- Shared constant indirect bridge: shared_const bridges A and C into one subgraph via X.
        // split_cyclic_dependencies() detects the cycle re-entry point and promotes
        // C.input(1) (from shared_const) as a boundary, splitting the M0 group into
        // {param, other_const, shared_const, A, X, res2} and {C, res1}, yielding 3 subgraphs.
        SubgraphCollectorTestParam{
            "shared_const_indirect_bridge_cycle",
            create_shared_const_indirect_bridge_cycle_model,
            {{"input", "MOCK.0"}, {"shared_const", "MOCK.0"}, {"other_const", "MOCK.0"},
             {"A", "MOCK.0"}, {"X", "MOCK.0"},
             {"B", "MOCK.1"}, {"C", "MOCK.0"}, {"res1", "MOCK.0"}, {"res2", "MOCK.0"}},
            "",
            3,
            {"MOCK.0", "MOCK.0", "MOCK.1"},
            {},
            {},
            {},
            0,
            false,
            false,
        },
        // --- Shared constant cross-device fanout: shared_const fans out to 3 consumers on 2 devices.
        // Union-Find merges same-affinity edges into one GPU subgraph containing Node_A and Node_C.
        // split_cyclic_dependencies() promotes Node_C.input(0) (from Node_B) as a boundary,
        // yielding 3 subgraphs: {param, shared_const, Node_A}(GPU), {Node_B}(CPU), {Node_C, res}(GPU).
        SubgraphCollectorTestParam{
            "shared_const_cross_device_fanout_cycle",
            create_shared_const_cross_device_fanout_model,
            {{"param", "MOCK.0"}, {"shared_const", "MOCK.0"}, {"Node_A", "MOCK.0"},
             {"Node_B", "MOCK.1"}, {"Node_C", "MOCK.0"}, {"res", "MOCK.0"}},
            "",
            3,
            {"MOCK.0", "MOCK.0", "MOCK.1"},
            {},
            {},
            {},
            0,
            true,
            true,
        },
        // --- Shared constant AS the cycle source: shared_const directly feeds B (cross-device),
        // making it the cycle producer. It also merges with the re-entry node C via the internal
        // chain shared_const→X→C. split_cyclic_dependencies() promotes C.input(1) (from X),
        // yielding 4 subgraphs: {param,A}(DEV0), {B}(DEV1), {shared_const,X,res2}(DEV0), {C,res1}(DEV0).
        SubgraphCollectorTestParam{
            "shared_const_as_cycle_source",
            create_shared_const_as_cycle_source_model,
            {{"param", "MOCK.0"}, {"shared_const", "MOCK.0"}, {"A", "MOCK.0"}, {"X", "MOCK.0"},
             {"B", "MOCK.1"}, {"C", "MOCK.0"}, {"res1", "MOCK.0"}, {"res2", "MOCK.0"}},
            "",
            4,
            {"MOCK.0", "MOCK.0", "MOCK.0", "MOCK.1"},
            {},
            {},
            {},
            0,
            true,
            true,
        }
    ),
    [](const testing::TestParamInfo<SubgraphCollectorTestParam>& info) {
        return info.param.test_name;
    });
// clang-format on


