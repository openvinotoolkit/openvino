// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "pass_manager.h"
#include "test_utils.h"

using namespace cldnn;

namespace {

class memory_dependency_construction : public ::testing::Test {
protected:
    engine& engine_ref = tests::get_test_engine();
    program prog{engine_ref, tests::get_test_default_config(engine_ref)};
    basic_memory_dependencies pass;

    program_node* add_node(const primitive_id& id, const std::vector<program_node*>& deps = {}, bool concat = false) {
        std::shared_ptr<primitive> prim;
        if (deps.empty()) {
            prim = std::make_shared<input_layout>(id, layout({1, 1, 1, 1}, data_types::f32, format::bfyx));
        } else if (concat) {
            std::vector<input_info> inputs;
            for (auto* dep : deps)
                inputs.emplace_back(dep->id());
            prim = std::make_shared<concatenation>(id, inputs, 1);
        } else {
            prim = std::make_shared<activation>(id, input_info(deps.front()->id()), activation_func::relu);
        }
        auto* node = &prog.get_or_create(prim);
        node->set_unique_id();
        if (deps.empty())
            prog.get_inputs().push_back(node);
        for (auto* dep : deps)
            prog.add_connection(*dep, *node, 0);
        return node;
    }

    void expect_dependencies(program_node* node, std::initializer_list<program_node*> deps) {
        std::vector<uint32_t> expected;
        for (const auto* dep : deps)
            expected.push_back(static_cast<uint32_t>(dep->get_unique_id()));
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(node->get_memory_dependencies(), expected);
        for (const auto id : expected)
            EXPECT_TRUE(node->has_memory_dependency(id));
    }
};

TEST_F(memory_dependency_construction, self_and_repeated_bidirectional_additions) {
    auto* a = add_node("a");
    auto* b = add_node("b");
    pass.add_memory_dependency(a, a);
    expect_dependencies(a, {});
    a->add_memory_dependency(*a);
    for (size_t i = 0; i < 100; ++i) {
        pass.add_memory_dependency(a, b);
        pass.add_memory_dependency(b, a);
    }
    expect_dependencies(a, {a, b});
    expect_dependencies(b, {a});
}

TEST_F(memory_dependency_construction, constant_nodes_are_excluded_from_node_additions) {
    auto* a = add_node("a");
    auto memory = engine_ref.allocate_memory(layout({1, 1, 1, 1}, data_types::f32, format::bfyx));
    auto* constant = &prog.get_or_create(std::make_shared<data>("constant", memory));
    constant->set_unique_id();
    ASSERT_TRUE(constant->is_constant());
    pass.add_memory_dependency(a, constant);
    pass.add_memory_dependency(constant, a);
    expect_dependencies(a, {});
    expect_dependencies(constant, {});
    // Explicit ID lists preserve their existing behavior, including IDs outside this program.
    a->add_memory_dependency(std::vector<size_t>{constant->get_unique_id(), 1000000});
    EXPECT_TRUE(a->has_memory_dependency(static_cast<uint32_t>(constant->get_unique_id())));
    EXPECT_TRUE(a->has_memory_dependency(1000000));
}

TEST_F(memory_dependency_construction, skippable_chain_expands_to_buffer_owners) {
    auto* root = add_node("root");
    auto* view1 = add_node("view1", {root});
    auto* view2 = add_node("view2", {view1});
    auto* user = add_node("user", {view2});
    for (auto* view : {view1, view2}) {
        view->can_be_optimized(true);
        view->set_runtime_skippable(true);
    }
    pass.add_memory_dependency(user, view2);
    expect_dependencies(user, {root, view1, view2});
    expect_dependencies(root, {user});
    expect_dependencies(view1, {user});
    pass.add_memory_dependency(user, view2);
    expect_dependencies(user, {root, view1, view2});
}

TEST_F(memory_dependency_construction, optimized_concat_restricts_predecessor_inputs) {
    auto* root = add_node("root");
    auto* pred = add_node("pred", {root});
    auto* concat = add_node("concat", {pred}, true);
    concat->can_be_optimized(true);
    concat->set_runtime_skippable(true);
    pass.add_memory_dependency(concat, pred);
    expect_dependencies(concat, {root, pred});
    EXPECT_TRUE(root->has_memory_dependency(static_cast<uint32_t>(concat->get_unique_id())));
}

TEST_F(memory_dependency_construction, lora_restricts_main_flow_in_both_directions) {
    auto* root = add_node("root");
    auto* user = add_node("user");
    std::vector<input_info> inputs(5, input_info(root->id()));
    auto* lora_node = &prog.get_or_create(std::make_shared<lora>("lora", inputs, false));
    lora_node->set_unique_id();
    for (size_t i = 0; i < inputs.size(); ++i)
        prog.add_connection(*root, *lora_node, 0);
    pass.add_memory_dependency(user, lora_node);
    expect_dependencies(user, {root, lora_node});
    expect_dependencies(root, {user});
}

TEST_F(memory_dependency_construction, dense_restrictions_keep_immediate_membership) {
    std::vector<program_node*> nodes;
    for (size_t i = 0; i < 192; ++i)
        nodes.push_back(add_node("node" + std::to_string(i)));
    std::vector<uint32_t> ids;
    for (auto* node : nodes) {
        node->add_memory_dependency(*node);
        ids.push_back(static_cast<uint32_t>(node->get_unique_id()));
    }
    std::sort(ids.begin(), ids.end());
    std::mt19937 rng(42);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    for (auto* node : nodes) {
        for (auto* dep : nodes) {
            pass.add_memory_dependency(node, dep);
            ASSERT_TRUE(node->has_memory_dependency(static_cast<uint32_t>(dep->get_unique_id())));
        }
    }
    for (auto* node : nodes)
        EXPECT_EQ(node->get_memory_dependencies(), ids);
}

TEST_F(memory_dependency_construction, serialize_dense_set_then_load_and_extend) {
    auto* node = add_node("node");
    std::vector<uint32_t> ids(4096);
    std::iota(ids.begin(), ids.end(), 10000);
    node->add_memory_dependency(std::vector<size_t>(ids.begin(), ids.end()));
    std::stringstream storage;
    BinaryOutputBuffer output(storage);
    node->save(output);
    node->add_memory_dependency(std::vector<size_t>{9999});
    BinaryInputBuffer input(storage, engine_ref);
    node->load(input);
    EXPECT_FALSE(node->has_memory_dependency(9999));
    EXPECT_EQ(node->get_memory_dependencies(), ids);
    node->add_memory_dependency(std::vector<size_t>{9999, 10000, 14096});
    EXPECT_EQ(node->get_memory_dependencies().size(), ids.size() + 2);
}

TEST_F(memory_dependency_construction, id_list_preserves_range_check) {
    auto* node = add_node("node");
    const auto max_id = std::numeric_limits<uint32_t>::max();
    node->add_memory_dependency(std::vector<size_t>{0, max_id, max_id});
    EXPECT_EQ(node->get_memory_dependencies(), (std::vector<uint32_t>{0, max_id}));
    if (std::numeric_limits<size_t>::max() > max_id) {
        EXPECT_ANY_THROW(node->add_memory_dependency(std::vector<size_t>{static_cast<size_t>(max_id) + 1}));
    }
}

// A set-based oracle keeps intermediate membership independent of the production representation.
class restriction_reference {
public:
    std::unordered_map<program_node*, std::set<uint32_t>> values;

    void insert(program_node* node, program_node* dep) {
        if (node->may_use_mempool() && dep->may_use_mempool())
            values[node].insert(static_cast<uint32_t>(dep->get_unique_id()));
    }

    void add(program_node* node, program_node* dep) {
        if (node == dep || values[node].count(static_cast<uint32_t>(dep->get_unique_id())) != 0)
            return;
        const bool direct = std::any_of(node->get_dependencies().begin(), node->get_dependencies().end(), [dep](const std::pair<program_node*, int32_t>& edge) {
            return edge.first == dep;
        });
        if (node->is_type<concatenation>() && node->can_be_optimized() && node->is_runtime_skippable() && direct) {
            for (const auto& edge : dep->get_dependencies()) {
                add(node, edge.first);
                add(edge.first, node);
            }
        }
        if (dep->is_type<lora>()) {
            insert(node, &dep->get_dependency(0));
            insert(&dep->get_dependency(0), node);
        }
        if ((!dep->can_be_optimized() || !dep->is_runtime_skippable()) &&
            ((node->can_be_optimized() && !node->is_runtime_skippable()) || !dep->can_be_optimized())) {
            insert(node, dep);
        } else {
            if (node->is_runtime_skippable() || dep->is_runtime_skippable() || dep->can_be_optimized())
                insert(node, dep);
            for (const auto& edge : dep->get_dependencies()) {
                add(node, edge.first);
                add(edge.first, node);
            }
        }
    }
};

TEST_F(memory_dependency_construction, random_dag_recursive_expansion_matches_sets) {
    std::mt19937 rng(2026);
    std::vector<program_node*> nodes;
    restriction_reference reference;
    for (size_t i = 0; i < 192; ++i) {
        std::vector<program_node*> deps;
        if (i != 0) {
            deps.push_back(nodes[rng() % i]);
            if (i > 2)
                deps.push_back(nodes[rng() % i]);
        }
        auto* node = add_node("node" + std::to_string(i), deps, true);
        node->can_be_optimized(i % 3 != 0);
        node->set_runtime_skippable(i % 4 != 0);
        nodes.push_back(node);
    }
    for (size_t i = 0; i < 4000; ++i) {
        auto* node = nodes[rng() % nodes.size()];
        auto* dep = nodes[rng() % nodes.size()];
        reference.add(node, dep);
        pass.add_memory_dependency(node, dep);
        ASSERT_EQ(node->has_memory_dependency(static_cast<uint32_t>(dep->get_unique_id())),
                  reference.values[node].count(static_cast<uint32_t>(dep->get_unique_id())) != 0);
    }
    for (auto* node : nodes) {
        const auto& expected = reference.values[node];
        EXPECT_EQ(node->get_memory_dependencies(), std::vector<uint32_t>(expected.begin(), expected.end())) << node->id();
    }
}

}  // namespace
