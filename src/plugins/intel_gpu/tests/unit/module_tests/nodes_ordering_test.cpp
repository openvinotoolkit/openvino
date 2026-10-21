// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "pass_manager.h"
#include "test_utils.h"

using namespace cldnn;

namespace {

class nodes_ordering_test : public ::testing::Test {
protected:
    engine& engine_ref = tests::get_test_engine();
    program prog{engine_ref, tests::get_test_default_config(engine_ref)};

    program_node* add_node(const primitive_id& id, const std::vector<program_node*>& dependencies = {}) {
        std::shared_ptr<primitive> prim;
        if (dependencies.empty()) {
            prim = std::make_shared<input_layout>(id, layout({1, 1, 1, 1}, data_types::f32, format::bfyx));
        } else {
            std::vector<input_info> inputs;
            for (const auto* dependency : dependencies) {
                inputs.emplace_back(dependency->id());
            }
            prim = std::make_shared<concatenation>(id, inputs, 1);
        }
        auto* node = &prog.get_or_create(prim);
        if (dependencies.empty()) {
            prog.get_inputs().push_back(node);
        }
        for (auto* dependency : dependencies) {
            prog.add_connection(*dependency, *node, 0);
        }
        return node;
    }

    void set_order(const std::vector<program_node*>& nodes) {
        auto& order = prog.get_processing_order();
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
            if (*order.begin() != *it) {
                order.erase(*it);
                order.insert(*order.begin(), *it);
            }
        }
    }

    bool reference_is_correct() const {
        const auto& order = prog.get_processing_order();
        for (auto* node : order) {
            for (const auto& dependency : node->get_dependencies()) {
                if (order.get_processing_number(node) < order.get_processing_number(dependency.first)) {
                    return false;
                }
            }
        }
        return true;
    }
};

TEST_F(nodes_ordering_test, empty_and_independent_nodes) {
    auto& order = prog.get_processing_order();
    EXPECT_TRUE(order.is_correct());
    auto* a = add_node("a");
    order.calc_processing_order(prog);
    EXPECT_TRUE(order.is_correct());
    auto* b = add_node("b");
    auto* c = add_node("c");
    order.calc_processing_order(prog);
    set_order({c, a, b});
    EXPECT_TRUE(order.is_correct());
}

TEST_F(nodes_ordering_test, invalid_chain_is_recalculated) {
    auto* a = add_node("a");
    auto* b = add_node("b", {a});
    auto* c = add_node("c", {b});
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    EXPECT_TRUE(order.is_correct());
    set_order({c, b, a});
    EXPECT_FALSE(order.is_correct());
    EXPECT_EQ(order.is_correct(), reference_is_correct());
    remove_redundant_reorders{}.run(prog);
    EXPECT_TRUE(order.is_correct());
    EXPECT_EQ(order.is_correct(), reference_is_correct());
}

TEST_F(nodes_ordering_test, diamond_and_disconnected_component) {
    auto* a = add_node("a");
    auto* b = add_node("b", {a});
    auto* c = add_node("c", {a});
    auto* d = add_node("d", {b, c});
    auto* e = add_node("e");
    auto* f = add_node("f", {e});
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    set_order({e, a, c, f, b, d});
    EXPECT_TRUE(order.is_correct());
    set_order({e, a, c, f, d, b});
    EXPECT_FALSE(order.is_correct());
    EXPECT_EQ(order.is_correct(), reference_is_correct());
}

TEST_F(nodes_ordering_test, positions_are_rebuilt_after_insert_and_move) {
    auto* a = add_node("a");
    auto* b = add_node("b", {a});
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    EXPECT_TRUE(order.is_correct());
    auto* inserted = add_node("inserted");
    prog.add_connection(*inserted, *b, 0);
    order.insert_next(b, inserted);
    EXPECT_FALSE(order.is_correct());
    order.erase(inserted);
    order.insert(b, inserted);
    EXPECT_TRUE(order.is_correct());
}

TEST_F(nodes_ordering_test, positions_are_rebuilt_after_erase_and_clear) {
    auto* a = add_node("a");
    auto* b = add_node("b", {a});
    auto* unrelated = add_node("unrelated");
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    EXPECT_TRUE(order.is_correct());
    order.erase(unrelated);
    EXPECT_TRUE(order.is_correct());
    set_order({b, a});
    EXPECT_FALSE(order.is_correct());
    order.clear();
    EXPECT_TRUE(order.is_correct());
    order.calc_processing_order(prog);
    EXPECT_TRUE(order.is_correct());
}

TEST_F(nodes_ordering_test, missing_dependency_preserves_error) {
    auto* a = add_node("a");
    add_node("b", {a});
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    order.erase(a);
    EXPECT_THROW(reference_is_correct(), std::out_of_range);
    EXPECT_THROW(order.is_correct(), std::out_of_range);
}

TEST_F(nodes_ordering_test, breadth_first_rebuild_preserves_valid_order) {
    auto* a = add_node("a");
    auto* b = add_node("b", {a});
    auto* c = add_node("c", {b});
    auto* d = add_node("d", {a});
    add_node("e", {c, d});
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    EXPECT_TRUE(order.is_correct());
    order.calculate_BFS_processing_order();
    EXPECT_TRUE(order.is_correct());
    EXPECT_EQ(order.is_correct(), reference_is_correct());
}

TEST_F(nodes_ordering_test, serialized_order_is_checked_after_load) {
    auto* a = add_node("a");
    auto* b = add_node("b", {a});
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    set_order({b, a});
    ASSERT_FALSE(order.is_correct());
    std::stringstream storage;
    BinaryOutputBuffer output(storage);
    order.save(output);
    order.clear();
    EXPECT_TRUE(order.is_correct());
    BinaryInputBuffer input(storage, engine_ref);
    order.load(input, prog);
    EXPECT_FALSE(order.is_correct());
    order.calc_processing_order(prog);
    EXPECT_TRUE(order.is_correct());
}

TEST_F(nodes_ordering_test, self_dependency_preserves_existing_comparison) {
    auto* a = add_node("a");
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    prog.add_connection(*a, *a, 0);
    EXPECT_TRUE(reference_is_correct());
    EXPECT_TRUE(order.is_correct());
}

TEST_F(nodes_ordering_test, randomized_orders_match_reference) {
    std::mt19937 random(20260923);
    std::vector<program_node*> nodes;
    for (size_t i = 0; i < 64; ++i) {
        std::vector<program_node*> dependencies;
        for (size_t j = 0; j < i; ++j) {
            if (random() % 8 == 0) {
                dependencies.push_back(nodes[j]);
            }
        }
        nodes.push_back(add_node("node_" + std::to_string(i), dependencies));
    }
    auto& order = prog.get_processing_order();
    order.calc_processing_order(prog);
    EXPECT_TRUE(order.is_correct());
    for (size_t trial = 0; trial < 32; ++trial) {
        std::shuffle(nodes.begin(), nodes.end(), random);
        set_order(nodes);
        EXPECT_EQ(order.is_correct(), reference_is_correct()) << "trial " << trial;
        order.calc_processing_order(prog);
        EXPECT_TRUE(order.is_correct());
    }
}

}  // namespace
