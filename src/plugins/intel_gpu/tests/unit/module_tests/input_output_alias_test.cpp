// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/stateless_kv.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "test_utils.h"

#include <map>
#include <string>

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr auto past = "past";
constexpr auto new_token = "new_token";
constexpr auto present_len = "present_len";
constexpr auto producer = "producer";
constexpr auto net_output = "net_output";

class InputOutputAliasTest : public ::testing::Test {
protected:
    layout past_layout{ov::PartialShape{1, 2, 16, 4}, data_types::f32, format::bfyx};
    layout new_token_layout{ov::PartialShape{1, 2, 2, 4}, data_types::f32, format::bfyx};
    layout present_len_layout{ov::PartialShape{1}, data_types::i64, format::bfyx};

    topology make_topology(int output_port = 0,
                           bool converting_output = false,
                           bool other_reader = false,
                           bool intermediate = false) {
        auto stateless_kv_prim = stateless_kv(producer, {input_info(past), input_info(new_token), input_info(present_len)}, 2, true);
        stateless_kv_prim.num_outputs = 2;
        stateless_kv_prim.output_data_types = {data_types::f32, data_types::f32};
        auto present_len_memory = get_test_engine().allocate_memory(present_len_layout);
        set_values<int64_t>(present_len_memory, {16});

        topology topo{
            input_layout(past, past_layout),
            input_layout(new_token, new_token_layout),
            data(present_len, present_len_memory),
            stateless_kv_prim,
        };
        if (other_reader)
            topo.add(reorder("other_reader", input_info(past), format::bfyx, data_types::f16));

        if (intermediate) {
            topo.add(activation("middle", input_info(producer, 0), activation_func::relu));
            topo.add(reorder(net_output, input_info("middle"), format::bfyx, data_types::f32));
        } else {
            topo.add(reorder(net_output,
                             input_info(producer, output_port),
                             format::bfyx,
                             converting_output ? data_types::f16 : data_types::f32));
        }
        return topo;
    }

    // Build the real stateless_kv graph, bind its inputs, then query whether an output buffer may
    // reuse one of those inputs. Present length 16 and a two-token update produce a 16-element cache.
    bool query(topology& topo,
               const primitive_id& output_id,
               const std::string& candidate_input,
               bool separate_candidate = false) {
        ExecutionConfig config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        network net(get_test_engine(), topo, config);

        std::map<std::string, memory::ptr> inputs;
        inputs[past] = get_test_engine().allocate_memory(past_layout);
        inputs[new_token] = get_test_engine().allocate_memory(new_token_layout);

        const auto input_ids = net.get_input_ids();
        for (const auto& input_id : input_ids)
            net.set_input_data(input_id, inputs.at(input_id));

        auto candidate = separate_candidate ? get_test_engine().allocate_memory(past_layout) : inputs.at(candidate_input);
        return net.can_bind_user_output_memory(output_id, *candidate);
    }
};

TEST_F(InputOutputAliasTest, stateless_kv_accepts_its_past_buffer_for_output_zero) {
    auto topo = make_topology();

    EXPECT_TRUE(query(topo, net_output, past));
}

TEST_F(InputOutputAliasTest, stateless_kv_rejects_other_input_buffers) {
    auto topo = make_topology();

    EXPECT_FALSE(query(topo, net_output, new_token));
}

TEST_F(InputOutputAliasTest, stateless_kv_rejects_unsupported_output_port) {
    auto topo = make_topology(1);

    EXPECT_FALSE(query(topo, net_output, past));
}

TEST_F(InputOutputAliasTest, another_reader_of_past_disables_in_place_binding) {
    auto topo = make_topology(0, false, true);

    EXPECT_FALSE(query(topo, net_output, past));
}

TEST_F(InputOutputAliasTest, converting_output_reorder_is_the_writer) {
    auto topo = make_topology(0, true);

    EXPECT_FALSE(query(topo, net_output, past));
}

TEST_F(InputOutputAliasTest, intermediate_writer_does_not_inherit_stateless_kv_support) {
    auto topo = make_topology(0, false, false, true);

    EXPECT_FALSE(query(topo, net_output, past));
}

TEST_F(InputOutputAliasTest, non_overlapping_output_memory_is_allowed) {
    auto topo = make_topology();

    EXPECT_TRUE(query(topo, net_output, past, true));
}

TEST_F(InputOutputAliasTest, ordinary_reorder_rejects_input_alias_but_allows_separate_memory) {
    topology topo{
        input_layout(past, past_layout),
        reorder(net_output, input_info(past), format::bfyx, data_types::f16),
    };

    EXPECT_FALSE(query(topo, net_output, past));
    EXPECT_TRUE(query(topo, net_output, past, true));
}

}  // namespace
