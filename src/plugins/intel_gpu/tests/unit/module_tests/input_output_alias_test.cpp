// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/stateless_kv.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "test_utils.h"

#include <string>
#include <vector>

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
                           bool intermediate = false,
                           bool late_reader = false) {
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
        // Reading stateless_kv's output 1 forces this reader of past to run after stateless_kv.
        if (late_reader)
            topo.add(eltwise("late_reader", input_info(past), input_info(producer, 1), eltwise_mode::sum));

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

    // Build the real stateless_kv graph and query whether an output buffer may reuse one of its inputs.
    // Present length 16 and a two-token update produce a 16-element cache.
    bool query(topology& topo, const primitive_id& output_id, const primitive_id& input_id) {
        ExecutionConfig config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        // network::may_alias() always rejects on out-of-order queues.
        config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
        network net(get_test_engine(), topo, config);
        return net.may_alias(output_id, input_id);
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

// The other reader of past is scheduled before stateless_kv, so the in-place update can't affect it.
TEST_F(InputOutputAliasTest, earlier_reader_of_past_keeps_in_place_binding) {
    auto topo = make_topology(0, false, true);

    EXPECT_TRUE(query(topo, net_output, past));
}

TEST_F(InputOutputAliasTest, later_reader_of_past_disables_in_place_binding) {
    auto topo = make_topology(0, false, false, false, true);

    EXPECT_FALSE(query(topo, net_output, past));
}

// past has no reader left once the output is written; stateless_kv then computes into its own buffer.
TEST_F(InputOutputAliasTest, converting_output_reorder_after_last_past_reader_is_aliasable) {
    auto topo = make_topology(0, true);

    EXPECT_TRUE(query(topo, net_output, past));
}

TEST_F(InputOutputAliasTest, intermediate_writer_after_last_past_reader_is_aliasable) {
    auto topo = make_topology(0, false, false, true);

    EXPECT_TRUE(query(topo, net_output, past));
}

TEST_F(InputOutputAliasTest, ordinary_reorder_rejects_input_alias) {
    topology topo{
        input_layout(past, past_layout),
        reorder(net_output, input_info(past), format::bfyx, data_types::f16),
    };

    EXPECT_FALSE(query(topo, net_output, past));
}

constexpr auto in0 = "in0";
constexpr auto in1 = "in1";

class InputOutputAliasGenericTest : public ::testing::TestWithParam<bool> {
protected:
    layout data_layout{ov::PartialShape{1, 4}, data_types::f32, format::bfyx};

    // The parameter toggles the memory pool: aliasing decisions must not depend on it.
    bool may_alias(topology& topo,
                   const primitive_id& output_id,
                   const primitive_id& input_id,
                   std::vector<std::string> outputs = {},
                   QueueTypes queue_type = QueueTypes::in_order,
                   bool* queue_type_supported = nullptr) {
        ExecutionConfig config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::enable_memory_pool(GetParam()));
        config.set_property(ov::intel_gpu::queue_type(queue_type));
        // oneDNN forces an in-order queue; it must be user-set to stay disabled on XMX devices.
        if (queue_type == QueueTypes::out_of_order)
            config.set_user_property(ov::intel_gpu::use_onednn(false));
        if (!outputs.empty())
            config.set_property(ov::intel_gpu::custom_outputs(outputs));
        network net(get_test_engine(), topo, config);
        const bool queue_type_matches = net.get_stream().get_queue_type() == queue_type;
        if (queue_type_supported) {
            *queue_type_supported = queue_type_matches;
            if (!queue_type_matches)
                return false;
        } else {
            EXPECT_EQ(net.get_stream().get_queue_type(), queue_type);
        }
        return net.may_alias(output_id, input_id);
    }
};

// The output writer reads the input, so it can't write into the input's buffer.
TEST_P(InputOutputAliasGenericTest, writer_reading_input_is_not_aliasable) {
    topology topo{
        input_layout(in0, data_layout),
        input_layout(in1, data_layout),
        eltwise(net_output, input_info(in0), input_info(in1), eltwise_mode::sum),
    };

    EXPECT_FALSE(may_alias(topo, net_output, in0));
    EXPECT_FALSE(may_alias(topo, net_output, in1));
}

TEST_P(InputOutputAliasGenericTest, converting_reorder_of_input_is_not_aliasable) {
    topology topo{
        input_layout(in0, data_layout),
        reorder(net_output, input_info(in0), format::bfyx, data_types::f16),
    };

    EXPECT_FALSE(may_alias(topo, net_output, in0));
}

TEST_P(InputOutputAliasGenericTest, pass_through_of_input_is_not_aliasable) {
    topology topo{
        input_layout(in0, layout{ov::PartialShape{1, -1}, data_types::f32, format::bfyx}),
        reorder(net_output, input_info(in0), format::bfyx, data_types::f32),
    };

    EXPECT_FALSE(may_alias(topo, net_output, in0));
}

// in0 has no reader left once the output writer runs, so the writer may reuse in0's buffer.
TEST_P(InputOutputAliasGenericTest, writer_after_last_input_reader_is_aliasable) {
    topology topo{
        input_layout(in0, data_layout),
        input_layout(in1, data_layout),
        activation("reader", input_info(in0), activation_func::relu),
        eltwise(net_output, input_info("reader"), input_info(in1), eltwise_mode::sum),
    };

    EXPECT_TRUE(may_alias(topo, net_output, in0));
    EXPECT_FALSE(may_alias(topo, net_output, in1));
}

// Independent readers of an input aren't ordered before the writer on an out-of-order queue.
TEST_P(InputOutputAliasGenericTest, out_of_order_queue_is_not_aliasable) {
    topology topo{
        input_layout(in0, data_layout),
        input_layout(in1, data_layout),
        activation("reader", input_info(in0), activation_func::relu),
        eltwise(net_output, input_info("reader"), input_info(in1), eltwise_mode::sum),
    };

    bool out_of_order_queue_supported = false;
    const auto aliasable = may_alias(topo, net_output, in0, {}, QueueTypes::out_of_order, &out_of_order_queue_supported);
    if (!out_of_order_queue_supported)
        GTEST_SKIP() << "Out-of-order queues are unavailable for this device/configuration";

    EXPECT_FALSE(aliasable);
}

// A later reader of in0 still needs its data after the output writer ran, wherever in0 sits in the processing order.
TEST_P(InputOutputAliasGenericTest, writer_before_last_input_reader_is_not_aliasable) {
    topology topo{
        input_layout(in0, data_layout),
        input_layout(in1, data_layout),
        activation(net_output, input_info(in1), activation_func::relu),
        eltwise("late_reader", input_info(in0), input_info(net_output), eltwise_mode::sum),
    };

    EXPECT_FALSE(may_alias(topo, net_output, in0, {net_output, "late_reader"}));
}

INSTANTIATE_TEST_SUITE_P(smoke, InputOutputAliasGenericTest, ::testing::Bool(), [](const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "MemoryPool" : "NoMemoryPool";
});

}  // namespace
