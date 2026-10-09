// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/op/stateless_kv.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include <algorithm>
#include <numeric>

#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/stateless_kv.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "stateless_kv_inst.h"

using namespace cldnn;
using namespace ::tests;

struct stateless_kv_runtime_params {
    int64_t new_token_len;
    int64_t seq_len;
    bool is_seq_len_present_len;
    bool has_pos_idx;
    bool use_caching = false;
};

class stateless_kv_runtime : public testing::TestWithParam<stateless_kv_runtime_params> {};

namespace {

topology make_stateless_kv_topology(const layout& new_token_layout, data_types result_type = data_types::f32) {
    auto kv = stateless_kv("stateless_kv", {input_info("past"), input_info("new_token"), input_info("present_len")}, 2, true);
    kv.num_outputs = 2;
    kv.output_data_types = {data_types::f32, data_types::f32};
    return topology(input_layout("past", layout{ov::PartialShape{1, 2, -1, 4}, data_types::f32, format::bfyx}),
                    input_layout("new_token", new_token_layout),
                    input_layout("present_len", layout{ov::PartialShape{1}, data_types::i64, format::bfyx}),
                    kv,
                    reorder("result", input_info("stateless_kv", 0), format::bfyx, result_type));
}

std::vector<float> expected_concatenated_cache(const std::vector<float>& past_values, const std::vector<float>& token_values,
                                               const ov::Shape& past_shape, size_t output_capacity, size_t logical_past_len) {
    const size_t planes = past_shape[0] * past_shape[1];
    const size_t past_capacity = past_shape[2];
    const size_t head_size = past_shape[3];
    const size_t token_len = token_values.size() / (planes * head_size);
    std::vector<float> expected(planes * output_capacity * head_size);
    for (size_t plane = 0; plane < planes; ++plane) {
        std::copy_n(past_values.begin() + plane * past_capacity * head_size,
                    past_capacity * head_size,
                    expected.begin() + plane * output_capacity * head_size);
        std::copy_n(token_values.begin() + plane * token_len * head_size,
                    token_len * head_size,
                    expected.begin() + (plane * output_capacity + logical_past_len) * head_size);
    }
    return expected;
}

}  // namespace

TEST_P(stateless_kv_runtime, output_memory_reuse) {
    const auto& params = GetParam();
    auto& engine = get_test_engine();

    constexpr int64_t past_capacity = 16;
    constexpr int64_t batch = 1;
    constexpr int64_t heads = 2;
    constexpr int64_t head_size = 4;
    const auto logical_past_len = params.is_seq_len_present_len ? params.seq_len - params.new_token_len : params.seq_len;
    const auto logical_present_len = params.is_seq_len_present_len ? params.seq_len : params.seq_len + params.new_token_len;
    const auto output_capacity = std::max(past_capacity, logical_present_len);
    const bool reuse_past_buffer = logical_present_len <= past_capacity;
    const auto past_layout = layout{ov::PartialShape{batch, heads, past_capacity, head_size}, data_types::f32, format::bfyx};
    const auto new_token_layout = layout{ov::PartialShape{batch, heads, params.new_token_len, head_size}, data_types::f32, format::bfyx};
    const auto seq_len_layout = layout{ov::PartialShape{1}, data_types::i64, format::bfyx};
    const auto pos_idx_layout = layout{ov::PartialShape{params.new_token_len}, data_types::i32, format::bfyx};

    std::vector<input_info> stateless_kv_inputs = {
        input_info("past"),
        input_info("new_token"),
        input_info("present_len"),
    };
    if (params.has_pos_idx) {
        stateless_kv_inputs.emplace_back("pos_idx");
    }
    auto stateless_kv_prim = stateless_kv("stateless_kv", stateless_kv_inputs, 2, params.is_seq_len_present_len);
    stateless_kv_prim.num_outputs = 2;
    stateless_kv_prim.output_data_types = {data_types::f32, data_types::f32};

    topology topology;
    topology.add(input_layout("past", layout{ov::PartialShape{1, 2, -1, 4}, data_types::f32, format::bfyx}));
    topology.add(input_layout("new_token", new_token_layout));
    topology.add(input_layout("present_len", seq_len_layout));
    if (params.has_pos_idx) {
        topology.add(input_layout("pos_idx", pos_idx_layout));
    }
    topology.add(stateless_kv_prim);
    topology.add(reorder("result", input_info("stateless_kv", 0), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto network_ptr = get_network(engine, topology, config, get_test_stream_ptr(), params.use_caching);
    auto& network = *network_ptr;
    auto past = engine.allocate_memory(past_layout);
    auto new_token = engine.allocate_memory(new_token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    auto pos_idx = engine.allocate_memory(pos_idx_layout);
    std::vector<float> past_values(past_layout.count());
    std::vector<float> new_token_values(new_token_layout.count());
    for (size_t i = 0; i < past_values.size(); ++i) {
        past_values[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < new_token_values.size(); ++i) {
        new_token_values[i] = static_cast<float>(1000 + i);
    }
    set_values(past, past_values);
    set_values(new_token, new_token_values);
    set_values<int64_t>(seq_len, {params.seq_len});
    std::vector<int32_t> pos_idx_values(static_cast<size_t>(params.new_token_len));
    std::iota(pos_idx_values.begin(), pos_idx_values.end(), static_cast<int32_t>(logical_past_len));
    set_values(pos_idx, pos_idx_values);

    const auto expected = expected_concatenated_cache(past_values, new_token_values, past_layout.get_shape(), output_capacity, logical_past_len);

    network.set_input_data("past", past);
    network.set_input_data("new_token", new_token);
    network.set_input_data("present_len", seq_len);
    if (params.has_pos_idx) {
        network.set_input_data("pos_idx", pos_idx);
    }
    network.set_output_memory("result", past);

    const auto outputs = network.execute();
    const auto stateless_kv_inst = network.get_primitive("stateless_kv");
    const auto output0 = stateless_kv_inst->output_memory_ptr(0);
    const auto output1 = stateless_kv_inst->output_memory_ptr(1);
    const auto result = outputs.at("result").get_memory();

    ASSERT_NE(output0, nullptr);
    ASSERT_NE(output1, nullptr);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(engine.is_the_same_buffer(*output0, *result));
    EXPECT_TRUE(engine.is_the_same_buffer(*output1, *output0));
    EXPECT_EQ(engine.is_the_same_buffer(*output0, *past), reuse_past_buffer);
    EXPECT_EQ(output0->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, output_capacity, head_size}));
    EXPECT_EQ(output1->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, logical_present_len, head_size}));
    ASSERT_NE(output0->get_mem_tracker(), nullptr);
    ASSERT_GE(output0->get_mem_tracker()->size(), output0->get_layout().bytes_count());

    mem_lock<float, mem_lock_type::read> output_lock(output0, get_test_stream());
    const std::vector<float> actual(output_lock.begin(), output_lock.begin() + expected.size());
    EXPECT_THAT(actual, testing::ElementsAreArray(expected));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         stateless_kv_runtime,
                         testing::Values(stateless_kv_runtime_params{2, 15, true, false},
                                         stateless_kv_runtime_params{1, 16, true, false},
                                         stateless_kv_runtime_params{2, 18, true, false},
                                         stateless_kv_runtime_params{2, 13, false, false},
                                         stateless_kv_runtime_params{1, 15, false, false},
                                         stateless_kv_runtime_params{2, 15, false, false},
                                         stateless_kv_runtime_params{2, 16, false, false},
                                         stateless_kv_runtime_params{2, 15, true, true},
                                         stateless_kv_runtime_params{1, 16, true, true},
                                         stateless_kv_runtime_params{2, 18, true, true},
                                         stateless_kv_runtime_params{2, 13, false, true},
                                         stateless_kv_runtime_params{1, 15, false, true},
                                         stateless_kv_runtime_params{2, 16, false, true},
                                         stateless_kv_runtime_params{2, 15, true, false, true},
                                         stateless_kv_runtime_params{2, 15, false, false, true},
                                         stateless_kv_runtime_params{2, 15, true, true, true}),
                         [](const testing::TestParamInfo<stateless_kv_runtime_params>& info) {
                             const auto& params = info.param;
                             return std::string{params.is_seq_len_present_len ? "PresentSeq" : "PastSeq"} + std::to_string(params.seq_len) + "_NewToken" +
                                    std::to_string(params.new_token_len) + (params.has_pos_idx ? "_Scatter" : "_Concat") +
                                    (params.use_caching ? "_cached" : "");
                         });

TEST(stateless_kv_runtime, reallocation_across_executions) {
    auto& engine = get_test_engine();

    constexpr int64_t batch = 1;
    constexpr int64_t heads = 2;
    constexpr int64_t initial_capacity = 16;
    constexpr int64_t new_token_len = 2;
    constexpr int64_t head_size = 4;
    const auto initial_past_layout = layout{ov::PartialShape{batch, heads, initial_capacity, head_size}, data_types::f32, format::bfyx};
    const auto new_token_layout = layout{ov::PartialShape{batch, heads, new_token_len, head_size}, data_types::f32, format::bfyx};
    const auto seq_len_layout = layout{ov::PartialShape{1}, data_types::i64, format::bfyx};

    auto topo = make_stateless_kv_topology(new_token_layout);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topo, config);
    auto current_past = engine.allocate_memory(initial_past_layout);
    auto new_token = engine.allocate_memory(new_token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    std::vector<float> current_values(initial_past_layout.count());
    for (size_t i = 0; i < current_values.size(); ++i) {
        current_values[i] = static_cast<float>(i);
    }
    set_values(current_past, current_values);
    network.set_output_memory("result", current_past);

    int64_t current_capacity = initial_capacity;
    auto run_and_verify = [&](int64_t present_len, float token_base, bool expect_reuse) {
        const auto logical_past_len = present_len - new_token_len;
        const auto output_capacity = std::max(current_capacity, present_len);
        std::vector<float> new_token_values(new_token_layout.count());
        for (size_t i = 0; i < new_token_values.size(); ++i) {
            new_token_values[i] = token_base + static_cast<float>(i);
        }
        set_values(new_token, new_token_values);
        set_values<int64_t>(seq_len, {present_len});

        auto expected = expected_concatenated_cache(current_values, new_token_values, current_past->get_layout().get_shape(),
                                                    output_capacity, logical_past_len);

        network.set_input_data("past", current_past);
        network.set_input_data("new_token", new_token);
        network.set_input_data("present_len", seq_len);

        const auto outputs = network.execute();
        const auto stateless_kv_inst = network.get_primitive("stateless_kv");
        const auto output0 = stateless_kv_inst->output_memory_ptr(0);
        const auto output1 = stateless_kv_inst->output_memory_ptr(1);
        const auto result = outputs.at("result").get_memory();

        ASSERT_NE(output0, nullptr);
        ASSERT_NE(output1, nullptr);
        ASSERT_NE(result, nullptr);
        EXPECT_TRUE(engine.is_the_same_buffer(*output0, *result));
        EXPECT_TRUE(engine.is_the_same_buffer(*output1, *output0));
        EXPECT_EQ(engine.is_the_same_buffer(*output0, *current_past), expect_reuse);
        EXPECT_EQ(output0->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, output_capacity, head_size}));
        EXPECT_EQ(output1->get_layout().get_partial_shape(), ov::PartialShape({batch, heads, present_len, head_size}));
        ASSERT_NE(output0->get_mem_tracker(), nullptr);
        ASSERT_GE(output0->get_mem_tracker()->size(), output0->get_layout().bytes_count());

        mem_lock<float, mem_lock_type::read> output_lock(output0, get_test_stream());
        const std::vector<float> actual(output_lock.begin(), output_lock.begin() + expected.size());
        EXPECT_THAT(actual, testing::ElementsAreArray(expected));

        current_past = output0;
        current_values = std::move(expected);
        current_capacity = output_capacity;
    };

    run_and_verify(15, 1000.0f, true);
    run_and_verify(18, 2000.0f, false);
    run_and_verify(17, 3000.0f, true);
    run_and_verify(20, 4000.0f, false);
}

TEST(stateless_kv_runtime, caller_output_binding_respects_inplace_contract) {
    auto& engine = get_test_engine();
    const layout past_layout{ov::Shape{1, 2, 16, 4}, data_types::f32, format::bfyx};
    const layout token_layout{ov::Shape{1, 2, 1, 4}, data_types::f32, format::bfyx};
    const layout seq_len_layout{ov::Shape{1}, data_types::i64, format::bfyx};

    auto topo = make_stateless_kv_topology(token_layout);
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    // network::may_alias() always rejects on out-of-order queues.
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    network net(engine, topo, config);

    auto past = engine.allocate_memory(past_layout, allocation_type::usm_host);
    auto token = engine.allocate_memory(token_layout, allocation_type::usm_host);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    std::vector<float> expected(past_layout.count(), 1.0f);
    set_values(past, expected);
    set_values(token, std::vector<float>(token_layout.count(), 3.0f));
    set_values<int64_t>(seq_len, {5});
    net.set_input_data("past", past);
    net.set_input_data("new_token", token);
    net.set_input_data("present_len", seq_len);

    EXPECT_TRUE(net.may_alias("result", "past"));
    EXPECT_FALSE(net.may_alias("result", "new_token"));

    net.set_output_memory("result", past, true);
    auto outputs = net.execute();
    auto result = outputs.at("result").get_memory();
    auto kv_output = net.get_primitive("stateless_kv")->output_memory_ptr(0);
    ASSERT_NE(result, nullptr);
    ASSERT_NE(kv_output, nullptr);
    EXPECT_TRUE(engine.is_the_same_buffer(*result, *past));
    EXPECT_TRUE(engine.is_the_same_buffer(*kv_output, *past));

    for (size_t head = 0; head < 2; ++head)
        std::fill_n(expected.begin() + head * 16 * 4 + 4 * 4, 4, 3.0f);
    mem_lock<float, mem_lock_type::read> actual(past, get_test_stream());
    EXPECT_THAT(std::vector<float>(actual.begin(), actual.begin() + expected.size()), testing::ElementsAreArray(expected));
}

// A reader of past scheduled after stateless_kv restricts it from sharing past's buffer, so a Result bound
// to past (bypassing network::can_bind_user_output_memory()) must not make stateless_kv update past in place.
TEST(stateless_kv_runtime, restricted_past_forces_private_present) {
    auto& engine = get_test_engine();
    const layout past_layout{ov::Shape{1, 2, 16, 4}, data_types::f32, format::bfyx};
    const layout token_layout{ov::Shape{1, 2, 1, 4}, data_types::f32, format::bfyx};
    const layout seq_len_layout{ov::Shape{1}, data_types::i64, format::bfyx};

    auto topo = make_stateless_kv_topology(token_layout);
    // Reading output 1 orders this reader of past after stateless_kv.
    topo.add(eltwise("late_reader", input_info("past"), input_info("stateless_kv", 1), eltwise_mode::sum));
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    network net(engine, topo, config);

    auto past = engine.allocate_memory(past_layout, allocation_type::usm_host);
    auto token = engine.allocate_memory(token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    set_values(past, std::vector<float>(past_layout.count(), 1.0f));
    set_values(token, std::vector<float>(token_layout.count(), 3.0f));
    // Full capacity keeps output 1 the same shape as past for the eltwise.
    set_values<int64_t>(seq_len, {16});
    net.set_input_data("past", past);
    net.set_input_data("new_token", token);
    net.set_input_data("present_len", seq_len);
    EXPECT_FALSE(net.may_alias("result", "past"));

    net.set_output_memory("result", past, true);
    net.execute();
    auto kv_output = net.get_primitive("stateless_kv")->output_memory_ptr(0);
    ASSERT_NE(kv_output, nullptr);
    EXPECT_FALSE(engine.is_the_same_buffer(*kv_output, *past));
}

// Growing present over a caller buffer that also backs past changes the head pitch, so stateless_kv must not
// update in place: it computes into its own buffer and result copies that into the caller buffer.
TEST(stateless_kv_runtime, concat_growth_with_aliased_caller_buffer) {
    auto& engine = get_test_engine();

    constexpr int64_t heads = 2;
    constexpr int64_t head_size = 4;
    constexpr int64_t past_capacity = 16;
    constexpr int64_t new_token_len = 2;
    constexpr int64_t present_len = past_capacity + new_token_len;
    const layout past_layout{ov::PartialShape{1, heads, past_capacity, head_size}, data_types::f32, format::bfyx};
    const layout present_layout{ov::PartialShape{1, heads, present_len, head_size}, data_types::f32, format::bfyx};
    const layout token_layout{ov::PartialShape{1, heads, new_token_len, head_size}, data_types::f32, format::bfyx};
    const layout seq_len_layout{ov::PartialShape{1}, data_types::i64, format::bfyx};

    auto topo = make_stateless_kv_topology(token_layout);
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network net(engine, topo, config);

    auto caller_buffer = engine.allocate_memory(present_layout, allocation_type::usm_host);
    auto past = engine.reinterpret_buffer(*caller_buffer, past_layout);
    auto new_token = engine.allocate_memory(token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    std::vector<float> past_values(past_layout.count());
    std::iota(past_values.begin(), past_values.end(), 0.0f);
    std::vector<float> token_values(token_layout.count());
    std::iota(token_values.begin(), token_values.end(), 1000.0f);
    set_values(past, past_values);
    set_values(new_token, token_values);
    set_values<int64_t>(seq_len, {present_len});

    const auto expected = expected_concatenated_cache(past_values, token_values, past_layout.get_shape(), present_len, past_capacity);

    net.set_input_data("past", past);
    net.set_input_data("new_token", new_token);
    net.set_input_data("present_len", seq_len);
    net.set_output_memory("result", caller_buffer, true);

    auto outputs = net.execute();
    auto result = outputs.at("result").get_memory();
    auto kv_output = net.get_primitive("stateless_kv")->output_memory_ptr(0);
    ASSERT_NE(result, nullptr);
    ASSERT_NE(kv_output, nullptr);
    EXPECT_TRUE(engine.is_the_same_buffer(*result, *caller_buffer));
    EXPECT_FALSE(engine.is_the_same_buffer(*kv_output, *caller_buffer));

    mem_lock<float, mem_lock_type::read> actual(caller_buffer, get_test_stream());
    EXPECT_THAT(std::vector<float>(actual.begin(), actual.begin() + expected.size()), testing::ElementsAreArray(expected));
}

// stateless_kv must not write f32 present into the f16 buffer of a converting Result.
TEST(stateless_kv_runtime, converting_result_does_not_share_present_buffer) {
    auto& engine = get_test_engine();

    constexpr int64_t heads = 2;
    constexpr int64_t capacity = 16;
    constexpr int64_t head_size = 4;
    constexpr int64_t present_len = 5;
    const layout past_layout{ov::PartialShape{1, heads, capacity, head_size}, data_types::f32, format::bfyx};
    const layout token_layout{ov::PartialShape{1, heads, 1, head_size}, data_types::f32, format::bfyx};
    const layout seq_len_layout{ov::PartialShape{1}, data_types::i64, format::bfyx};

    auto topo = make_stateless_kv_topology(token_layout, data_types::f16);
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network net(engine, topo, config);

    auto past = engine.allocate_memory(past_layout);
    auto new_token = engine.allocate_memory(token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    std::vector<float> past_values(past_layout.count());
    std::iota(past_values.begin(), past_values.end(), 0.0f);
    set_values(past, past_values);
    set_values(new_token, std::vector<float>(token_layout.count(), 1000.0f));
    set_values<int64_t>(seq_len, {present_len});

    auto expected = past_values;
    for (int64_t head = 0; head < heads; ++head)
        std::fill_n(expected.begin() + (head * capacity + present_len - 1) * head_size, head_size, 1000.0f);

    net.set_input_data("past", past);
    net.set_input_data("new_token", new_token);
    net.set_input_data("present_len", seq_len);

    auto outputs = net.execute();
    auto result = outputs.at("result").get_memory();
    auto kv_output = net.get_primitive("stateless_kv")->output_memory_ptr(0);
    ASSERT_NE(result, nullptr);
    ASSERT_NE(kv_output, nullptr);
    EXPECT_FALSE(engine.is_the_same_buffer(*kv_output, *result));

    mem_lock<ov::float16, mem_lock_type::read> actual(result, get_test_stream());
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t i = 0; i < present_len * head_size; ++i) {
            const auto offset = static_cast<size_t>(head * capacity * head_size + i);
            ASSERT_FLOAT_EQ(static_cast<float>(actual[offset]), expected[offset]) << "index=" << offset;
        }
    }
}

// The private present buffer of a converting Result is kept across executions while it is large enough.
TEST(stateless_kv_runtime, converting_result_reuses_private_present_buffer) {
    auto& engine = get_test_engine();

    constexpr int64_t heads = 2;
    constexpr int64_t head_size = 4;
    const layout token_layout{ov::PartialShape{1, heads, 1, head_size}, data_types::f32, format::bfyx};
    const layout seq_len_layout{ov::PartialShape{1}, data_types::i64, format::bfyx};

    auto topo = make_stateless_kv_topology(token_layout, data_types::f16);
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network net(engine, topo, config);

    auto new_token = engine.allocate_memory(token_layout);
    auto seq_len = engine.allocate_memory(seq_len_layout);
    net.set_input_data("new_token", new_token);
    net.set_input_data("present_len", seq_len);

    auto run = [&](int64_t capacity, int64_t present_len) -> memory::ptr {
        const layout past_layout{ov::PartialShape{1, heads, capacity, head_size}, data_types::f32, format::bfyx};
        auto past = engine.allocate_memory(past_layout);
        std::vector<float> past_values(past_layout.count());
        std::iota(past_values.begin(), past_values.end(), 0.0f);
        set_values(past, past_values);
        const float token_value = 1000.0f + static_cast<float>(present_len);
        set_values(new_token, std::vector<float>(token_layout.count(), token_value));
        set_values<int64_t>(seq_len, {present_len});
        net.set_input_data("past", past);

        auto outputs = net.execute();
        auto result = outputs.at("result").get_memory();
        auto kv_output = net.get_primitive("stateless_kv")->output_memory_ptr(0);
        EXPECT_NE(result, nullptr);
        EXPECT_NE(kv_output, nullptr);
        if (!result || !kv_output)
            return nullptr;
        EXPECT_FALSE(engine.is_the_same_buffer(*kv_output, *result));

        auto expected = past_values;
        for (int64_t head = 0; head < heads; ++head)
            std::fill_n(expected.begin() + (head * capacity + present_len - 1) * head_size, head_size, token_value);
        mem_lock<ov::float16, mem_lock_type::read> actual(result, get_test_stream());
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t i = 0; i < present_len * head_size; ++i) {
                const auto offset = static_cast<size_t>(head * capacity * head_size + i);
                EXPECT_FLOAT_EQ(static_cast<float>(actual[offset]), expected[offset])
                    << "capacity=" << capacity << " present_len=" << present_len << " index=" << offset;
            }
        }
        return kv_output;
    };

    auto first = run(16, 5);
    ASSERT_NE(first, nullptr);
    for (int64_t present_len : {6, 7}) {
        auto kv_output = run(16, present_len);
        ASSERT_NE(kv_output, nullptr);
        EXPECT_TRUE(engine.is_the_same_buffer(*kv_output, *first)) << "present_len=" << present_len;
    }

    // A larger cache needs a larger buffer; shrinking back must keep using it.
    auto grown = run(32, 20);
    ASSERT_NE(grown, nullptr);
    auto shrunk = run(16, 8);
    ASSERT_NE(shrunk, nullptr);
    EXPECT_TRUE(engine.is_the_same_buffer(*shrunk, *grown));
}

namespace {

bool gpu_supports_usm(ov::Core& core) {
    const auto caps = core.get_property("GPU", ov::device::capabilities);
    return std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::USM_MEMORY) != caps.end();
}

// Output 0 is present; with_past_reader adds output 1 = past * 2, a second reader of past.
std::shared_ptr<ov::Model> make_stateless_kv_model(size_t new_token_len, bool with_past_reader = false) {
    auto past = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, -1, 4});
    auto token = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, new_token_len, 4});
    auto present_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    auto kv = std::make_shared<ov::intel_gpu::op::StatelessKV>(past, token, present_len, 2, true);
    ov::OutputVector outputs{kv->output(0)};
    if (with_past_reader) {
        auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
        outputs.push_back(std::make_shared<ov::op::v1::Multiply>(past, scale));
    }
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{past, token, present_len});
}

}  // namespace

class stateless_kv_caller_host_output : public testing::TestWithParam<bool> {};

// The caller output shares past's USM-host allocation, whether past is a remote or plain host tensor.
TEST_P(stateless_kv_caller_host_output, repeated_inference) {
    ov::Core core;
    if (!gpu_supports_usm(core)) {
        GTEST_SKIP() << "USM is not supported";
    }

    auto context = core.get_default_context("GPU");
    auto compiled_model = core.compile_model(make_stateless_kv_model(1), context, ov::hint::inference_precision(ov::element::f32));
    auto request = compiled_model.create_infer_request();

    const ov::Shape cache_shape{1, 2, 16, 4};
    auto gpu_context = context.as<ov::intel_gpu::ocl::ClContext>();
    auto cache_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, cache_shape);
    auto* cache = static_cast<float*>(cache_allocation.get());
    ov::Tensor past_tensor(ov::element::f32, cache_shape, cache);
    ov::Tensor present_tensor(ov::element::f32, cache_shape, cache);
    ov::Tensor token_tensor(ov::element::f32, {1, 2, 1, 4});
    ov::Tensor length_tensor(ov::element::i64, {1});
    std::vector<float> expected(ov::shape_size(cache_shape), 1.0f);
    std::copy(expected.begin(), expected.end(), cache);
    if (GetParam()) {
        request.set_input_tensor(0, cache_allocation);
    } else {
        request.set_input_tensor(0, past_tensor);
    }
    request.set_input_tensor(1, token_tensor);
    request.set_input_tensor(2, length_tensor);
    request.set_output_tensor(0, present_tensor);

    for (int64_t length : {5, 6, 7}) {
        const float token_value = static_cast<float>(length);
        std::fill_n(token_tensor.data<float>(), token_tensor.get_size(), token_value);
        length_tensor.data<int64_t>()[0] = length;
        OV_ASSERT_NO_THROW(request.infer());

        auto actual = request.get_output_tensor(0);
        ASSERT_FALSE(actual.is<ov::intel_gpu::ocl::USMTensor>());
        ASSERT_EQ(actual.data(), cache);
        ASSERT_EQ(actual.get_shape(), cache_shape);
        for (size_t head = 0; head < 2; ++head)
            std::fill_n(expected.begin() + head * 16 * 4 + static_cast<size_t>(length - 1) * 4, 4, token_value);
        // Rows at or beyond present_len are undefined when the plugin falls back to a copy.
        for (size_t head = 0; head < 2; ++head) {
            for (size_t i = 0; i < static_cast<size_t>(length) * 4; ++i) {
                const size_t offset = head * 16 * 4 + i;
                ASSERT_FLOAT_EQ(actual.data<const float>()[offset], expected[offset]) << "length=" << length << " index=" << offset;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         stateless_kv_caller_host_output,
                         testing::Bool(),
                         [](const testing::TestParamInfo<bool>& info) { return info.param ? "RemotePast" : "CallerHostPast"; });

// Another reader of past must still see the original past, whether or not present is bound over it.
TEST(stateless_kv_runtime, infer_request_aliased_past_with_another_reader) {
    ov::Core core;
    if (!gpu_supports_usm(core)) {
        GTEST_SKIP() << "USM is not supported";
    }

    auto context = core.get_default_context("GPU");
    auto compiled_model = core.compile_model(make_stateless_kv_model(1, true), context, ov::hint::inference_precision(ov::element::f32));
    auto request = compiled_model.create_infer_request();

    const ov::Shape cache_shape{1, 2, 16, 4};
    auto gpu_context = context.as<ov::intel_gpu::ocl::ClContext>();
    auto cache_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, cache_shape);
    auto* cache = static_cast<float*>(cache_allocation.get());
    ov::Tensor past_tensor(ov::element::f32, cache_shape, cache);
    ov::Tensor present_tensor(ov::element::f32, cache_shape, cache);
    ov::Tensor token_tensor(ov::element::f32, {1, 2, 1, 4});
    ov::Tensor length_tensor(ov::element::i64, {1});
    request.set_input_tensor(0, past_tensor);
    request.set_input_tensor(1, token_tensor);
    request.set_input_tensor(2, length_tensor);
    request.set_output_tensor(0, present_tensor);

    std::vector<float> original(ov::shape_size(cache_shape));
    for (size_t i = 0; i < original.size(); ++i)
        original[i] = static_cast<float>(i % 13 + 1);
    constexpr int64_t length = 5;
    constexpr float token_value = 100.0f;
    std::fill_n(token_tensor.data<float>(), token_tensor.get_size(), token_value);
    length_tensor.data<int64_t>()[0] = length;

    auto expected_present = original;
    for (size_t head = 0; head < 2; ++head)
        std::fill_n(expected_present.begin() + head * 16 * 4 + static_cast<size_t>(length - 1) * 4, 4, token_value);

    // An unsafe in-place update would race with the past reader, so repeat to catch it reliably.
    for (int iter = 0; iter < 8; ++iter) {
        std::copy(original.begin(), original.end(), cache);
        OV_ASSERT_NO_THROW(request.infer());

        auto present = request.get_output_tensor(0);
        auto scaled_past = request.get_output_tensor(1);
        ASSERT_EQ(present.data(), cache);
        ASSERT_EQ(scaled_past.get_shape(), cache_shape);
        // Rows at or beyond present_len are undefined when the plugin falls back to a copy.
        for (size_t head = 0; head < 2; ++head) {
            for (size_t i = 0; i < static_cast<size_t>(length) * 4; ++i) {
                const size_t offset = head * 16 * 4 + i;
                ASSERT_FLOAT_EQ(present.data<const float>()[offset], expected_present[offset]) << "iter=" << iter << " index=" << offset;
            }
        }
        for (size_t i = 0; i < original.size(); ++i) {
            ASSERT_FLOAT_EQ(scaled_past.data<const float>()[i], original[i] * 2.0f) << "iter=" << iter << " index=" << i;
        }
    }
}