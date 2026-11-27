// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph/include/primitive_inst.h"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "random_generator.hpp"
#include "test_utils.h"

#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/assign.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/read_value.hpp>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

template<typename T>
struct VariableParams {
    cldnn::layout layout;
    std::vector<T> values;
};

template<typename T>
struct variable_test : public ::testing::TestWithParam<VariableParams<T>> {
    void test(bool is_caching_test) {
        const VariableParams<T> param = testing::TestWithParam<VariableParams<T>>::GetParam();

        auto& engine = get_test_engine();

        const auto variable_layout = param.layout;
        const auto input_data = engine.allocate_memory(variable_layout);
        set_values(input_data, param.values);

        topology topology;
        topology.add(input_layout("input", input_data->get_layout()));
        topology.add(read_value{"read_value", { input_info("input") }, "v0", { variable_layout }});
        topology.add(eltwise{"sum", { input_info("input"), input_info("read_value") }, eltwise_mode::sum, {}, variable_layout.data_type});
        topology.add(assign{"assign", { input_info("sum") }, "v0", variable_layout});

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
        auto variable = std::make_shared<VariableState>(VariableStateInfo{"v0", variable_layout}, context, network->get_shape_predictor());
        network->set_variable("v0", variable);
        network->set_input_data("input", input_data);

        constexpr size_t number_of_inferences = 5;
        for (size_t inference = 1; inference <= number_of_inferences; ++inference) {
            const auto outputs = network->execute();
            const auto output = outputs.at("assign").get_memory();
            const cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
            const auto output_count = output_ptr.size();
            ASSERT_EQ(output_count, param.values.size()) << "inference " << inference;

            for (size_t i = 0; i < output_count; ++i) {
                if (ov::element::Type(output->get_layout().data_type).is_real()) {
                    ASSERT_FLOAT_EQ(output_ptr[i], (inference + 1) * param.values[i]) << "inference " << inference;
                } else {
                    ASSERT_EQ(output_ptr[i], (inference + 1) * param.values[i]) << "inference " << inference;
                }
            }
        }
    }
};

using variable_test_i32 = variable_test<int32_t>;
using variable_test_i64 = variable_test<int64_t>;
using variable_test_f32 = variable_test<float>;

TEST_P(variable_test_i32, variable_i32) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(variable_test_i64, variable_i64) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(variable_test_f32, variable_f32) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

INSTANTIATE_TEST_SUITE_P(
        basic,
        variable_test_i32,
        ::testing::Values(
                VariableParams<int32_t>{ {data_types::i32, format::bfyx, tensor{1}}, {333666} },
                VariableParams<int32_t>{ {data_types::i32, format::bfyx, tensor{1, 1, 1, 3}}, {444, 555, 666} },
                VariableParams<int32_t>{ {data_types::i32, format::bfzyx, tensor{1, 2, 3, 2}},
                                {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1} }
        )
);

INSTANTIATE_TEST_SUITE_P(
        basic,
        variable_test_i64,
        ::testing::Values(
                VariableParams<int64_t>{ {data_types::i64, format::bfyx, tensor{1}}, {333666L} },
                VariableParams<int64_t>{ {data_types::i64, format::bfyx, tensor{1, 1, 1, 3}}, {444L, 555L, 666L} },
                VariableParams<int64_t>{ {data_types::i64, format::bfzyx, tensor{1, 2, 3, 2}},
                                         {1L, 2L, 3L, 4L, 5L, 6L, 6L, 5L, 4L, 3L, 2L, 1L} }
        )
);

INSTANTIATE_TEST_SUITE_P(
        basic,
        variable_test_f32,
        ::testing::Values(
                VariableParams<float>{ {data_types::f32, format::bfyx, tensor{1}}, {333666.f} },
                VariableParams<float>{ {data_types::f32, format::bfyx, tensor{1, 1, 1, 3}}, {44.4f, 55.5f, 66.6f} },
                VariableParams<float>{ {data_types::f32, format::bfzyx, tensor{1, 2, 3, 2}},
                                         {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f} }
        )
);

template <typename T>
void test_exception_on_wrong_layout(bool is_caching_test) {
    auto& engine = get_test_engine();

    const layout variable_layout{data_types::i32, format::bfyx, tensor{1}};
    const auto input_data = engine.allocate_memory(variable_layout);
    set_values(input_data, {333666});

    auto wrong_layout = variable_layout;
    wrong_layout.data_type = data_types::f32;
    const auto wrong_input_data = engine.allocate_memory(wrong_layout);
    set_values(input_data, {333.666f});

    topology topology;
    topology.add(input_layout("input", input_data->get_layout()));
    topology.add(read_value{"read_value", { input_info("input") }, "v0", { variable_layout }});
    topology.add(input_layout("wrong_input", wrong_input_data->get_layout()));
    topology.add(assign{"assign", { input_info("wrong_input") }, "v0", wrong_layout});

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable = std::make_shared<VariableState>(VariableStateInfo{"v0", variable_layout}, context, network->get_shape_predictor());
    network->set_variable("v0", variable);
    network->set_input_data("input", input_data);
    network->set_input_data("wrong_input", wrong_input_data);

    bool layout_mismatch_exception = false;
    try {
        network->execute();
    } catch(std::exception& exc) {
        const std::string error = exc.what();
        layout_mismatch_exception = error.find("Layout mismatch") != std::string::npos;
    }
    ASSERT_TRUE(layout_mismatch_exception);
}

TEST(variable_test_common, exception_on_wrong_layout) {
    test_exception_on_wrong_layout<float>(false);
}

template <typename T>
void test_variable_copy_from_fake_aligned_fc(bool is_caching_test) {
    tests::random_generator rg;
    auto& engine = get_test_engine();

    const int32_t input_b = 3, input_f = 590, input_x = 768, weight_b = 768;
    auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(), input_f, input_x }, data_types::f16, format::bfyx };
    auto input_data = engine.allocate_memory(layout{ ov::PartialShape{ input_b, input_f, input_x }, data_types::f16, format::bfyx });
    auto input_data_rnd = rg.generate_random_1d<ov::float16>(input_b * input_f * input_x, 0, 1);
    set_values(input_data, input_data_rnd);

    auto weights_data_layout = layout{ov::PartialShape{weight_b, input_x}, data_types::f16, format::os_iyx_osv32};
    auto weights_data = engine.allocate_memory(weights_data_layout);
    auto weights_data_rnd = rg.generate_random_1d<ov::float16>(weights_data_layout.count(), 0, 1);
    set_values(weights_data, weights_data_rnd);

    auto bias_data_layout = layout{ov::PartialShape{1, 1, weight_b}, data_types::f16, format::bfyx};
    auto bias_data = engine.allocate_memory(bias_data_layout);
    auto bias_data_rnd = rg.generate_random_1d<ov::float16>(bias_data_layout.count(), 0, 1);
    set_values(bias_data, bias_data_rnd);

    const layout variable_layout{{input_b, input_f, input_x}, data_types::f16, format::bfyx};

    topology topology;
    topology.add(input_layout("input", input_dyn_layout));
    topology.add(data("weights", weights_data));
    topology.add(data("bias", bias_data));
    topology.add(fully_connected("fc", input_info("input"), "weights", "bias", input_dyn_layout.get_partial_shape().size()));
    topology.add(read_value{"read_value", { input_info("fc") }, "v0", { variable_layout }});


    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable = std::make_shared<VariableState>(VariableStateInfo{"v0", variable_layout}, context, network->get_shape_predictor());
    network->set_variable("v0", variable);
    network->set_input_data("input", input_data);

    auto outputs = network->execute();

    // fc
    const auto fc_output_memory = network->get_output_memory("fc");
    const auto fc_output_shape = fc_output_memory->get_layout().get_partial_shape().to_shape();
    auto fc_output_batch_size = std::accumulate(fc_output_shape.begin(),
                                          fc_output_shape.end() - 1,
                                          size_t{1},
                                          std::multiplies<size_t>());

    auto weights_layout_dt = weights_data_layout.data_type;
    auto is_4bit = weights_layout_dt == data_types::i4 || weights_layout_dt == data_types::u4;
    auto is_8bit = weights_layout_dt == data_types::i8 || weights_layout_dt == data_types::u8;
    auto is_extra_alignment_needed = input_b >= 256;
    // Match runtime logic in fully_connected_inst::get_fake_aligned_params():
    // iGPU uses base 16 (or 64 for quantized + large batch), dGPU uses base 8.
    size_t fake_align_base = 8;
    if (engine.get_device_info().dev_type == cldnn::device_type::integrated_gpu) {
        fake_align_base = (is_4bit || is_8bit) && is_extra_alignment_needed ? 64 : 16;
    }

    const size_t batch_size = static_cast<size_t>(input_b * input_f);
    const auto fc_output_batch_size_aligned = align_to(batch_size, fake_align_base);
    // On Xe2+ and IMMAD iGPUs the runtime may roll back fake alignment when
    // the predecessor buffer is too small (see get_fake_aligned_params_if_possible).
    // Accept either the aligned or the original unaligned batch size.
    ASSERT_TRUE(fc_output_batch_size == fc_output_batch_size_aligned ||
                fc_output_batch_size == batch_size)
        << "fc_output_batch_size=" << fc_output_batch_size
        << " expected aligned=" << fc_output_batch_size_aligned
        << " or unaligned=" << batch_size;

    // read_value
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "read_value");

    auto output_memory = outputs.begin()->second.get_memory();
    auto output_layout = output_memory->get_layout();
    ASSERT_EQ(output_layout, variable_layout);
}

TEST(variable_test_common, variable_copy_from_fc) {
    test_variable_copy_from_fake_aligned_fc<float>(false);
}

template <typename T>
void test_different_output_data_type(bool is_caching_test) {
    auto& engine = get_test_engine();

    const layout in_layout{{ 1 }, data_types::f32, format::bfyx};
    const auto input_data = engine.allocate_memory(in_layout);
    std::vector<float> inputs = { 70.0f };
    set_values(input_data, inputs);

    const layout variable_layout{{ 1 }, data_types::f16, format::bfyx};

    topology topology;
    topology.add(input_layout("input", input_data->get_layout()));
    topology.add(assign("assign", { input_info("input") }, "v0", variable_layout));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable = std::make_shared<VariableState>(VariableStateInfo{"v0", variable_layout}, context, network->get_shape_predictor());
    network->set_variable("v0", variable);
    network->set_input_data("input", input_data);

    const auto outputs = network->execute();
    const auto output = outputs.at("assign").get_memory();
    const cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < output_ptr.size(); ++i) {
         ASSERT_EQ(half_to_float(output_ptr[i]), inputs[i]);
    }
}

TEST(variable_test_common, different_output_data_type) {
    test_different_output_data_type<int16_t>(false);
}

template <typename T>
void test_variables_are_preserved_across_inferences(bool is_caching_test) {
    auto& engine = get_test_engine();

    const layout variable_layout{data_types::i32, format::bfyx, tensor{1}};

    const auto input_1 = engine.allocate_memory(variable_layout);
    constexpr auto value_1 = 222;
    set_values(input_1, {value_1});

    const auto input_2 = engine.allocate_memory(variable_layout);
    constexpr auto value_2 = 555;
    set_values(input_2, {value_2});

    const auto dummy1 = engine.allocate_memory(variable_layout);
    set_values(dummy1, {11});
    const auto dummy2 = engine.allocate_memory(variable_layout);
    set_values(dummy2, {22});

    topology topology;
    topology.add(input_layout("input_1", input_1->get_layout()));
    topology.add(assign{"assign_1", { input_info("input_1") }, "v1", variable_layout});

    topology.add(input_layout("input_2", input_2->get_layout()));
    topology.add(assign{"assign_2", { input_info("input_2") }, "v2", variable_layout});

    topology.add(data("dummy1", dummy1));
    topology.add(read_value{"read_value_1", { input_info("dummy1") }, "v1", { variable_layout }});
    topology.add(read_value{"read_value_2", { input_info("dummy1") }, "v2", { variable_layout }});

    topology.add(eltwise{"sum", { input_info("read_value_1"), input_info("read_value_2") }, eltwise_mode::sum, {}, variable_layout.data_type});
    topology.add(assign{"assign_result", { input_info("sum") }, "v_result", variable_layout});

    topology.add(data("dummy2", dummy2));
    topology.add(read_value{"read_result", { input_info("dummy2") }, "v_result", { variable_layout }});

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable1 = std::make_shared<VariableState>(VariableStateInfo{"v1", variable_layout}, context, network->get_shape_predictor());
    auto variable2 = std::make_shared<VariableState>(VariableStateInfo{"v2", variable_layout}, context, network->get_shape_predictor());
    auto variable3 = std::make_shared<VariableState>(VariableStateInfo{"v_result", variable_layout}, context, network->get_shape_predictor());
    network->set_variable("v1", variable1);
    network->set_variable("v2", variable2);
    network->set_variable("v_result", variable3);
    network->set_input_data("input_1", input_1);
    network->set_input_data("input_2", input_2);

    // set variables with assign on 1st inference, read with read_values on 2nd one
    network->execute();
    const auto outputs = network->execute();
    const auto output = outputs.at("read_result").get_memory();
    const cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr[0], value_1 + value_2);
}

TEST(variable_test_common, variables_are_preserved_across_inferences) {
    test_variables_are_preserved_across_inferences<int>(false);
}

TEST(variable_test_common, variables_release) {
    auto& engine = get_test_engine();

    const layout variable_layout{{ 1 }, data_types::f32, format::bfyx};
    const auto input_data = engine.allocate_memory(variable_layout);
    std::vector<float> inputs = { 70.0f };
    set_values(input_data, inputs);

    topology topology;
    topology.add(input_layout("input", input_data->get_layout()));
    topology.add(read_value{"read_value", {input_info("input")}, "v0", {variable_layout}});

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto read_value_inst = network->get_primitive("read_value");
    VariableStateInfo var_info{"v0", variable_layout};
    var_info.m_release_variable_inst.emplace_back(std::dynamic_pointer_cast<memory_state::releasable_variable>(read_value_inst));
    auto variable = std::make_shared<VariableState>(var_info, context, network->get_shape_predictor());
    network->set_variable("v0", variable);
    network->set_input_data("input", input_data);

    const auto outputs = network->execute();

    const auto output_count = read_value_inst->outputs_memory_count();
    for (size_t i = 0; i < output_count; ++i) {
        ASSERT_TRUE(read_value_inst->output_memory_ptr(i));
    }

    variable->reset();
    for (size_t i = 0; i < output_count; ++i) {
        ASSERT_FALSE(read_value_inst->output_memory_ptr(i));
    }
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(variable_test_i32, variable_i32_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(variable_test_i64, variable_i64_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(variable_test_f32, variable_f32_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST(variable_test_common, exception_on_wrong_layout_cached) {
    test_exception_on_wrong_layout<float>(true);
}

TEST(variable_test_common, variable_copy_from_fc_cached) {
    test_variable_copy_from_fake_aligned_fc<float>(true);
}

TEST(variable_test_common, different_output_data_type_cached) {
    test_different_output_data_type<int16_t>(true);
}
#endif
TEST(variable_test_common, variables_are_preserved_across_inferences_cached) {
    test_variables_are_preserved_across_inferences<int>(true);
}

// A GPU primitive feeding directly into `assign` must not be forced into
// lockable (usm_host) memory. The input does not need to be host-accessible.
TEST(variable_test_common, assign_producer_output_is_not_lockable) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_usm) {
        GTEST_SKIP() << "USM not supported on this device";
    }
    if (!engine.supports_allocation(allocation_type::usm_device)) {
        GTEST_SKIP() << "usm_device not supported on this device";
    }

    const layout var_layout{ov::PartialShape{1, 4}, data_types::f32, format::bfyx};
    const auto input_data = engine.allocate_memory(var_layout);
    set_values<float>(input_data, {1.0f, 2.0f, 3.0f, 4.0f});

    topology t;
    t.add(input_layout("input", var_layout));
    t.add(read_value{"read_value", {input_info("input")}, "v0", {var_layout}});
    // "producer" is a GPU eltwise. Its only user is `assign`, so we can observe
    // whether the allocator forces its output into usm_host.
    t.add(eltwise{"producer", {input_info("input"), input_info("read_value")},
                  eltwise_mode::sum, {}, var_layout.data_type});
    t.add(assign{"assign", {input_info("producer")}, "v0", var_layout});

    cldnn::network::ptr net = get_network(engine, t, get_test_default_config(engine),
                                          get_test_stream_ptr(), false);

    auto context = std::make_shared<RemoteContextImpl>(
        "GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable = std::make_shared<VariableState>(
        VariableStateInfo{"v0", var_layout}, context, net->get_shape_predictor());
    net->set_variable("v0", variable);
    net->set_input_data("input", input_data);
    net->execute();

    // Impl-level contract: assign is handled by CPU impl
    // but does not require a host-accessible input.
    auto assign_inst = net->get_primitive("assign");
    ASSERT_NE(assign_inst->get_impl(), nullptr);
    EXPECT_TRUE(assign_inst->get_impl()->is_cpu());
    EXPECT_FALSE(assign_inst->get_impl()->requires_lockable_input());

    // Check proper memory allocation
    auto producer_inst = net->get_primitive("producer");
    ASSERT_NE(producer_inst->output_memory_ptr(), nullptr);
    EXPECT_NE(producer_inst->output_memory_ptr()->get_allocation_type(),
              allocation_type::usm_host)
        << "Producer output was forced into usm_host by `assign` user. "
           "assign_impl::requires_lockable_input() should be false so the producer "
           "can stay in device memory.";
}

// ---------------------------------------------------------------------------
// Negative control: when the downstream user is a *real* CPU impl that reads
// tensor data on the host, the producer output must still be allocated in
// lockable memory. This guards against over-widening the requires_lockable_input()
// opt-out in the future.
// ---------------------------------------------------------------------------
TEST(variable_test_common, real_cpu_user_still_forces_lockable) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_usm) {
        GTEST_SKIP() << "USM not supported on this device";
    }
    // On integrated devices the lockable and non-lockable USM types may
    // coincide, making this control assertion meaningless.
    if (engine.get_device_info().dev_type != device_type::discrete_gpu) {
        GTEST_SKIP() << "Only meaningful on dGPU where lockable != device USM";
    }

    const layout in_layout{ov::PartialShape{1, 4}, data_types::f32, format::bfyx};
    const auto input_data = engine.allocate_memory(in_layout);
    set_values<float>(input_data, {1.0f, 2.0f, 3.0f, 4.0f});

    topology t;
    t.add(input_layout("input", in_layout));
    t.add(eltwise{"producer", {input_info("input"), input_info("input")},
                  eltwise_mode::sum, {}, in_layout.data_type});
    // Force the activation user onto the CPU impl. CPU activation reads and
    // writes tensor data on the host, so it truly requires lockable input.
    t.add(activation{"consumer", input_info("producer"), activation_func::relu});

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc cpu_activation_impl = {format::bfyx, "", impl_types::cpu};
    config.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{{"consumer", cpu_activation_impl}}));

    cldnn::network::ptr net = get_network(engine, t, config, get_test_stream_ptr(), false);
    net->set_input_data("input", input_data);
    net->execute();

    auto consumer_inst = net->get_primitive("consumer");
    ASSERT_NE(consumer_inst->get_impl(), nullptr);
    // Sanity: the forced CPU activation impl reports itself correctly.
    EXPECT_TRUE(consumer_inst->get_impl()->is_cpu());
    EXPECT_TRUE(consumer_inst->get_impl()->requires_lockable_input());

    auto producer_inst = net->get_primitive("producer");
    ASSERT_NE(producer_inst->output_memory_ptr(), nullptr);
    EXPECT_EQ(producer_inst->output_memory_ptr()->get_allocation_type(),
              allocation_type::usm_host)
        << "Producer output should be lockable (usm_host on dGPU) when the "
           "downstream user is a real CPU impl that reads tensor data on the host.";
}

// ---------------------------------------------------------------------------
// Several ReadValue -> Add -> Assign states, executed repeatedly with changing
// inputs, on an in-order and on an out-of-order queue. On an in-order queue the
// ReadValue/Assign CPU impls skip the host-side wait and Assign's copy is
// non-blocking; correctness then relies on the queue ordering alone. Every element
// of every state is checked after every inference, so a stale, partial or
// reordered state copy fails the test.
// ---------------------------------------------------------------------------
struct variable_multi_state_test : public ::testing::TestWithParam<QueueTypes> {};

TEST_P(variable_multi_state_test, accumulates_across_inferences) {
    auto& engine = get_test_engine();
    const QueueTypes queue_type = GetParam();
    if (queue_type == QueueTypes::out_of_order && engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "out-of-order queue is not supported on this device";
    }

    constexpr size_t num_states = 6;
    constexpr size_t num_inferences = 10;
    // Large enough that each state copy is a real transfer, not a few bytes.
    const layout var_layout{ov::PartialShape{1, 8, 64, 128}, data_types::f32, format::bfyx};
    const size_t count = var_layout.count();

    topology topology;
    std::vector<memory::ptr> inputs;
    for (size_t s = 0; s < num_states; ++s) {
        const auto id = std::to_string(s);
        inputs.push_back(engine.allocate_memory(var_layout));
        topology.add(input_layout("input" + id, var_layout));
        topology.add(read_value{"read_value" + id, {input_info("input" + id)}, "v" + id, {var_layout}});
        topology.add(eltwise{"sum" + id, {input_info("input" + id), input_info("read_value" + id)}, eltwise_mode::sum, {}, var_layout.data_type});
        topology.add(assign{"assign" + id, {input_info("sum" + id)}, "v" + id, var_layout});
    }

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::queue_type(queue_type));
    cldnn::network::ptr network = std::make_shared<cldnn::network>(engine, topology, config);
    ASSERT_EQ(network->get_stream().get_queue_type(), queue_type);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    std::vector<std::shared_ptr<VariableState>> variables;
    for (size_t s = 0; s < num_states; ++s) {
        const auto id = std::to_string(s);
        variables.push_back(std::make_shared<VariableState>(VariableStateInfo{"v" + id, var_layout}, context, network->get_shape_predictor()));
        network->set_variable("v" + id, variables.back());
        network->set_input_data("input" + id, inputs[s]);
    }

    // Small integers stay exact in f32, so the expected state can be compared exactly.
    std::vector<std::vector<float>> expected(num_states, std::vector<float>(count, 0.f));
    std::vector<float> values(count);
    for (size_t inference = 1; inference <= num_inferences; ++inference) {
        for (size_t s = 0; s < num_states; ++s) {
            for (size_t i = 0; i < count; ++i)
                values[i] = static_cast<float>((i * 7 + s * 13 + inference * 31) % 97);
            set_values(inputs[s], values);
            for (size_t i = 0; i < count; ++i)
                // 1st inference: read_value returns its initializer (the input itself).
                expected[s][i] = (inference == 1 ? 2.f * values[i] : expected[s][i] + values[i]);
        }

        network->execute();

        for (size_t s = 0; s < num_states; ++s) {
            ASSERT_TRUE(variables[s]->is_set()) << "state " << s << ", inference " << inference;
            const cldnn::mem_lock<float, mem_lock_type::read> state(variables[s]->get_memory(), network->get_stream());
            ASSERT_EQ(state.size(), count);
            for (size_t i = 0; i < count; ++i) {
                ASSERT_EQ(state[i], expected[s][i]) << "state " << s << ", element " << i << ", inference " << inference;
            }
        }
    }

    // Both state primitives must have run as CPU impls: that is the code path under test.
    ASSERT_NE(network->get_primitive("read_value0")->get_impl(), nullptr);
    ASSERT_NE(network->get_primitive("assign0")->get_impl(), nullptr);
    EXPECT_TRUE(network->get_primitive("read_value0")->get_impl()->is_cpu());
    EXPECT_TRUE(network->get_primitive("assign0")->get_impl()->is_cpu());
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         variable_multi_state_test,
                         ::testing::Values(QueueTypes::in_order, QueueTypes::out_of_order),
                         [](const ::testing::TestParamInfo<QueueTypes>& info) {
                             return info.param == QueueTypes::in_order ? std::string("in_order") : std::string("out_of_order");
                         });
