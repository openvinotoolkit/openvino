// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include "ocl_v2/moe/residency/lookahead_router.hpp"
#include "ocl_v2/moe/residency/moe_lookahead_registry.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <future>
#include "intel_gpu/runtime/les_context.hpp"
#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_event.hpp"

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
using ov::intel_gpu::ocl::moe::ResidentExpertWeightProvider;

namespace {
namespace otd = ov::intel_gpu::ocl::moe_otd;

class moe_les_upload : public ::testing::Test {
protected:
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ov_les_upload_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");

    void TearDown() override { std::filesystem::remove(path); }

    void write(const std::vector<uint8_t>& bytes) {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ASSERT_TRUE(output.good());
    }

    stream::ptr lane(bool profiling = false) {
        auto& engine = get_test_engine();
        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order), ov::enable_profiling(profiling));
        return engine.create_stream(config);
    }

    stream::ptr copy_lane(bool profiling = false) {
        auto& engine = get_test_engine();
        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order), ov::enable_profiling(profiling));
        return otd::create_copy_stream(engine, config);
    }

    memory::ptr buffer(size_t count, data_types type = data_types::u8, size_t experts = 1) {
        return get_test_engine().allocate_memory(layout{type, format::bfyx,
            {static_cast<int>(experts), 1, 1, static_cast<int>(count)}});
    }
};

TEST_F(moe_les_upload, all_nine_components_are_ready_and_repacked) {
    MOECompressed::Config config{};
    config.num_expert = 2;
    config.hidden_size = 4;
    config.inter_size = 4;
    config.group_size = 2;
    for (bool profiling : {false, true}) {
        auto transfer = lane(profiling);
        moe_weights weights;
        std::array<memory::ptr*, 9> tensors{&weights.gate_w, &weights.up_w, &weights.down_w,
            &weights.gate_s, &weights.up_s, &weights.down_s, &weights.gate_z, &weights.up_z, &weights.down_z};
        std::vector<size_t> offsets;
        std::vector<uint8_t> file;
        for (size_t component = 0; component < tensors.size(); ++component) {
            *tensors[component] = component < 3 ? buffer(16, data_types::u8, 2) :
                buffer(8, component < 6 ? data_types::f16 : data_types::u4, 2);
            offsets.push_back(file.size());
            auto bytes = (*tensors[component])->get_layout().bytes_count();
            for (size_t index = 0; index < bytes; ++index) {
                file.push_back(static_cast<uint8_t>((index + component * 17) % 251));
            }
        }
        write(file);
        otd::ParallelWeightReader reader(path);
        auto events = otd::fill_weights_memory_async(*transfer, config, offsets, reader, weights, {1}, {1});
        ASSERT_EQ(events.size(), 9U);
        for (const auto& event : events) {
            EXPECT_TRUE(event->is_set());
        }
        for (size_t component = 0; component < tensors.size(); ++component) {
            auto tensor = *tensors[component];
            auto bytes = tensor->get_layout().bytes_count() / 2;
            auto begin = file.begin() + offsets[component] + bytes;
            std::vector<uint8_t> expected(begin, begin + bytes);
            otd::maybe_transpose_scale_zp(config, component, tensor->get_layout(), expected, bytes);
            std::vector<uint8_t> actual(bytes);
            tensor->copy_to(*transfer, actual.data(), bytes, 0, bytes, true);
            EXPECT_EQ(actual, expected);
        }
    }
}

TEST_F(moe_les_upload, staging_reuse_preserves_large_tensor_source) {
    std::vector<uint8_t> source(16 * 1024 * 1024 + 37);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint8_t>(index % 251);
    }
    write(source);
    MOECompressed::Config config{};
    config.num_expert = 1;
    auto transfer = copy_lane();
    otd::ParallelWeightReader reader(path);
    for (auto allocation : {allocation_type::cl_mem, allocation_type::usm_device}) {
        moe_weights weights;
        weights.gate_w = get_test_engine().allocate_memory(
            layout{data_types::u8, format::bfyx, {1, 1, 1, static_cast<int32_t>(source.size())}}, allocation, false);
        auto events = otd::fill_weights_memory_async(*transfer, config, std::vector<size_t>(9), reader, weights, {0}, {0});
        EXPECT_EQ(events.size(), 3U);
        std::vector<uint8_t> actual(source.size());
        weights.gate_w->copy_to(*transfer, actual.data(), true);
        EXPECT_EQ(actual, source);
    }
}

TEST_F(moe_les_upload, native_lookahead_router_keeps_actual_outputs_independent) {
    auto& engine = get_test_engine();
    const layout hidden_layout{ov::PartialShape{1, 16}, data_types::f16, format::bfyx};
    auto actual_hidden = engine.allocate_memory(hidden_layout);
    auto predicted_hidden = engine.allocate_memory(hidden_layout);
    auto weights = engine.allocate_memory(layout{ov::PartialShape{16, 16}, data_types::f16, format::bfyx});
    std::vector<ov::float16> identity(256, 0);
    for (size_t expert = 0; expert < 16; ++expert) {
        identity[expert * 16 + expert] = 1;
    }
    set_values(weights, identity);
    std::vector<ov::float16> hidden(16, 0);
    hidden[2] = 4;
    set_values(actual_hidden, hidden);
    hidden[2] = 0;
    hidden[5] = 4;
    set_values(predicted_hidden, hidden);
    MoERouterFused::Config config{};
    config.num_expert = 16;
    config.top_k = 1;
    config.routing_type = MoERouterFused::RoutingType::SOFTMAX;
    topology graph;
    graph.add(input_layout("hidden", hidden_layout));
    graph.add(data("weights", weights));
    graph.add(fully_connected("projection", input_info("hidden"), "weights"));
    graph.add(moe_router_fused("router", {input_info("projection")}, config));
    network actual(engine, graph, get_test_default_config(engine));
    actual.set_input_data("hidden", actual_hidden);
    ov::intel_gpu::ocl::moe::LookaheadRouter lookahead(actual, "hidden", "router");
    write(std::vector<uint8_t>(256, 11));
    MOECompressed::Config expert_config{};
    expert_config.num_expert = 16;
    auto provider = std::make_shared<OffloadExpertWeightProvider>(4, expert_config, std::vector<size_t>(9), path);
    moe_weights expert_weights;
    expert_weights.gate_w = engine.reinterpret_buffer(*buffer(16, data_types::u8, 4),
        layout{data_types::u8, format::bfyx, {16, 1, 1, 16}});
    provider->bind(expert_weights);
    ov::intel_gpu::ocl::moe::ExpertPrefetcher prefetcher(engine, get_test_default_config(engine));
    prefetcher.bind(provider);
    uint32_t actual_id = 0;
    auto& compute = actual.get_stream();
    auto readback_memory = engine.allocate_memory(
        layout{data_types::i32, format::bfyx, {1, 1, 1, 1}}, allocation_type::usm_host, false);
    cl::UserEvent producer_gate(dynamic_cast<cldnn::ocl::ocl_engine&>(engine).get_cl_context());
    cl::UserEvent prediction_gate(dynamic_cast<cldnn::ocl::ocl_engine&>(engine).get_cl_context());
    dynamic_cast<cldnn::ocl::ocl_stream&>(compute).enqueue_external_barrier(
        {std::make_shared<cldnn::ocl::ocl_event>(producer_gate)});
    std::promise<event::ptr> submitted;
    auto readback_submitted = submitted.get_future();
    auto current_finished = std::async(std::launch::async, [&] {
        actual.execute();
        auto actual_ready = compute.enqueue_marker({}, true);
        compute.enqueue_barrier({actual_ready});
        auto readback = readback_memory->copy_from(
            compute, *actual.get_primitive("router")->output_memory_ptr(1), 0, 0, sizeof(actual_id), false);
        compute.flush();
        submitted.set_value(readback);
        dynamic_cast<cldnn::ocl::ocl_stream&>(compute).enqueue_external_barrier(
            {std::make_shared<cldnn::ocl::ocl_event>(prediction_gate)});
        auto prediction = lookahead.enqueue(predicted_hidden, actual_ready);
        compute.enqueue_barrier({compute.enqueue_marker({}, true)});
        compute.flush();
        readback->wait();
        {
            mem_lock<uint32_t, mem_lock_type::read> indices(readback_memory, compute);
            actual_id = indices[0];
        }
        auto demand = provider->cache().partition_3way({actual_id}, [&] { return prefetcher.make_ready_event(); });
        OPENVINO_ASSERT(demand, "Current demand must fit before prediction completes");
        prefetcher.demand_async(*provider, {actual_id})->wait();
        EXPECT_FALSE(prediction.ready->is_set());
        return prediction;
    });
    const auto before_producer = readback_submitted.wait_for(std::chrono::seconds(2));
    if (before_producer == std::future_status::ready) {
        EXPECT_FALSE(readback_submitted.get()->is_set());
    }
    producer_gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(before_producer, std::future_status::ready);
    const auto before_prediction = current_finished.wait_for(std::chrono::seconds(2));
    prediction_gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(before_prediction, std::future_status::ready);
    auto prediction = current_finished.get();
    prediction.ready->wait();
    uint32_t predicted_id = 0;
    prediction.indices->copy_to(actual.get_stream(), &predicted_id, true);
    EXPECT_EQ(predicted_id, 5U);
    EXPECT_EQ(actual_id, 2U);
    std::vector<uint8_t> uploaded(16);
    expert_weights.gate_w->copy_to(compute, uploaded.data(), 0, 0, uploaded.size(), true);
    EXPECT_EQ(uploaded, std::vector<uint8_t>(16, 11));
    actual.get_primitive("router")->output_memory_ptr(1)->copy_to(compute, &actual_id, true);
    EXPECT_EQ(actual_id, 2U);
}

TEST_F(moe_les_upload, lookahead_registry_shares_without_owning_context) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::moe_lookahead_prefetch(true), ov::intel_gpu::offload_ratio(75));
    topology graph(input_layout("hidden", layout{data_types::f16, format::bfyx, {1, 16, 1, 1}}));
    network owner(engine, graph, config);
    auto context = ov::intel_gpu::ocl::moe::LookaheadRegistry::acquire(owner);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context, ov::intel_gpu::ocl::moe::LookaheadRegistry::acquire(owner));
    EXPECT_EQ(context->get_next_layer(owner.get_primitive("hidden").get()), nullptr);
    MOECompressed::Config expert_config{};
    expert_config.num_expert = 2;
    auto provider = std::make_shared<OffloadExpertWeightProvider>(1, expert_config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider->bind(weights);
    auto canonical = context->bind("hidden", provider);
    auto cloned = provider->clone();
    cloned->bind(weights);
    EXPECT_EQ(context->bind("hidden", cloned), canonical);
    EXPECT_EQ(canonical, provider);
    std::weak_ptr<ov::intel_gpu::ocl::moe::IExpertWeightProvider> provider_lifetime = canonical;
    provider.reset();
    canonical.reset();
    cloned.reset();
    EXPECT_FALSE(provider_lifetime.expired());
    std::weak_ptr<ov::intel_gpu::ocl::moe::LookaheadContext> observer = context;
    context.reset();
    EXPECT_TRUE(observer.expired());
    EXPECT_TRUE(provider_lifetime.expired());
}

TEST_F(moe_les_upload, validates_entire_plan_before_first_upload) {
    write(std::vector<uint8_t>(32, 7));
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(16);
    weights.up_w = buffer(16);
    auto transfer = lane();
    std::vector<uint8_t> sentinel(16, 77);
    weights.gate_w->copy_from(*transfer, sentinel.data(), true);
    std::vector<size_t> offsets(9);
    offsets[1] = 100;
    otd::ParallelWeightReader reader(path);
    EXPECT_ANY_THROW(otd::fill_weights_memory_async(*transfer, config, offsets, reader, weights, {0}, {0}));
    std::vector<uint8_t> actual(16);
    weights.gate_w->copy_to(*transfer, actual.data(), true);
    EXPECT_EQ(actual, sentinel);
}

TEST_F(moe_les_upload, truncated_file_is_an_error_not_gpu_ready) {
    write(std::vector<uint8_t>(32, 7));
    otd::ParallelWeightReader reader(path);
    ASSERT_EQ(reader.file_size(), 32U);
    std::filesystem::resize_file(path, 8);
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(32);
    auto transfer = lane();
    EXPECT_ANY_THROW(otd::fill_weights_memory_async(*transfer, config, std::vector<size_t>(9), reader, weights, {0}, {0}));
    EXPECT_NO_THROW(transfer->finish());
}

TEST_F(moe_les_upload, missing_file_reports_open_failure) {
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(16);
    auto transfer = lane();
    otd::ParallelWeightReader reader(path);
    EXPECT_ANY_THROW(otd::fill_weights_memory_async(*transfer, config, std::vector<size_t>(9), reader, weights, {0}, {0}));
}

TEST_F(moe_les_upload, later_short_read_drains_already_submitted_dma) {
    write(std::vector<uint8_t>(64, 7));
    otd::ParallelWeightReader reader(path);
    ASSERT_EQ(reader.file_size(), 64U);
    std::filesystem::resize_file(path, 32);
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(32);
    weights.up_w = buffer(32);
    auto transfer = lane();
    std::vector<size_t> offsets(9);
    offsets[1] = 32;
    EXPECT_ANY_THROW(otd::fill_weights_memory_async(*transfer, config, offsets, reader, weights, {0}, {0}));
    std::vector<uint8_t> actual(32);
    weights.gate_w->copy_to(*transfer, actual.data(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(32, 7));
}
TEST_F(moe_les_upload, split_demand_returns_ready_pairs_before_upload_finishes) {
    auto file = std::vector<uint8_t>(48, 7);
    std::fill(file.begin() + 16, file.begin() + 32, 11);
    write(file);
    MOECompressed::Config config{};
    config.num_expert = 3;
    auto provider = std::make_shared<OffloadExpertWeightProvider>(2, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 3);
    provider->bind(weights);
    auto compute = lane();
    ASSERT_TRUE(provider->try_acquire_simultaneous({0, 2}, *compute));
    provider->release();
    ov::intel_gpu::ocl::moe::ExpertPrefetcher prefetcher(get_test_engine(), get_test_default_config(get_test_engine()));
    prefetcher.bind(provider);
    auto make_event = [&] { return prefetcher.make_ready_event(); };
    auto initial = provider->cache().partition_3way({0, 2}, make_event);
    ASSERT_TRUE(initial);
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    provider->cache().unpin_all(std::make_shared<cldnn::ocl::ocl_event>(gate));
    auto demand = provider->cache().partition_3way({1, 0, 1, 0}, make_event);
    ASSERT_TRUE(demand);
    auto submit = std::async(std::launch::async, [&] { return prefetcher.demand_async(*provider, {1}); });
    auto submitted = submit.wait_for(std::chrono::seconds(1));
    bool gpu_pending = false;
    if (submitted == std::future_status::ready) {
        dynamic_cast<cldnn::ocl::ocl_stream&>(*compute).enqueue_external_barrier({submit.get()});
        auto consumer = compute->enqueue_marker({}, true);
        compute->flush();
        gpu_pending = !consumer->is_set();
    }
    gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(submitted, std::future_status::ready);
    EXPECT_TRUE(gpu_pending);
    if (submit.valid()) {
        EXPECT_NO_THROW(submit.get());
    }
    EXPECT_NO_THROW(prefetcher.drain());
    ASSERT_TRUE(demand);
    EXPECT_EQ(demand->ready_pairs, (std::vector<size_t>{1, 3}));
    EXPECT_EQ(demand->missing_pairs, (std::vector<size_t>{0, 2}));
    EXPECT_EQ(demand->slots[0], demand->slots[2]);
    EXPECT_EQ(demand->slots[1], demand->slots[3]);
    std::vector<uint8_t> actual(16);
    weights.gate_w->copy_to(*compute, actual.data(), demand->slots[0] * actual.size(), 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 11));
    weights.gate_w->copy_to(*compute, actual.data(), demand->slots[1] * actual.size(), 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 7));
    provider->cache().unpin_all(compute->enqueue_marker({}, true));
    compute->flush();
}

TEST_F(moe_les_upload, current_demand_does_not_wait_for_next_prediction) {
    write(std::vector<uint8_t>(128, 11));
    MOECompressed::Config config{};
    config.num_expert = 8;
    config.top_k = 1;
    auto provider = std::make_shared<OffloadExpertWeightProvider>(8, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 8);
    provider->bind(weights);
    auto current_provider = std::make_shared<OffloadExpertWeightProvider>(8, config, std::vector<size_t>(9), path);
    moe_weights current_weights;
    current_weights.gate_w = buffer(16, data_types::u8, 8);
    current_provider->bind(current_weights);
    auto compute = lane();
    auto predicted_ids = buffer(1, data_types::i32);
    const uint32_t predicted_expert = 1;
    predicted_ids->copy_from(*compute, &predicted_expert, true);
    compute->flush();
    ov::intel_gpu::ocl::moe::ExpertPrefetcher prefetcher(get_test_engine(), get_test_default_config(get_test_engine()));
    prefetcher.bind(provider);
    prefetcher.bind(current_provider);
    prefetcher.advance(1, 0);
    auto current = current_provider->cache().partition_3way({0}, [&] { return prefetcher.make_ready_event(); });
    ASSERT_TRUE(current);
    auto current_copy = prefetcher.demand_async(*current_provider, {0});
    cl::UserEvent prediction_gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    auto consumed = std::make_shared<std::atomic<bool>>(false);
    prefetcher.submit_prefetch({1, 1, predicted_ids, std::make_shared<cldnn::ocl::ocl_event>(prediction_gate), provider, consumed});
    dynamic_cast<cldnn::ocl::ocl_stream&>(*compute).enqueue_external_barrier({current_copy});
    auto pass_b_ready = compute->enqueue_marker({}, true);
    compute->flush();
    auto current_finished = std::async(std::launch::async, [&] { pass_b_ready->wait(); });
    const auto before_prediction = current_finished.wait_for(std::chrono::seconds(2));
    EXPECT_FALSE(consumed->load(std::memory_order_acquire));
    EXPECT_EQ(provider->cache().size(), 0U);
    prediction_gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(before_prediction, std::future_status::ready);
    EXPECT_NO_THROW(current_finished.get());
    current_provider->cache().unpin_all(pass_b_ready);
    EXPECT_NO_THROW(prefetcher.drain());
    EXPECT_TRUE(consumed->load(std::memory_order_acquire));
    auto demand = provider->cache().partition_3way({1}, [&] { return prefetcher.make_ready_event(); });
    ASSERT_TRUE(demand);
    EXPECT_EQ(demand->ready_pairs, std::vector<size_t>{0});
    EXPECT_TRUE(demand->missing.empty());
    std::vector<uint8_t> actual(16);
    weights.gate_w->copy_to(*compute, actual.data(), demand->slots.front() * actual.size(), 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 11));
    provider->cache().unpin_all(compute->enqueue_marker({}, true));
    compute->flush();
}

TEST_F(moe_les_upload, cancellation_terminates_unsubmitted_completion_bridge) {
    auto compute = lane();
    event::ptr completion;
    event::ptr consumer;
    {
        ov::intel_gpu::ocl::moe::ExpertPrefetcher prefetcher(get_test_engine(), get_test_default_config(get_test_engine()));
        completion = prefetcher.make_ready_event();
        dynamic_cast<cldnn::ocl::ocl_stream&>(*compute).enqueue_external_barrier({completion});
        consumer = compute->enqueue_marker({}, true);
        compute->flush();
        prefetcher.cancel_all();
    }
    auto native = std::dynamic_pointer_cast<cldnn::ocl::ocl_event>(completion);
    ASSERT_NE(native, nullptr);
    EXPECT_EQ(native->get().getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(), CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    EXPECT_ANY_THROW(consumer->wait());
}

TEST_F(moe_les_upload, stale_prediction_is_cancelled_without_waiting_for_gpu) {
    write(std::vector<uint8_t>(128, 11));
    MOECompressed::Config config{};
    config.num_expert = 8;
    config.top_k = 1;
    auto provider = std::make_shared<OffloadExpertWeightProvider>(8, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 8);
    provider->bind(weights);
    ov::intel_gpu::ocl::moe::ExpertPrefetcher prefetcher(get_test_engine(), get_test_default_config(get_test_engine()));
    prefetcher.bind(provider);
    prefetcher.advance(1, 0);
    cl::UserEvent prediction_gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    auto consumed = std::make_shared<std::atomic<bool>>(false);
    prefetcher.submit_prefetch({1, 1, buffer(1, data_types::i32),
                               std::make_shared<cldnn::ocl::ocl_event>(prediction_gate), provider, consumed});
    prefetcher.advance(1, 1);
    auto drained = std::async(std::launch::async, [&] { prefetcher.drain(); });
    const auto status = drained.wait_for(std::chrono::seconds(2));
    prediction_gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_NO_THROW(drained.get());
    EXPECT_TRUE(consumed->load(std::memory_order_acquire));
    EXPECT_EQ(provider->cache().size(), 0U);
}

TEST_F(moe_les_upload, lookahead_upload_retains_resources_after_provider_destruction) {
    write(std::vector<uint8_t>(32, 11));
    MOECompressed::Config config{};
    config.num_expert = 2;
    auto provider = std::make_shared<OffloadExpertWeightProvider>(1, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider->bind(weights);
    auto compute = lane();
    ASSERT_TRUE(provider->try_acquire_simultaneous({0}, *compute));
    ov::intel_gpu::ocl::moe::ExpertPrefetcher prefetcher(get_test_engine(), get_test_default_config(get_test_engine()));
    prefetcher.bind(provider);
    auto make_event = [&] { return prefetcher.make_ready_event(); };
    ASSERT_TRUE(provider->cache().partition_3way({0}, make_event));
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    provider->cache().unpin_all(std::make_shared<cldnn::ocl::ocl_event>(gate));
    ASSERT_TRUE(provider->cache().partition_3way({1}, make_event));
    auto completion = prefetcher.demand_async(*provider, {1});
    std::weak_ptr<cldnn::memory> backing = weights.gate_w;
    provider.reset();
    weights = {};
    EXPECT_FALSE(backing.expired());
    gate.setStatus(CL_COMPLETE);
    EXPECT_NO_THROW(prefetcher.drain());
    EXPECT_NO_THROW(completion->wait());
    auto retained = backing.lock();
    ASSERT_NE(retained, nullptr);
    std::vector<uint8_t> actual(16);
    retained->copy_to(*compute, actual.data(), 0, 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 11));
}

TEST_F(moe_les_upload, lookahead_cancellation_allows_callback_after_worker_destruction) {
    write(std::vector<uint8_t>(32, 7));
    MOECompressed::Config config{};
    config.num_expert = 2;
    auto provider = std::make_shared<OffloadExpertWeightProvider>(1, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider->bind(weights);
    auto compute = lane();
    ASSERT_TRUE(provider->try_acquire_simultaneous({0}, *compute));
    auto prefetcher = std::make_unique<ov::intel_gpu::ocl::moe::ExpertPrefetcher>(
        get_test_engine(), get_test_default_config(get_test_engine()));
    prefetcher->bind(provider);
    auto make_event = [&] { return prefetcher->make_ready_event(); };
    ASSERT_TRUE(provider->cache().partition_3way({0}, make_event));
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    provider->cache().unpin_all(std::make_shared<cldnn::ocl::ocl_event>(gate));
    ASSERT_TRUE(provider->cache().partition_3way({1}, make_event));
    auto completion = prefetcher->demand_async(*provider, {1});
    auto destroyed = std::async(std::launch::async, [&] { prefetcher.reset(); });
    const auto status = destroyed.wait_for(std::chrono::seconds(2));
    gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_NO_THROW(destroyed.get());
    EXPECT_ANY_THROW(completion->wait());
}

TEST_F(moe_les_upload, lookahead_upload_failure_terminates_all_waiters) {
    write(std::vector<uint8_t>(16, 7));
    MOECompressed::Config config{};
    config.num_expert = 2;
    auto provider = std::make_shared<OffloadExpertWeightProvider>(2, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider->bind(weights);
    ov::intel_gpu::ocl::moe::ExpertPrefetcher prefetcher(get_test_engine(), get_test_default_config(get_test_engine()));
    prefetcher.bind(provider);
    auto demand = provider->cache().partition_3way({1, 0}, [&] { return prefetcher.make_ready_event(); });
    ASSERT_TRUE(demand);
    auto completion = prefetcher.demand_async(*provider, {1, 0});
    EXPECT_ANY_THROW(prefetcher.drain());
    EXPECT_ANY_THROW(prefetcher.rethrow_if_failed());
    EXPECT_ANY_THROW(completion->wait());
    for (const auto& event : demand->wait_events) {
        EXPECT_ANY_THROW(event->wait());
    }
}

TEST_F(moe_les_upload, split_demand_release_drains_unfinished_upload) {
    write(std::vector<uint8_t>(32, 7));
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(1, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    auto transfer = lane();
    cldnn::les_context context(2, transfer);
    provider.set_les_context(&context, context.begin_epoch());
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    dynamic_cast<cldnn::ocl::ocl_stream&>(*transfer).enqueue_external_barrier({std::make_shared<cldnn::ocl::ocl_event>(gate)});
    transfer->flush();
    auto demand = provider.begin_les_demand({0});
    auto released = std::async(std::launch::async, [&] { provider.release(); });
    auto blocked = released.wait_for(std::chrono::milliseconds(20));
    gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(blocked, std::future_status::timeout);
    ASSERT_EQ(released.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_NO_THROW(released.get());
    ASSERT_TRUE(demand);
    auto retry = provider.begin_les_demand({0});
    ASSERT_TRUE(retry);
    EXPECT_TRUE(retry->ready_pairs.empty());
    EXPECT_EQ(retry->missing_pairs, std::vector<size_t>{0});
    provider.finish_les_demand(get_test_stream());
    provider.release();
}

TEST_F(moe_les_upload, split_demand_destruction_joins_active_upload) {
    write(std::vector<uint8_t>(32, 7));
    MOECompressed::Config config{};
    config.num_expert = 2;
    auto transfer = lane();
    cldnn::les_context context(2, transfer);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    auto provider = std::make_unique<OffloadExpertWeightProvider>(1, config, std::vector<size_t>(9), path);
    provider->bind(weights);
    provider->set_les_context(&context, context.begin_epoch());
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    dynamic_cast<cldnn::ocl::ocl_stream&>(*transfer).enqueue_external_barrier({std::make_shared<cldnn::ocl::ocl_event>(gate)});
    transfer->flush();
    auto demand = provider->begin_les_demand({0});
    auto destroyed = std::async(std::launch::async, [&] { provider.reset(); });
    auto blocked = destroyed.wait_for(std::chrono::milliseconds(20));
    gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(blocked, std::future_status::timeout);
    ASSERT_EQ(destroyed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_NO_THROW(destroyed.get());
    ASSERT_TRUE(demand);
    EXPECT_NO_THROW(context.wait());
    std::vector<uint8_t> actual(16);
    weights.gate_w->copy_to(get_test_stream(), actual.data(), 0, 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 7));
}

TEST_F(moe_les_upload, provider_capacity_overflow_does_no_io) {
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(1, config, std::vector<size_t>(9), path);
    moe_weights weights;
    provider.bind(weights);
    cldnn::les_context context(2, lane());
    provider.set_les_context(&context, context.begin_epoch());
    EXPECT_FALSE(provider.try_acquire_simultaneous({0, 1}, get_test_stream()));
    EXPECT_NO_THROW(context.wait());
}

TEST_F(moe_les_upload, provider_waits_for_real_consumer_before_reusing_slot) {
    auto file = std::vector<uint8_t>(32, 7);
    std::fill(file.begin() + 16, file.end(), 11);
    write(file);
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(1, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    cldnn::les_context context(2, lane());
    provider.set_les_context(&context, context.begin_epoch());
    auto compute = lane();
    ASSERT_TRUE(provider.try_acquire_simultaneous({0}, *compute));
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    dynamic_cast<cldnn::ocl::ocl_stream&>(*compute).enqueue_external_barrier({std::make_shared<cldnn::ocl::ocl_event>(gate)});
    provider.set_consumer(compute->enqueue_marker({}));
    compute->flush();
    auto next = std::async(std::launch::async, [&] { return provider.try_acquire_simultaneous({1}, *compute); });
    auto blocked = next.wait_for(std::chrono::milliseconds(20));
    gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(blocked, std::future_status::timeout);
    ASSERT_EQ(next.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_TRUE(next.get());
    std::vector<uint8_t> actual(16);
    weights.gate_w->copy_to(*compute, actual.data(), 0, 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 11));
    provider.release();
}

TEST_F(moe_les_upload, provider_recovers_after_partial_upload_failure) {
    write(std::vector<uint8_t>(64, 7));
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(1, config, std::vector<size_t>{0, 32, 0, 0, 0, 0, 0, 0, 0}, path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    weights.up_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    ASSERT_EQ(provider.weight_reader().file_size(), 64U);
    std::filesystem::resize_file(path, 32);
    cldnn::les_context context(2, lane());
    provider.set_les_context(&context, context.begin_epoch());
    auto compute = lane();
    EXPECT_ANY_THROW(provider.try_acquire_simultaneous({0}, *compute));
    context.cancel();
    write(std::vector<uint8_t>(64, 7));
    provider.set_les_context(&context, context.begin_epoch());
    ASSERT_TRUE(provider.try_acquire_simultaneous({0}, *compute));
    provider.release();
}

TEST_F(moe_les_upload, speculative_prefetch_failure_does_not_poison_real_demand) {
    write(std::vector<uint8_t>(64, 7));
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(2, config, std::vector<size_t>{0, 32, 0, 0, 0, 0, 0, 0, 0}, path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    weights.up_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    ASSERT_EQ(provider.weight_reader().file_size(), 64U);
    std::filesystem::resize_file(path, 32);
    cldnn::les_context context(2, lane());
    provider.set_les_context(&context, context.begin_epoch());
    auto compute = lane();
    EXPECT_NO_THROW(provider.prefetch_les_demand({1}, *compute));
    EXPECT_NO_THROW(context.wait());
    write(std::vector<uint8_t>(64, 11));
    EXPECT_NO_THROW(provider.try_acquire_simultaneous({1}, *compute));
    std::vector<uint8_t> actual(16);
    weights.up_w->copy_to(*compute, actual.data(), 0, 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 11));
    provider.release();
}

TEST_F(moe_les_upload, provider_cancelled_epoch_releases_queued_reservation) {
    write(std::vector<uint8_t>(32, 7));
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(1, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    cldnn::les_context context(2, lane());
    auto epoch = context.begin_epoch();
    provider.set_les_context(&context, epoch);
    std::promise<void> entered;
    std::promise<void> release;
    auto release_work = release.get_future().share();
    auto active = context.submit(epoch, [&] { entered.set_value(); release_work.wait(); });
    entered.get_future().wait();
    auto compute = lane();
    auto demand = std::async(std::launch::async, [&] { return provider.try_acquire_simultaneous({0}, *compute); });
    auto cancelled = std::async(std::launch::async, [&] { context.cancel(); });
    auto status = demand.wait_for(std::chrono::seconds(5));
    release.set_value();
    cancelled.get();
    ASSERT_EQ(status, std::future_status::ready);
    EXPECT_THROW(demand.get(), cldnn::les_task_cancelled);
    EXPECT_THROW(active.completion.get(), cldnn::les_task_cancelled);
    provider.set_les_context(&context, context.begin_epoch());
    ASSERT_TRUE(provider.try_acquire_simultaneous({0}, *compute));
    provider.release();
}

TEST_F(moe_les_upload, rejected_speculation_releases_reservation_without_upload) {
    std::vector<uint8_t> file(32, 7);
    std::fill(file.begin() + 16, file.end(), 11);
    write(file);
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(2, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    auto compute = lane();
    std::vector<uint8_t> sentinel(16, 77);
    weights.gate_w->copy_from(*compute, sentinel.data(), 0, 0, sentinel.size(), true);
    cldnn::les_context context(1, lane());
    auto epoch = context.begin_epoch();
    provider.set_les_context(&context, epoch);
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto queued = context.submit(epoch, [] {});
    auto speculative = std::async(std::launch::async, [&] { provider.prefetch_les_demand({1}, *compute); });
    const auto status = speculative.wait_for(std::chrono::seconds(5));
    release.set_value();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_NO_THROW(speculative.get());
    EXPECT_NO_THROW(active.completion.get());
    EXPECT_NO_THROW(queued.completion.get());
    std::vector<uint8_t> actual(16);
    weights.gate_w->copy_to(*compute, actual.data(), 0, 0, actual.size(), true);
    EXPECT_EQ(actual, sentinel);
    ASSERT_TRUE(provider.try_acquire_simultaneous({1}, *compute));
    weights.gate_w->copy_to(*compute, actual.data(), 0, 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 11));
    provider.release();
}

TEST(moe_les_reservations, speculation_cannot_evict_active_demand_and_unused_slots_are_reclaimed) {
    ov::intel_gpu::ocl::moe::ExpertSlotReservations reservations(2);
    auto demand = reservations.reserve({0}, 1);
    ASSERT_TRUE(demand);
    ASSERT_TRUE(reservations.complete(*demand));
    auto speculative = reservations.reserve({1}, 1, true);
    ASSERT_TRUE(speculative);
    EXPECT_FALSE(reservations.reserve({2}, 1, true));
    ASSERT_TRUE(reservations.release(*demand));
    ASSERT_TRUE(reservations.complete(*speculative));
    reservations.discard_unused(*speculative, {});
    ASSERT_TRUE(reservations.release(*speculative));
    auto original = reservations.reserve({0}, 1);
    ASSERT_TRUE(original);
    EXPECT_TRUE(original->missing_experts.empty());
    EXPECT_TRUE(reservations.release(*original));
    auto replacement = reservations.reserve({2}, 1, true);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->missing_experts, (std::vector<uint32_t>{2}));
    EXPECT_TRUE(reservations.release(*replacement));
}

TEST(moe_les_reservations, fixed_speculative_policy_separates_horizon_rank_and_byte_budget) {
    using Policy = otd::LesSpeculationPolicy;
    EXPECT_EQ(Policy::lookahead_layers, 1U);
    EXPECT_EQ(Policy::predict_k, 4U);
    EXPECT_EQ(Policy::expert_budget(1, 16), 0U);
    EXPECT_EQ(Policy::expert_budget(16, 0), 0U);
    EXPECT_EQ(Policy::expert_budget(16, 16), 4U);
    EXPECT_EQ(Policy::expert_budget(4, 16), 1U);
    EXPECT_EQ(Policy::expert_budget(16, Policy::max_bytes), 1U);
    EXPECT_EQ(Policy::expert_budget(16, Policy::max_bytes + 1), 0U);
}

TEST(moe_les_reservations, rollback_unstarted_speculation_restores_full_cache) {
    ov::intel_gpu::ocl::moe::ExpertSlotReservations reservations(2);
    auto demand = reservations.reserve({0, 1}, 1);
    ASSERT_TRUE(demand);
    ASSERT_TRUE(reservations.complete(*demand));
    ASSERT_TRUE(reservations.release(*demand));
    auto speculative = reservations.reserve({2}, 1, true);
    ASSERT_TRUE(speculative);
    ASSERT_TRUE(reservations.rollback(*speculative));
    EXPECT_FALSE(reservations.complete(*speculative));
    auto original = reservations.reserve({0, 1}, 1);
    ASSERT_TRUE(original);
    EXPECT_TRUE(original->missing_experts.empty());
    EXPECT_TRUE(reservations.release(*original));
}

TEST_F(moe_les_upload, async_prefetch_deduplicates_and_real_demand_reuses_upload) {
    write(std::vector<uint8_t>(32, 11));
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(2, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    cldnn::les_context context(3, lane());
    auto epoch = context.begin_epoch();
    provider.set_les_context(&context, epoch);
    auto compute = lane();
    auto* perf = otd::get_perf_counters();
    auto uploaded = perf ? perf->les_upload_bytes.load() : 0;
    auto useful = perf ? perf->les_prefetch_useful_bytes.load() : 0;
    auto deduplicated = perf ? perf->les_prefetch_deduplicated.load() : 0;
    auto skipped = perf ? perf->les_prefetch_skipped.load() : 0;
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    EXPECT_EQ(started.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto speculative = std::async(std::launch::async, [&] {
        provider.prefetch_les_demand({1}, *compute);
        provider.prefetch_les_demand({1}, *compute);
        provider.prefetch_les_demand({0}, *compute);
    });
    auto status = speculative.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        release.set_value();
    }
    EXPECT_EQ(status, std::future_status::ready);
    speculative.get();
    auto demand = std::async(std::launch::async, [&] { return provider.try_acquire_simultaneous({1}, *compute); });
    if (status == std::future_status::ready) {
        release.set_value();
    }
    ASSERT_TRUE(demand.get());
    EXPECT_NO_THROW(active.completion.get());
    std::vector<uint8_t> actual(16);
    weights.gate_w->copy_to(*compute, actual.data(), 0, 0, actual.size(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(16, 11));
    if (perf) {
        EXPECT_EQ(perf->les_upload_bytes.load() - uploaded, 16U);
        EXPECT_EQ(perf->les_prefetch_useful_bytes.load() - useful, 16U);
        EXPECT_EQ(perf->les_prefetch_deduplicated.load() - deduplicated, 1U);
        EXPECT_EQ(perf->les_prefetch_skipped.load() - skipped, 1U);
    }
    provider.release();
}

TEST_F(moe_les_upload, wrong_completed_prediction_counts_useless_bytes_without_extra_miss) {
    write(std::vector<uint8_t>(32, 11));
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(2, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    cldnn::les_context context(2, lane());
    provider.set_les_context(&context, context.begin_epoch());
    auto compute = lane();
    auto* perf = otd::get_perf_counters();
    auto useless = perf ? perf->les_prefetch_useless_bytes.load() : 0;
    auto useful = perf ? perf->les_prefetch_useful_bytes.load() : 0;
    auto extra = perf ? perf->les_prefetch_extra_misses.load() : 0;
    provider.prefetch_les_demand({1}, *compute);
    context.wait();
    ASSERT_TRUE(provider.try_acquire_simultaneous({0}, *compute));
    provider.release();
    if (perf) {
        EXPECT_EQ(perf->les_prefetch_useless_bytes.load() - useless, 16U);
        EXPECT_EQ(perf->les_prefetch_useful_bytes.load() - useful, 0U);
        EXPECT_EQ(perf->les_prefetch_extra_misses.load() - extra, 0U);
    }
    auto next = provider.begin_les_demand({0});
    ASSERT_TRUE(next);
    EXPECT_TRUE(next->missing_pairs.empty());
    provider.finish_les_demand(*compute);
    provider.release();
}

TEST_F(moe_les_upload, full_cache_speculation_records_pollution_against_demand_shadow) {
    write(std::vector<uint8_t>(48, 11));
    MOECompressed::Config config{};
    config.num_expert = 3;
    OffloadExpertWeightProvider provider(2, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 3);
    provider.bind(weights);
    cldnn::les_context context(2, lane());
    provider.set_les_context(&context, context.begin_epoch());
    auto compute = lane();
    ASSERT_TRUE(provider.try_acquire_simultaneous({0, 1}, *compute));
    provider.release();
    auto* perf = otd::get_perf_counters();
    auto useless = perf ? perf->les_prefetch_useless_bytes.load() : 0;
    auto extra = perf ? perf->les_prefetch_extra_misses.load() : 0;
    provider.prefetch_les_demand({2}, *compute);
    context.wait();
    ASSERT_TRUE(provider.try_acquire_simultaneous({0}, *compute));
    provider.release();
    if (perf) {
        EXPECT_EQ(perf->les_prefetch_useless_bytes.load() - useless, 16U);
        EXPECT_EQ(perf->les_prefetch_extra_misses.load() - extra, 1U);
    }
}

TEST_F(moe_les_upload, queued_prefetch_epoch_cancel_restores_mapping_and_retries) {
    write(std::vector<uint8_t>(48, 11));
    MOECompressed::Config config{};
    config.num_expert = 3;
    OffloadExpertWeightProvider provider(2, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 3);
    provider.bind(weights);
    cldnn::les_context context(2, lane());
    auto epoch = context.begin_epoch();
    provider.set_les_context(&context, epoch);
    auto compute = lane();
    ASSERT_TRUE(provider.try_acquire_simultaneous({0, 1}, *compute));
    provider.release();
    auto* perf = otd::get_perf_counters();
    auto uploaded = perf ? perf->les_upload_bytes.load() : 0;
    auto failed = perf ? perf->les_prefetch_failures.load() : 0;
    std::promise<void> release;
    std::promise<void> started;
    auto active = context.submit(epoch, [gate = release.get_future().share(), &started] {
        started.set_value();
        gate.wait();
    });
    started.get_future().wait();
    provider.prefetch_les_demand({2}, *compute);
    auto sentinel = context.submit(epoch, [] {});
    auto cancelled = std::async(std::launch::async, [&] { context.cancel(); });
    auto status = sentinel.completion.wait_for(std::chrono::seconds(5));
    release.set_value();
    cancelled.get();
    ASSERT_EQ(status, std::future_status::ready);
    EXPECT_THROW(sentinel.completion.get(), cldnn::les_task_cancelled);
    provider.set_les_context(&context, context.begin_epoch());
    ASSERT_TRUE(provider.try_acquire_simultaneous({0, 1}, *compute));
    provider.release();
    if (perf) {
        EXPECT_EQ(perf->les_upload_bytes.load(), uploaded);
        EXPECT_EQ(perf->les_prefetch_failures.load(), failed);
    }
    provider.prefetch_les_demand({2}, *compute);
    context.wait();
    ASSERT_TRUE(provider.try_acquire_simultaneous({2}, *compute));
    provider.release();
}
}  // namespace

TEST(moe_expert_weight_provider, resident_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_FALSE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 0U);

    // Fully resident: try_acquire_simultaneous returns the expert ids unchanged, including duplicates
    // (no remap, no dedup) - the expert id is already the addressable slot.
    std::vector<uint32_t> experts = {3, 1, 4, 1, 5, 9, 2, 6};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 1, 4, 1, 5, 9, 2, 6}));
}

TEST(moe_expert_weight_provider, offload_construction_state) {
    MOECompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};
    OffloadExpertWeightProvider provider(/*capacity=*/16, config, offsets, "/path/to/weights.bin");

    ASSERT_TRUE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 16U);
    // Resident buffers are bound lazily at first execute, not at construction.
    ASSERT_FALSE(provider.is_bound());
    ASSERT_FALSE(provider.cache().is_initialized());
}

TEST(moe_expert_weight_provider, offload_bind_resident_buffers) {
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/8, config, {}, "/path/to/weights.bin");
    ASSERT_FALSE(provider.is_bound());

    cldnn::moe_weights resident;  // null device pointers are fine; only the binding flag matters here
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
    ASSERT_TRUE(provider.cache().is_initialized());

    // Idempotent: a second bind keeps the provider bound.
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
}

TEST(moe_expert_weight_provider, resident_try_acquire_simultaneous_is_identity) {
    ResidentExpertWeightProvider provider;
    std::vector<uint32_t> experts = {3, 7, 3, 1, 7};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 7, 3, 1, 7}));
}

TEST(moe_expert_weight_provider, resident_acquire_one_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_EQ(provider.acquire_one(42, get_test_stream()), 42U);
    ASSERT_EQ(provider.acquire_one(0, get_test_stream()), static_cast<size_t>(0));
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_capacity_overflow) {
    // Capacity=2 means at most 2 unique experts can be acquired simultaneously.
    // Requesting 3 unique experts must return nullopt (capacity overflow).
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate the LRU cache so experts 0,1 are hits (no disk I/O needed)
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 3 unique experts (0, 1, 2) exceeds capacity=2 -> nullopt
    std::vector<uint32_t> experts = {0, 1, 2, 0, 1};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_FALSE(lease.has_value());
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_within_capacity) {
    // Same setup but with only 2 unique experts -> should succeed
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 2 unique experts (0, 1) with duplicates -> within capacity=2
    std::vector<uint32_t> experts = {0, 1, 0, 1, 0};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    // Slots should be valid (cache returned slots for experts 0 and 1)
    ASSERT_EQ(lease->size(), 5U);
    // Duplicates must map to the same slot
    ASSERT_EQ((*lease)[0], (*lease)[2]);
    ASSERT_EQ((*lease)[0], (*lease)[4]);
    ASSERT_EQ((*lease)[1], (*lease)[3]);
}

TEST(moe_expert_weight_provider, offload_acquire_one_basic) {
    // acquire_one always succeeds (evicts LRU if needed)
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    auto [slot0, hit0] = cache.get_lru_item(5);
    cache.set_filled(slot0);

    // acquire_one should return a valid slot
    auto slot = provider.acquire_one(5, get_test_stream());
    ASSERT_EQ(slot, slot0);
}
