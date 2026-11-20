// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
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
    moe_weights weights;
    weights.gate_w = buffer(source.size());
    auto transfer = lane();
    otd::ParallelWeightReader reader(path);
    auto events = otd::fill_weights_memory_async(*transfer, config, std::vector<size_t>(9), reader, weights, {0}, {0});
    EXPECT_EQ(events.size(), 3U);
    std::vector<uint8_t> actual(source.size());
    weights.gate_w->copy_to(*transfer, actual.data(), true);
    EXPECT_EQ(actual, source);
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
    auto file = std::vector<uint8_t>(32, 7);
    std::fill(file.begin() + 16, file.end(), 11);
    write(file);
    MOECompressed::Config config{};
    config.num_expert = 2;
    OffloadExpertWeightProvider provider(2, config, std::vector<size_t>(9), path);
    moe_weights weights;
    weights.gate_w = buffer(16, data_types::u8, 2);
    provider.bind(weights);
    auto transfer = lane();
    cldnn::les_context context(2, transfer);
    provider.set_les_context(&context, context.begin_epoch());
    auto compute = lane();
    ASSERT_TRUE(provider.try_acquire_simultaneous({0}, *compute));
    provider.release();
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(get_test_engine()).get_cl_context());
    dynamic_cast<cldnn::ocl::ocl_stream&>(*transfer).enqueue_external_barrier({std::make_shared<cldnn::ocl::ocl_event>(gate)});
    transfer->flush();
    auto begin = std::async(std::launch::async, [&] { return provider.begin_les_demand({1, 0, 1, 0}); });
    auto begin_status = begin.wait_for(std::chrono::seconds(5));
    if (begin_status != std::future_status::ready) {
        gate.setStatus(CL_COMPLETE);
    }
    ASSERT_EQ(begin_status, std::future_status::ready);
    auto demand = begin.get();
    auto finish = std::async(std::launch::async, [&] { provider.finish_les_demand(*compute); });
    auto blocked = finish.wait_for(std::chrono::milliseconds(20));
    gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(blocked, std::future_status::timeout);
    ASSERT_EQ(finish.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_NO_THROW(finish.get());
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
    EXPECT_THROW(provider.finish_les_demand(*compute), ov::Exception);
    provider.release();
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
