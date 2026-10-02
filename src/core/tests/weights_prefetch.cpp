// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weights_prefetch.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/memory_prefetch.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::test {

using ov::weight_sharing::PrefetchScheduler;
using testing::_;

namespace {

constexpr size_t mib = 1UL << 20;

class MockMappedMemory final : public ov::MappedMemory {
public:
    explicit MockMappedMemory(size_t size) : m_data(size, '\0') {}

    const std::byte* data() const noexcept override {
        return reinterpret_cast<const std::byte*>(m_data.data());
    }
    std::byte* data() noexcept override {
        return reinterpret_cast<std::byte*>(m_data.data());
    }
    size_t size() const noexcept override {
        return m_data.size();
    }
    void hint_evict(size_t offset, size_t size) noexcept override {
        evicted(offset, size);
    }
    void hint_prefetch(size_t, size_t) noexcept override {}
    void hint_prefetch_async(size_t, size_t) override {}

    MOCK_METHOD(void, evicted, (size_t offset, size_t size));

private:
    std::vector<char> m_data;
};

using MmapBuffer = ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>;

std::shared_ptr<ov::op::v0::Constant> make_constant(const std::shared_ptr<ov::MappedMemory>& memory,
                                                    size_t offset,
                                                    size_t size) {
    auto buffer = std::make_shared<MmapBuffer>(reinterpret_cast<char*>(memory->data()) + offset, size, memory);
    return std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{size}, buffer);
}

// Constants laid out back to back in a single mapping, used one per step.
struct Layout {
    std::shared_ptr<MockMappedMemory> memory;
    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;

    Layout(size_t count, size_t size) : memory(std::make_shared<MockMappedMemory>(count * size)) {
        for (size_t i = 0; i < count; ++i) {
            constants.push_back(make_constant(memory, i * size, size));
        }
    }

    PrefetchScheduler::Plan plan(bool evict) const {
        PrefetchScheduler::Plan plan(constants.size());
        for (size_t i = 0; i < constants.size(); ++i) {
            plan[i].push_back({constants[i], evict});
        }
        return plan;
    }
};

PrefetchScheduler::Config make_config(size_t populate, size_t readahead) {
    PrefetchScheduler::Config config;
    config.populate_bytes = populate;
    config.readahead_bytes = readahead;
    return config;
}

void run_all(PrefetchScheduler& scheduler, size_t steps) {
    for (size_t step = 0; step < steps; ++step) {
        scheduler.begin(step);
        scheduler.end(step);
    }
}

class EnvGuard {
public:
    EnvGuard(const char* name, const char* value) : m_name(name) {
        if (const auto old = std::getenv(name)) {
            m_old = old;
        }
        set(value);
    }
    ~EnvGuard() {
        set(m_old ? m_old->c_str() : nullptr);
    }

private:
    void set(const char* value) {
#ifdef _WIN32
        _putenv_s(m_name.c_str(), value ? value : "");
#else
        if (value) {
            ::setenv(m_name.c_str(), value, 1);
        } else {
            ::unsetenv(m_name.c_str());
        }
#endif
    }

    std::string m_name;
    std::optional<std::string> m_old;
};

}  // namespace

TEST(MemoryPrefetchTest, populate_token_completes) {
    std::vector<char> data(8 * mib, 1);
    auto token = ov::util::prefetch_async(data.data(), data.size(), ov::util::PrefetchMode::populate);
    ASSERT_TRUE(token.valid());
    token.wait();
    EXPECT_TRUE(token.ready());
    EXPECT_FALSE(token.valid());
}

TEST(MemoryPrefetchTest, readahead_token_completes) {
    std::vector<char> data(8 * mib, 1);
    auto token = ov::util::prefetch_async(data.data(), data.size(), ov::util::PrefetchMode::readahead);
    token.wait();
    EXPECT_TRUE(token.ready());
}

TEST(MemoryPrefetchTest, cancelled_token_joins) {
    std::vector<char> data(64 * mib, 1);
    auto token = ov::util::prefetch_async(data.data(), data.size(), ov::util::PrefetchMode::populate);
    token.cancel();
    token.wait();
    EXPECT_TRUE(token.ready());
}

TEST(MemoryPrefetchTest, sub_page_or_null_range_is_ignored) {
    std::vector<char> data(16);
    EXPECT_FALSE(ov::util::prefetch_async(data.data(), data.size(), ov::util::PrefetchMode::populate).valid());
    EXPECT_FALSE(ov::util::prefetch_async(nullptr, 64 * mib, ov::util::PrefetchMode::populate).valid());
}

TEST(MemoryPrefetchTest, unaligned_range_on_mmap_file) {
    const auto path = ov::test::utils::generateTestFilePrefix() + "_prefetch.bin";
    {
        std::ofstream file(path, std::ios::binary);
        std::vector<char> data(8 * mib + 123, 7);
        file.write(data.data(), data.size());
    }
    {
        const auto mapped = ov::load_mmap_object(path);
        auto token =
            ov::util::prefetch_async(mapped->data() + 17, mapped->size() - 17, ov::util::PrefetchMode::populate);
        token.wait();
        EXPECT_EQ(static_cast<char>(mapped->data()[mapped->size() - 1]), 7);
    }
    std::filesystem::remove(path);
}

TEST(WeightsPrefetchSchedulerTest, populate_window_stays_within_budget) {
    const Layout layout(16, mib);
    PrefetchScheduler scheduler(layout.plan(false), make_config(3 * mib, 4 * mib));

    const auto& stats = scheduler.get_stats();
    EXPECT_EQ(stats.regions, 16);
    EXPECT_EQ(stats.populated, 3);
    EXPECT_EQ(stats.read_ahead, 4);

    run_all(scheduler, layout.constants.size());
    EXPECT_EQ(stats.populated, 16);
    EXPECT_EQ(stats.late, 0);
    EXPECT_LE(stats.peak_populated_ahead, 3 * mib);
    EXPECT_LE(stats.peak_read_ahead, 4 * mib);
    EXPECT_EQ(stats.evicted, 0);
}

TEST(WeightsPrefetchSchedulerTest, oversized_region_is_populated_alone) {
    const Layout layout(4, 4 * mib);
    PrefetchScheduler scheduler(layout.plan(false), make_config(mib, 0));
    EXPECT_EQ(scheduler.get_stats().populated, 1);
    run_all(scheduler, layout.constants.size());
    EXPECT_EQ(scheduler.get_stats().populated, 4);
    EXPECT_EQ(scheduler.get_stats().read_ahead, 0);
    EXPECT_EQ(scheduler.get_stats().late, 0);
}

TEST(WeightsPrefetchSchedulerTest, zero_populate_budget_populates_on_demand_only) {
    const Layout layout(4, mib);
    PrefetchScheduler scheduler(layout.plan(false), make_config(0, 8 * mib));
    EXPECT_EQ(scheduler.get_stats().populated, 0);
    EXPECT_EQ(scheduler.get_stats().read_ahead, 4);
    run_all(scheduler, layout.constants.size());
    EXPECT_EQ(scheduler.get_stats().populated, 0);
    EXPECT_EQ(scheduler.get_stats().late, 4);
}

TEST(WeightsPrefetchSchedulerTest, evicts_after_last_use_only) {
    const Layout layout(4, mib);
    PrefetchScheduler scheduler(layout.plan(true), make_config(2 * mib, 0));

    for (size_t step = 0; step < layout.constants.size(); ++step) {
        scheduler.begin(step);
        testing::Mock::VerifyAndClearExpectations(layout.memory.get());
        EXPECT_CALL(*layout.memory, evicted(step * mib, mib)).Times(1);
        scheduler.end(step);
        testing::Mock::VerifyAndClearExpectations(layout.memory.get());
    }
    EXPECT_EQ(scheduler.get_stats().evicted, 4);
}

TEST(WeightsPrefetchSchedulerTest, does_not_evict_when_not_allowed) {
    const Layout layout(4, mib);
    EXPECT_CALL(*layout.memory, evicted(_, _)).Times(0);
    PrefetchScheduler scheduler(layout.plan(false), make_config(2 * mib, 0));
    run_all(scheduler, layout.constants.size());
}

TEST(WeightsPrefetchSchedulerTest, shared_data_forms_single_region_evicted_after_last_use) {
    const Layout layout(1, mib);
    // A second constant over the same data, e.g. tied embedding and lm_head weights.
    const auto tied = make_constant(layout.memory, 0, mib);

    PrefetchScheduler::Plan plan(4);
    plan[0].push_back({layout.constants[0], true});
    plan[3].push_back({tied, true});

    PrefetchScheduler scheduler(plan, make_config(4 * mib, 0));
    EXPECT_EQ(scheduler.get_stats().regions, 1);

    EXPECT_CALL(*layout.memory, evicted(_, _)).Times(0);
    for (size_t step = 0; step < 3; ++step) {
        scheduler.begin(step);
        scheduler.end(step);
    }
    testing::Mock::VerifyAndClearExpectations(layout.memory.get());

    // Both constants are evicted, they cover the same pages.
    EXPECT_CALL(*layout.memory, evicted(0, mib)).Times(2);
    scheduler.begin(3);
    scheduler.end(3);
    EXPECT_EQ(scheduler.get_stats().evicted, 1);
}

TEST(WeightsPrefetchSchedulerTest, evict_is_disabled_if_any_use_forbids_it) {
    const Layout layout(1, mib);
    const auto tied = make_constant(layout.memory, 0, mib);

    PrefetchScheduler::Plan plan(2);
    plan[0].push_back({layout.constants[0], true});
    plan[1].push_back({tied, false});

    EXPECT_CALL(*layout.memory, evicted(_, _)).Times(0);
    PrefetchScheduler scheduler(plan, make_config(4 * mib, 0));
    run_all(scheduler, plan.size());
}

TEST(WeightsPrefetchSchedulerTest, skipped_steps_free_the_window) {
    const Layout layout(8, mib);
    PrefetchScheduler scheduler(layout.plan(false), make_config(2 * mib, 0));
    EXPECT_EQ(scheduler.get_stats().populated, 2);

    // The consumer jumps straight to the last steps.
    scheduler.begin(6);
    scheduler.end(6);
    scheduler.begin(7);
    scheduler.end(7);
    EXPECT_LE(scheduler.get_stats().peak_populated_ahead, 2 * mib);
    // Step 6 was outside of the window when the consumer jumped to it, step 7 was populated meanwhile.
    EXPECT_EQ(scheduler.get_stats().late, 1);
}

TEST(WeightsPrefetchSchedulerTest, unreached_populated_regions_are_evicted_on_destruction) {
    const Layout layout(4, mib);
    EXPECT_CALL(*layout.memory, evicted(_, _)).Times(2);
    PrefetchScheduler scheduler(layout.plan(true), make_config(2 * mib, 0));
}

TEST(WeightsPrefetchSchedulerTest, expired_constants_are_skipped) {
    auto layout = std::make_unique<Layout>(4, mib);
    auto plan = layout->plan(false);
    auto memory = layout->memory;
    layout->constants[2].reset();
    layout->constants[3].reset();

    PrefetchScheduler scheduler(plan, make_config(mib, 0));
    EXPECT_EQ(scheduler.get_stats().regions, 2);
    layout->constants[1].reset();
    run_all(scheduler, plan.size());
    EXPECT_EQ(scheduler.get_stats().populated, 1);
}

TEST(WeightsPrefetchSchedulerTest, small_constants_are_ignored) {
    const Layout layout(4, 64);
    PrefetchScheduler scheduler(layout.plan(true), make_config(mib, mib));
    EXPECT_EQ(scheduler.get_stats().regions, 0);
    run_all(scheduler, layout.constants.size());
}

TEST(WeightsPrefetchSchedulerTest, out_of_range_steps_are_ignored) {
    const Layout layout(2, mib);
    PrefetchScheduler scheduler(layout.plan(false), make_config(mib, mib));
    scheduler.begin(100);
    scheduler.end(100);
}

TEST(WeightsPrefetchSchedulerTest, config_from_env) {
    {
        EnvGuard enable("OV_WEIGHTS_PREFETCH", nullptr);
        EXPECT_FALSE(PrefetchScheduler::Config::from_env("cf"));
    }
    {
        EnvGuard enable("OV_WEIGHTS_PREFETCH", "1");
        EnvGuard sites("OV_WEIGHTS_PREFETCH_SITES", "cf,gpu_build");
        EnvGuard populate("OV_WEIGHTS_PREFETCH_POPULATE_MB", "16");
        EnvGuard readahead("OV_WEIGHTS_PREFETCH_READAHEAD_MB", "0");
        const auto config = PrefetchScheduler::Config::from_env("cf");
        ASSERT_TRUE(config);
        EXPECT_EQ(config->populate_bytes, 16 * mib);
        EXPECT_EQ(config->readahead_bytes, 0);
        EXPECT_EQ(config->site, "cf");
        EXPECT_FALSE(PrefetchScheduler::Config::from_env("cpu_compile"));
    }
    {
        EnvGuard enable("OV_WEIGHTS_PREFETCH", "1");
        EnvGuard sites("OV_WEIGHTS_PREFETCH_SITES", nullptr);
        EnvGuard populate("OV_WEIGHTS_PREFETCH_POPULATE_MB", nullptr);
        EnvGuard readahead("OV_WEIGHTS_PREFETCH_READAHEAD_MB", nullptr);
        const auto defaults = PrefetchScheduler::Config::from_env("gpu_build");
        ASSERT_TRUE(defaults);
        EXPECT_EQ(defaults->populate_bytes, 64 * mib);
        EXPECT_EQ(defaults->readahead_bytes, 0);

        PrefetchScheduler::Config site_defaults;
        site_defaults.populate_bytes = 256 * mib;
        const auto config = PrefetchScheduler::Config::from_env("cpu_compile", site_defaults);
        ASSERT_TRUE(config);
        EXPECT_EQ(config->populate_bytes, 256 * mib);
    }
}

TEST(WeightsPrefetchSchedulerTest, constant_folding_results_do_not_depend_on_prefetch) {
    const auto path = ov::test::utils::generateTestFilePrefix() + "_cf_prefetch.bin";
    constexpr size_t count = 3 * mib / sizeof(float);
    {
        std::vector<float> values(2 * count);
        std::iota(values.begin(), values.end(), 0.f);
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    }

    auto fold = [&]() {
        const auto mapped = ov::load_mmap_object(path);
        auto make = [&](size_t offset) {
            auto buffer = std::make_shared<MmapBuffer>(reinterpret_cast<char*>(mapped->data()) + offset,
                                                       count * sizeof(float),
                                                       mapped);
            return std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{count}, buffer);
        };
        const auto lhs = make(0);
        const auto rhs = make(count * sizeof(float));
        const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{count});
        const auto folded = std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
        const auto add = std::make_shared<ov::op::v1::Add>(param, folded);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{param});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(model);

        const auto result = ov::as_type_ptr<ov::op::v0::Constant>(add->get_input_node_shared_ptr(1));
        EXPECT_NE(result, nullptr);
        return result ? result->cast_vector<float>() : std::vector<float>{};
    };

    std::vector<float> reference;
    {
        EnvGuard enable("OV_WEIGHTS_PREFETCH", nullptr);
        reference = fold();
    }
    std::vector<float> prefetched;
    {
        EnvGuard enable("OV_WEIGHTS_PREFETCH", "1");
        EnvGuard populate("OV_WEIGHTS_PREFETCH_POPULATE_MB", "1");
        prefetched = fold();
    }
    std::filesystem::remove(path);

    ASSERT_EQ(reference.size(), count);
    EXPECT_EQ(reference, prefetched);
}

TEST(WeightsPrefetchConstantFoldingTest, constants_read_by_folding_only_are_released) {
    const Layout layout(2, mib);
    const auto folded = std::make_shared<ov::op::v1::Add>(layout.constants[0], layout.constants[1]);
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{mib});
    const auto add = std::make_shared<ov::op::v1::Add>(param, folded);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{param});

    EXPECT_CALL(*layout.memory, evicted(0, mib)).Times(1);
    EXPECT_CALL(*layout.memory, evicted(mib, mib)).Times(1);
    {
        EnvGuard enable("OV_WEIGHTS_PREFETCH", "1");
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(model);
    }
    EXPECT_NE(ov::as_type_ptr<ov::op::v0::Constant>(add->get_input_node_shared_ptr(1)), nullptr);
}

TEST(WeightsPrefetchConstantFoldingTest, view_folds_are_not_prefetched_nor_released) {
    const Layout layout(1, mib);
    const auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1024, 1024});
    const auto reshape = std::make_shared<ov::op::v1::Reshape>(layout.constants[0], shape, false);
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{1024, 1024});
    const auto add = std::make_shared<ov::op::v1::Add>(param, reshape);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{param});

    EXPECT_CALL(*layout.memory, evicted(_, _)).Times(0);
    {
        EnvGuard enable("OV_WEIGHTS_PREFETCH", "1");
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(model);
    }
    const auto folded = ov::as_type_ptr<ov::op::v0::Constant>(add->get_input_node_shared_ptr(1));
    ASSERT_NE(folded, nullptr);
    EXPECT_EQ(folded->get_data_ptr(), layout.constants[0]->get_data_ptr());
}

}  // namespace ov::test
