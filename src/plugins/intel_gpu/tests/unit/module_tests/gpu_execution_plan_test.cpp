// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils/gpu_execution_plan.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace cldnn;

namespace {

class plan_test_event final : public event {
protected:
    void wait_impl() override {}
    bool is_set_impl() override {
        return true;
    }
    void set_impl() override {}
};

class plan_test_kernel final : public kernel {
public:
    explicit plan_test_kernel(std::string id) : _id(std::move(id)) {}

    kernel::ptr clone(bool) const override {
        return std::make_shared<plan_test_kernel>(_id);
    }
    bool is_same(const kernel& other) const override {
        return get_id() == other.get_id();
    }
    std::string get_id() const override {
        return _id;
    }
    std::vector<uint8_t> get_binary() const override {
        return {};
    }
    std::string get_build_log() const override {
        return {};
    }

private:
    std::string _id;
};

class plan_test_stream final : public stream {
public:
    plan_test_stream() : stream(QueueTypes::in_order, SyncMethods::none) {}

    void flush() const override {}
    void finish() const override {}
    void wait() override {}
    void set_arguments(kernel&, const kernel_arguments_desc&, const kernel_arguments_data&) override {}
    event::ptr enqueue_kernel(kernel& selected,
                              const kernel_arguments_desc&,
                              const kernel_arguments_data&,
                              const std::vector<event::ptr>& dependencies,
                              bool is_output_event) override {
        kernel_ids.push_back(selected.get_id());
        dependency_counts.push_back(dependencies.size());
        output_event_requests.push_back(is_output_event);
        auto result = std::make_shared<plan_test_event>();
        events.push_back(result);
        return result;
    }
    event::ptr enqueue_marker(const std::vector<event::ptr>&, bool is_output_event) override {
        marker_output_event_requests.push_back(is_output_event);
        return std::make_shared<plan_test_event>();
    }
    void enqueue_barrier() override {}
    event::ptr group_events(const std::vector<event::ptr>&) override {
        return std::make_shared<plan_test_event>();
    }
    void wait_for_events(const std::vector<event::ptr>&) override {}
    event::ptr create_user_event(bool) override {
        return std::make_shared<plan_test_event>();
    }
    event::ptr create_base_event() override {
        return std::make_shared<plan_test_event>();
    }
    std::unique_ptr<surfaces_lock> create_surfaces_lock(const std::vector<memory::ptr>&) const override {
        return nullptr;
    }
#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream() override {
        throw std::runtime_error("oneDNN stream is not used by gpu_execution_plan tests");
    }
#endif

    std::vector<std::string> kernel_ids;
    std::vector<size_t> dependency_counts;
    std::vector<bool> output_event_requests;
    std::vector<bool> marker_output_event_requests;
    std::vector<event::ptr> events;
};

gpu_kernel_lifecycle make_lifecycle(size_t count) {
    std::vector<std::pair<kernel::ptr, size_t>> entries;
    for (size_t index = 0; index < count; ++index) {
        entries.emplace_back(std::make_shared<plan_test_kernel>("kernel_" + std::to_string(index)), index);
    }
    gpu_kernel_lifecycle lifecycle;
    lifecycle.adopt_entries(entries);
    return lifecycle;
}

}  // namespace

TEST(gpu_execution_plan, chains_arbitrary_kernel_count_and_requests_only_final_completion) {
    auto lifecycle = make_lifecycle(3);
    gpu_execution_plan plan(3);
    plan_test_stream stream;
    kernel_arguments_desc descriptor;

    const auto completion = plan.execute(stream, lifecycle, {}, true, [&](size_t) {
        return gpu_dispatch_binding{&descriptor, {}};
    });

    ASSERT_NE(completion, nullptr);
    EXPECT_EQ(stream.kernel_ids, (std::vector<std::string>{"kernel_0", "kernel_1", "kernel_2"}));
    EXPECT_EQ(stream.dependency_counts, (std::vector<size_t>{0, 1, 1}));
    EXPECT_EQ(stream.output_event_requests, (std::vector<bool>{false, false, true}));
}

TEST(gpu_execution_plan, suppresses_skipped_and_zero_size_dispatches) {
    auto lifecycle = make_lifecycle(2);
    gpu_execution_plan plan(2);
    plan[0].skip_execution = true;
    plan_test_stream stream;
    kernel_arguments_desc descriptor;

    plan.execute(stream, lifecycle, {}, true, [&](size_t) {
        return gpu_dispatch_binding{&descriptor, {}};
    });
    ASSERT_EQ(stream.kernel_ids, (std::vector<std::string>{"kernel_1"}));

    stream.kernel_ids.clear();
    plan.suppress_zero_size(true);
    plan.execute(stream, lifecycle, {}, true, [&](size_t) {
        return gpu_dispatch_binding{&descriptor, {}};
    });
    EXPECT_TRUE(stream.kernel_ids.empty());
}

TEST(gpu_execution_plan, preserves_independent_dispatch_and_aggregate_policy) {
    auto lifecycle = make_lifecycle(2);
    gpu_execution_plan plan(2, {true, true});
    plan[0].dependency = gpu_dispatch_dependency_policy::external;
    plan[1].dependency = gpu_dispatch_dependency_policy::external;
    plan_test_stream stream;
    kernel_arguments_desc descriptor;
    const std::vector<event::ptr> dependencies{std::make_shared<plan_test_event>()};

    const auto completion = plan.execute(stream, lifecycle, dependencies, true, [&](size_t) {
        return gpu_dispatch_binding{&descriptor, {}};
    });

    ASSERT_NE(completion, nullptr);
    EXPECT_EQ(stream.dependency_counts, (std::vector<size_t>{1, 1}));
    EXPECT_EQ(stream.output_event_requests, (std::vector<bool>{true, true}));
}

TEST(gpu_execution_plan, returns_requested_kernel_completion_without_output_marker) {
    auto lifecycle = make_lifecycle(1);
    gpu_execution_plan plan(1, {true, true});
    plan_test_stream stream;
    kernel_arguments_desc descriptor;

    size_t dispatch_count = 0;
    const auto completion = plan.execute_with(
        stream,
        lifecycle,
        {},
        true,
        [&](size_t) {
            return gpu_dispatch_binding{&descriptor, {}};
        },
        [&](size_t dispatch_index,
            kernel& selected_kernel,
            const kernel_arguments_desc& selected_descriptor,
            const kernel_arguments_data& arguments,
            const std::vector<event::ptr>& dependencies,
            bool request_completion) {
            EXPECT_EQ(dispatch_index, 0);
            ++dispatch_count;
            return stream.enqueue_kernel(selected_kernel, selected_descriptor, arguments, dependencies, request_completion);
        });

    ASSERT_EQ(stream.events.size(), 1);
    EXPECT_EQ(dispatch_count, 1);
    EXPECT_EQ(completion, stream.events.front());
    EXPECT_EQ(stream.output_event_requests, (std::vector<bool>{true}));
    EXPECT_TRUE(stream.marker_output_event_requests.empty());
}
