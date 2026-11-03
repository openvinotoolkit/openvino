// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/command_list.hpp"
#include "intel_gpu/runtime/command_recorder.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace cldnn;

namespace {

// Minimal command_list implementation that counts backend calls so the base
// class state machine can be validated without a real device.
class mock_command_list : public command_list {
public:
    int reset_calls = 0;
    int close_calls = 0;
    int enqueue_calls = 0;
    int wait_calls = 0;

protected:
    void reset_impl() override { ++reset_calls; }
    void close_impl() override { ++close_calls; }
    void enqueue_impl() override { ++enqueue_calls; }
    void wait_impl() override { ++wait_calls; }
};

class mock_command_recorder : public command_recorder {
public:
    command_list::ptr create_command_list() const override {
        return std::make_shared<mock_command_list>();
    }
};

}  // namespace

TEST(command_list_test, new_list_is_open) {
    mock_command_list cmd_list;
    ASSERT_EQ(cmd_list.get_status(), command_list_status::open);
}

TEST(command_list_test, close_transitions_open_to_closed) {
    mock_command_list cmd_list;
    cmd_list.close();
    ASSERT_EQ(cmd_list.get_status(), command_list_status::closed);
    ASSERT_EQ(cmd_list.close_calls, 1);
}

TEST(command_list_test, close_when_not_open_throws) {
    mock_command_list cmd_list;
    cmd_list.close();
    ASSERT_ANY_THROW(cmd_list.close());
}

TEST(command_list_test, enqueue_when_open_throws) {
    mock_command_list cmd_list;
    ASSERT_ANY_THROW(cmd_list.enqueue());
    ASSERT_EQ(cmd_list.enqueue_calls, 0);
}

TEST(command_list_test, close_then_enqueue_transitions_to_enqueued) {
    mock_command_list cmd_list;
    cmd_list.close();
    cmd_list.enqueue();
    ASSERT_EQ(cmd_list.get_status(), command_list_status::enqueued);
    ASSERT_EQ(cmd_list.enqueue_calls, 1);
}

TEST(command_list_test, enqueue_while_enqueued_waits_before_resubmitting) {
    mock_command_list cmd_list;
    cmd_list.close();
    cmd_list.enqueue();
    cmd_list.enqueue();
    // Second enqueue must wait for the in-flight execution before re-submitting.
    ASSERT_EQ(cmd_list.wait_calls, 1);
    ASSERT_EQ(cmd_list.enqueue_calls, 2);
    ASSERT_EQ(cmd_list.get_status(), command_list_status::enqueued);
}

TEST(command_list_test, wait_when_open_throws) {
    mock_command_list cmd_list;
    ASSERT_ANY_THROW(cmd_list.wait());
    ASSERT_EQ(cmd_list.wait_calls, 0);
}

TEST(command_list_test, wait_when_closed_is_noop) {
    mock_command_list cmd_list;
    cmd_list.close();
    cmd_list.wait();
    ASSERT_EQ(cmd_list.wait_calls, 0);
    ASSERT_EQ(cmd_list.get_status(), command_list_status::closed);
}

TEST(command_list_test, wait_when_enqueued_transitions_to_closed) {
    mock_command_list cmd_list;
    cmd_list.close();
    cmd_list.enqueue();
    cmd_list.wait();
    ASSERT_EQ(cmd_list.wait_calls, 1);
    ASSERT_EQ(cmd_list.get_status(), command_list_status::closed);
}

TEST(command_list_test, reset_from_open_keeps_open) {
    mock_command_list cmd_list;
    cmd_list.reset();
    ASSERT_EQ(cmd_list.reset_calls, 1);
    ASSERT_EQ(cmd_list.get_status(), command_list_status::open);
}

TEST(command_list_test, reset_from_enqueued_waits_then_reopens) {
    mock_command_list cmd_list;
    cmd_list.close();
    cmd_list.enqueue();
    cmd_list.reset();
    ASSERT_EQ(cmd_list.wait_calls, 1);
    ASSERT_EQ(cmd_list.reset_calls, 1);
    ASSERT_EQ(cmd_list.get_status(), command_list_status::open);
}

TEST(command_recorder_test, no_active_command_list_by_default) {
    mock_command_recorder recorder;
    ASSERT_EQ(recorder.get_active_command_list(), nullptr);
}

TEST(command_recorder_test, start_recording_sets_active_command_list) {
    mock_command_recorder recorder;
    auto cmd_list = recorder.create_command_list();
    recorder.start_recording(cmd_list);
    ASSERT_EQ(recorder.get_active_command_list(), cmd_list);
}

TEST(command_recorder_test, start_recording_non_open_list_throws) {
    mock_command_recorder recorder;
    auto cmd_list = recorder.create_command_list();
    cmd_list->close();
    ASSERT_ANY_THROW(recorder.start_recording(cmd_list));
    ASSERT_EQ(recorder.get_active_command_list(), nullptr);
}

TEST(command_recorder_test, start_recording_twice_throws) {
    mock_command_recorder recorder;
    recorder.start_recording(recorder.create_command_list());
    ASSERT_ANY_THROW(recorder.start_recording(recorder.create_command_list()));
}

TEST(command_recorder_test, stop_recording_closes_and_enqueues) {
    mock_command_recorder recorder;
    auto cmd_list = std::static_pointer_cast<mock_command_list>(recorder.create_command_list());
    recorder.start_recording(cmd_list);

    auto finished = recorder.stop_recording();

    ASSERT_EQ(finished, cmd_list);
    ASSERT_EQ(cmd_list->close_calls, 1);
    ASSERT_EQ(cmd_list->enqueue_calls, 1);
    ASSERT_EQ(cmd_list->get_status(), command_list_status::enqueued);
    ASSERT_EQ(recorder.get_active_command_list(), nullptr);
}

TEST(command_recorder_test, stop_recording_without_active_returns_null) {
    mock_command_recorder recorder;
    ASSERT_EQ(recorder.stop_recording(), nullptr);
}
