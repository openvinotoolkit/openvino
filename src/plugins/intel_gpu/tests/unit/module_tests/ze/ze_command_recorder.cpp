// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#include "ze_test_context.hpp"

#include "intel_gpu/runtime/command_list.hpp"
#include "intel_gpu/runtime/command_recorder.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/graph/record_replay_session.hpp"

#include "runtime/ze/ze_stream.hpp"
#include "runtime/ze/ze_command_list.hpp"
#include "runtime/ze/ze_base_event.hpp"
#include "runtime/ze/ze_empty_event.hpp"

#include <vector>

using namespace cldnn;
using namespace ze_tests;

namespace {

std::shared_ptr<memory> allocate_device_memory(const ze_test_context& ctx, size_t bytes) {
    const layout mem_layout = {{static_cast<int64_t>(bytes)}, data_types::u8, format::bfyx};
    if (!ctx.ze_test_engine->supports_allocation(allocation_type::usm_device)) {
        return nullptr;
    }
    return ctx.ze_test_engine->allocate_memory(mem_layout, allocation_type::usm_device);
}

}  // namespace

TEST(ze_command_recorder, stream_provides_recorder) {
    auto ctx = create_ze_test_context();
    ASSERT_NE(ctx.ze_test_stream->get_recorder(), nullptr);
}

TEST(ze_command_recorder, created_command_list_is_open) {
    auto ctx = create_ze_test_context();
    auto cmd_list = ctx.ze_test_stream->get_recorder()->create_command_list();
    ASSERT_NE(cmd_list, nullptr);
    ASSERT_EQ(cmd_list->get_status(), command_list_status::open);
}

TEST(ze_command_recorder, recording_swaps_stream_command_list) {
    auto ctx = create_ze_test_context();
    auto& ze_test_stream = downcast<ze::ze_stream>(*ctx.ze_test_stream);
    auto recorder = ze_test_stream.get_recorder();

    auto immediate_handle = ze_test_stream.get_immediate_command_list().handle();
    // Without an active recording, the stream targets its immediate command list.
    ASSERT_EQ(ze_test_stream.get_command_list().handle(), immediate_handle);

    auto cmd_list = recorder->create_command_list();
    recorder->start_recording(cmd_list);

    auto recorded_handle = std::static_pointer_cast<ze::ze_command_list>(cmd_list)->resource().handle();
    // While recording, the stream must target the recorded command list.
    ASSERT_EQ(ze_test_stream.get_command_list().handle(), recorded_handle);
    ASSERT_NE(recorded_handle, immediate_handle);

    recorder->stop_recording();
    // After recording stops, the stream targets the immediate command list again.
    ASSERT_EQ(ze_test_stream.get_command_list().handle(), immediate_handle);
}

TEST(ze_command_recorder, recorded_command_list_replays_commands) {
    auto ctx = create_ze_test_context();
    auto& ze_test_stream = downcast<ze::ze_stream>(*ctx.ze_test_stream);

    constexpr size_t bytes = 256;
    auto mem = allocate_device_memory(ctx, bytes);
    if (mem == nullptr) {
        GTEST_SKIP() << "USM device allocation is not supported";
    }
    auto read_back = [&]() {
        std::vector<uint8_t> host(bytes, 0);
        mem->copy_to(ze_test_stream, host.data(), true);
        return host;
    };

    mem->fill(ze_test_stream, 0x00, {}, true);
    auto recorder = ze_test_stream.get_recorder();
    auto cmd_list = recorder->create_command_list();

    // Record a fill with a distinct pattern (non-blocking so recording is not interrupted).
    recorder->start_recording(cmd_list);
    mem->fill(ze_test_stream, 0xCD, {}, false);
    ASSERT_EQ(recorder->stop_recording(), cmd_list);
    ze_test_stream.finish();
    ASSERT_EQ(read_back(), std::vector<uint8_t>(bytes, 0xCD));

    mem->fill(ze_test_stream, 0x00, {}, true);
    ASSERT_EQ(read_back(), std::vector<uint8_t>(bytes, 0x00));

    // Replay recorded fill operation
    cmd_list->enqueue();
    cmd_list->wait();
    ASSERT_EQ(read_back(), std::vector<uint8_t>(bytes, 0xCD));
}

#endif  // OV_GPU_WITH_ZE_RT
