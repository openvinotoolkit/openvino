// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/thread_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <exception>

#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/stack_util.hpp"

#ifdef _WIN32
#    include <process.h>
#    include <windows.h>
#else
#    include <limits.h>
#    include <pthread.h>
#endif

namespace ov::test::utils {

namespace {

struct ThreadContext {
    const std::function<void()>* body = nullptr;
    std::exception_ptr error;
};

void run_body(ThreadContext& context) {
    try {
        (*context.body)();
    } catch (...) {
        context.error = std::current_exception();
    }
}

#ifdef _WIN32
// _beginthreadex (not CreateThread) to set up and clean up the CRT state.
unsigned __stdcall thread_entry(void* arg) {
    run_body(*static_cast<ThreadContext*>(arg));
    return 0;
}
#else
void* thread_entry(void* arg) {
    run_body(*static_cast<ThreadContext*>(arg));
    return nullptr;
}
#endif

}  // namespace

void run_with_stack_size(size_t stack_bytes, const std::function<void()>& body) {
    ThreadContext context;
    context.body = &body;
    stack_bytes = ov::util::align_size_up(stack_bytes, static_cast<size_t>(ov::util::get_system_page_size()));

#ifdef _WIN32
    const auto thread = reinterpret_cast<HANDLE>(_beginthreadex(nullptr,
                                                                static_cast<unsigned>(stack_bytes),
                                                                thread_entry,
                                                                &context,
                                                                STACK_SIZE_PARAM_IS_A_RESERVATION,
                                                                nullptr));
    if (thread == nullptr) {
        ADD_FAILURE() << "Failed to create a thread";
        return;
    }
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
        ADD_FAILURE() << "pthread_attr_init failed";
        return;
    }
    stack_bytes = std::max(stack_bytes, static_cast<size_t>(PTHREAD_STACK_MIN));
    pthread_t thread;
    if (pthread_attr_setstacksize(&attr, stack_bytes) != 0 ||
        pthread_create(&thread, &attr, thread_entry, &context) != 0) {
        pthread_attr_destroy(&attr);
        ADD_FAILURE() << "Failed to create a thread with a " << stack_bytes << " bytes stack";
        return;
    }
    pthread_attr_destroy(&attr);
    pthread_join(thread, nullptr);
#endif

    if (context.error) {
        std::rethrow_exception(context.error);
    }
}

bool is_stack_size_observable(size_t stack_bytes) {
    bool observable = false;
    run_with_stack_size(stack_bytes, [&observable] {
        observable = ov::util::stack_bytes_available().has_value();
    });
    return observable;
}

}  // namespace ov::test::utils
