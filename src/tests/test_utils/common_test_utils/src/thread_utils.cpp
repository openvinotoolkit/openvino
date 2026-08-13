// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/thread_utils.hpp"

#include <exception>
#include <stdexcept>

#ifdef _WIN32
#    include <process.h>
#    include <windows.h>
#else
#    include <limits.h>
#    include <pthread.h>
#    include <unistd.h>
#endif

namespace ov {
namespace test {
namespace utils {

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
// _beginthreadex (not CreateThread) so the CRT state of the thread is set up and cleaned up.
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

std::string run_with_stack_size(size_t stack_bytes, const std::function<void()>& body) {
    ThreadContext context;
    context.body = &body;

#ifdef _WIN32
    const auto thread = reinterpret_cast<HANDLE>(_beginthreadex(nullptr,
                                                                static_cast<unsigned>(stack_bytes),
                                                                thread_entry,
                                                                &context,
                                                                STACK_SIZE_PARAM_IS_A_RESERVATION,
                                                                nullptr));
    if (thread == nullptr) {
        throw std::runtime_error("run_with_stack_size: failed to create a thread");
    }
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
        throw std::runtime_error("run_with_stack_size: pthread_attr_init failed");
    }
    if (stack_bytes < static_cast<size_t>(PTHREAD_STACK_MIN)) {
        stack_bytes = static_cast<size_t>(PTHREAD_STACK_MIN);
    }
    // macOS requires a multiple of the page size.
    const auto page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    stack_bytes = (stack_bytes + page_size - 1) / page_size * page_size;
    if (pthread_attr_setstacksize(&attr, stack_bytes) != 0) {
        pthread_attr_destroy(&attr);
        throw std::runtime_error("run_with_stack_size: pthread_attr_setstacksize failed");
    }
    pthread_t thread;
    const int status = pthread_create(&thread, &attr, thread_entry, &context);
    pthread_attr_destroy(&attr);
    if (status != 0) {
        throw std::runtime_error("run_with_stack_size: pthread_create failed");
    }
    pthread_join(thread, nullptr);
#endif

    if (context.error) {
        try {
            std::rethrow_exception(context.error);
        } catch (const std::exception& e) {
            return e.what();
        } catch (...) {
            return "unknown exception";
        }
    }
    return {};
}

}  // namespace utils
}  // namespace test
}  // namespace ov
