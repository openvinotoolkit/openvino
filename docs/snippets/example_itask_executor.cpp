// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif
#include "openvino/runtime/threading/cpu_streams_executor.hpp"

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif

#include <future>
#include <iostream>
#include <memory>

void example1() {
// ! [itask_executor:define_pipeline]
    // std::promise is move only object so to satisfy copy callable constraint we use std::shared_ptr
    auto promise = std::make_shared<std::promise<void>>();
    // When the promise is created we can get std::future to wait the result
    auto future = promise->get_future();
    // Rather simple task
    ov::threading::Task task = [] {
        std::cout << "Some Output" << std::endl;
    };
    // Create an executor
    ov::threading::ITaskExecutor::Ptr taskExecutor =
        std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{});
    if (taskExecutor == nullptr) {
        // ProcessError(e);
        return;
    }
    // We capture the task and the promise. When the task is executed in the task executor context
    // we munually call std::promise::set_value() method
    taskExecutor->run([task, promise] {
        std::exception_ptr currentException;
        try {
            task();
        } catch(...) {
            // If there is some exceptions store the pointer to current exception
            currentException = std::current_exception();
        }

        if (nullptr == currentException) {
            promise->set_value();  //  <-- If there is no problems just call std::promise::set_value()
        } else {
            promise->set_exception(currentException);    //  <-- If there is an exception forward it to std::future object
        }
    });
    // To wait the task completion we call std::future::wait method
    future.wait();  //  The current thread will be blocked here and wait when std::promise::set_value()
                    //  or std::promise::set_exception() method will be called.

    // If the future store the exception it will be rethrown in std::future::get method
    try {
        future.get();
    } catch(std::exception& /*e*/) {
        // ProcessError(e);
    }
// ! [itask_executor:define_pipeline]
}
