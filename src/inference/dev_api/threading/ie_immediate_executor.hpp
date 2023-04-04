// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_immediate_executor.hpp
 * @brief A header file for Inference Engine Immediate Executor implementation
 */

#pragma once

#include <memory>
#include <string>

#include "openvino/runtime/threading/immediate_executor.hpp"
#include "threading/ie_itask_executor.hpp"

namespace InferenceEngine {

/**
 * @brief Task executor implementation that just run tasks in current thread during calling of run() method
 * @ingroup ie_dev_api_threading
 */
class ImmediateExecutor : public ITaskExecutor {
public:
    /**
     * @brief A shared pointer to a ImmediateExecutor object
     */
    using Ptr = std::shared_ptr<ImmediateExecutor>;

    /**
     * @brief Destroys the object.
     */
    ~ImmediateExecutor() override = default;

    void run(Task task) override {
        task();
    }
};

}  // namespace InferenceEngine
