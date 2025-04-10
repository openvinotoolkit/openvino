// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file openvino/runtime/threading/immediate_executor.hpp
 * @brief A header file for OpenVINO Immediate Executor implementation
 */

#pragma once

#include <memory>
#include <string>

#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {
namespace threading {

/**
 * @brief Task executor implementation that just run tasks in current thread during calling of run() method
 * @ingroup ov_dev_api_threading
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

}  // namespace threading
}  // namespace ov
