// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_immediate_executor.hpp
 * @brief A header file for Inference Engine Immediate Executor implementation
 */

#pragma once

#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {

/**
 * @brief Task executor implementation that just run tasks in current thread during calling of run() method
 * @ingroup ov_dev_api_threading
 */
class OPENVINO_API ImmediateExecutor : public ITaskExecutor {
public:
    /**
     * @brief Destroys the object.
     */
    ~ImmediateExecutor() override;

    void run(Task task) override;
};

}  // namespace ov
