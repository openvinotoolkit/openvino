// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "cpp_interfaces/ie_itask_executor.hpp"

namespace InferenceEngine {
/**
 * @class ImmediateExecutor
 * @brief Task executor implementation that just run tasks in current thread during calling of
 * run() method
 */
class INFERENCE_ENGINE_API_CLASS(ImmediateExecutor): public ITaskExecutor {
public:
    using Ptr = std::shared_ptr<ImmediateExecutor>;

    ~ImmediateExecutor() override {};

    void run(Task task) override {
        task();
    }
};

}  // namespace InferenceEngine
