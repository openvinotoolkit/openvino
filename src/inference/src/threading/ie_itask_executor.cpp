// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_itask_executor.hpp"

#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace InferenceEngine {

void ITaskExecutor::runAndWait(const std::vector<Task>& tasks) {
    run_and_wait(tasks);
}

}  // namespace InferenceEngine
