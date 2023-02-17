// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/immediate_executor.hpp"

ov::ImmediateExecutor::~ImmediateExecutor() = default;

void ov::ImmediateExecutor::run(Task task) {
    task();
}
