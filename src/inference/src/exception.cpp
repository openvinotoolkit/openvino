// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/exception.hpp"

void ov::Cancelled::create(const std::string& explanation) {
    throw ov::Cancelled(explanation);
}

ov::Cancelled::~Cancelled() = default;

void ov::Busy::create(const std::string& explanation) {
    throw ov::Busy(explanation);
}

ov::Busy::~Busy() = default;
