// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend.hpp"

#include <sstream>

#include "int_backend.hpp"
#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"

ov::runtime::Backend::~Backend() = default;

std::shared_ptr<ov::runtime::Backend> ov::runtime::Backend::create() {
    auto inner_backend = std::make_shared<interpreter::INTBackend>();

    return inner_backend;
}
