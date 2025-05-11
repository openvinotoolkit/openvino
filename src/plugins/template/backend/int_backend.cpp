// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "int_backend.hpp"

#include "int_executable.hpp"

ov::runtime::interpreter::INTBackend::INTBackend() {}

ov::runtime::interpreter::INTBackend::INTBackend(const std::vector<std::string>& unsupported_op_name_list)
    : m_unsupported_op_name_list{unsupported_op_name_list.begin(), unsupported_op_name_list.end()} {}

ov::Tensor ov::runtime::interpreter::INTBackend::create_tensor() {
    return ov::Tensor();
}

ov::Tensor ov::runtime::interpreter::INTBackend::create_tensor(const element::Type& type, const Shape& shape) {
    return ov::Tensor(type, shape);
}

ov::Tensor ov::runtime::interpreter::INTBackend::create_tensor(const element::Type& type,
                                                               const Shape& shape,
                                                               void* memory_pointer) {
    return ov::Tensor(type, shape, memory_pointer);
}

std::shared_ptr<ov::runtime::Executable> ov::runtime::interpreter::INTBackend::compile(
    std::shared_ptr<ov::Model> model) {
    return std::make_shared<INTExecutable>(model);
}
