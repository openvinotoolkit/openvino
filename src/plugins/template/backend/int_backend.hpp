// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "backend.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace runtime {
namespace interpreter {

class INTBackend : public Backend {
public:
    INTBackend();
    INTBackend(const std::vector<std::string>& unsupported_op_name_list);
    INTBackend(const INTBackend&) = delete;
    INTBackend(INTBackend&&) = delete;
    INTBackend& operator=(const INTBackend&) = delete;

    ov::Tensor create_tensor() override;

    ov::Tensor create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    ov::Tensor create_tensor(const element::Type& type, const Shape& shape) override;

    std::shared_ptr<Executable> compile(std::shared_ptr<ov::Model> model) override;

private:
    std::set<std::string> m_unsupported_op_name_list;
};

}  // namespace interpreter
}  // namespace runtime
}  // namespace ov
