// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "backend.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph {
namespace runtime {
namespace interpreter {
class INTBackend;
class INTExecutable;
}  // namespace interpreter
}  // namespace runtime
}  // namespace ngraph

class ngraph::runtime::interpreter::INTBackend : public Backend {
public:
    INTBackend();
    INTBackend(const std::vector<std::string>& unsupported_op_name_list);
    INTBackend(const INTBackend&) = delete;
    INTBackend(INTBackend&&) = delete;
    INTBackend& operator=(const INTBackend&) = delete;

    std::shared_ptr<Tensor> create_tensor() override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape) override;
    std::shared_ptr<Tensor> create_dynamic_tensor(const element::Type& type, const PartialShape& shape) override;

    std::shared_ptr<Executable> compile(std::shared_ptr<Function> function,
                                        bool enable_performance_data = false) override;

    bool is_supported(const Node& node) const override;

    bool set_config(const std::map<std::string, std::string>& config, std::string& error) override;

private:
    std::set<std::string> m_unsupported_op_name_list;
};
