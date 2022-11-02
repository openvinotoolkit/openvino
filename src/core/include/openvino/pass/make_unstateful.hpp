// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ngraph/opsets/opset9.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief The transformation replaces the provided list of Variables by pairs Parameter and Result
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API MakeUnStateful : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MakeUnStateful");

    using VariablesMap = std::map<std::shared_ptr<ngraph::Variable>, std::pair<std::string, std::string>>;
    using VariableNamesMap = std::map<std::string, std::pair<std::string, std::string>>;

    explicit MakeUnStateful(const VariablesMap& variable_map) : m_variable_map(variable_map) {}
    explicit MakeUnStateful(const VariableNamesMap& variable_names_map) : m_variable_names_map(variable_names_map) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    VariablesMap m_variable_map;
    VariableNamesMap m_variable_names_map;
};
}  // namespace pass
}  // namespace ov
