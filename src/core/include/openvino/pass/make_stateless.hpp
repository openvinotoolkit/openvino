// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief The transformation replaces the state operations ReadValue and Assign with Parameter
 * and Result operations, making the model stateless.
 *
 * The inverse of the MakeStateful transformation: every ReadValue operation of a selected
 * variable is replaced by a Parameter (named after the variable id), the input value of the
 * corresponding Assign operation is exposed as a Result, and the variable together with its
 * Assign sink is removed from the model. After the transformation the model has no variables,
 * and its shapes can be changed with the regular Model::reshape.
 *
 * A variable with several Assign operations cannot be represented without explicit state merge
 * logic, the transformation throws in that case.
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API MakeStateless : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MakeStateless");

    /**
     * @brief Constructs a MakeStateless transformation converting every variable of the model.
     */
    MakeStateless() = default;

    /**
     * @brief Constructs a MakeStateless transformation converting only the variables with the
     * given ids.
     *
     * @param variable_ids Identifiers of the variables to convert.
     */
    explicit MakeStateless(const std::unordered_set<std::string>& variable_ids)
        : m_variable_ids(variable_ids) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool selected(const std::string& variable_id) const;

    std::unordered_set<std::string> m_variable_ids;
};
}  // namespace pass
}  // namespace ov
