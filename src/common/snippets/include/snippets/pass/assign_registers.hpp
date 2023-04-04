// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/pass.hpp>

#include "snippets/generator.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface AssignRegisters
 * @brief Assigns internal `vector` register indexes to operations.
 * Changing order of variables or datafrow lead to invalidation of register assignment.
 * @ingroup snippets
 */
class AssignRegisters : public ngraph::pass::FunctionPass {
public:
    explicit AssignRegisters(const std::function<Generator::opRegType(const std::shared_ptr<Node>& op)>& mapper) : m_reg_type_mapper(mapper) {
        set_property(ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    std::function<Generator::opRegType(const std::shared_ptr<Node>& op)> m_reg_type_mapper;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
