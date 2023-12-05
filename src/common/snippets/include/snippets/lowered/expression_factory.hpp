// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_ir.hpp"

#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace lowered {

class LinearIR::ExpressionFactory {
public:
    template<class... Args>
    static ExpressionPtr build(const std::shared_ptr<Node>& n, Args&&... params) {
        if (const auto par = ov::as_type_ptr<ov::op::v0::Parameter>(n)) {
            return create(par, params...);
        } else if (const auto res = ov::as_type_ptr<ov::op::v0::Result>(n)) {
            return create(res, params...);
        } else if (const auto loop_begin = ov::as_type_ptr<op::LoopBegin>(n)) {
            return create(loop_begin, params...);
        } else if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(n)) {
            return create(loop_end, params...);
        }
        return create(n, params...);
    }
    template<class ExprType, typename std::enable_if<std::is_base_of<Expression, ExprType>::value, bool>::type = true>
    static ExpressionPtr shallow_copy(const std::shared_ptr<ExprType>& expr) {
        if (const auto& io_expr = std::dynamic_pointer_cast<IOExpression>(expr))
            return std::make_shared<IOExpression>(*io_expr);
        else
            return std::make_shared<ExprType>(*expr);
    }

private:
    /* -- Default Builders - initialize input port connectors from parents and create new output port connectors themselves */
    static ExpressionPtr create(const std::shared_ptr<ov::op::v0::Parameter>& par, const LinearIR& linear_ir,
                                const std::shared_ptr<ov::Model>& model);
    static ExpressionPtr create(const std::shared_ptr<ov::op::v0::Result>& res, const LinearIR& linear_ir,
                                const std::shared_ptr<ov::Model>& model);
    static ExpressionPtr create(const std::shared_ptr<ov::Node>& n, const LinearIR& linear_ir,
                                const std::shared_ptr<ov::Model>& model);

    /* -- Input Builders - get input port connectors from method parameters and create new output port connectors themselves */
    static ExpressionPtr create(const std::shared_ptr<op::LoopBegin>& n, const std::vector<PortConnectorPtr>& inputs, const LinearIR& linear_ir);
    static ExpressionPtr create(const std::shared_ptr<op::LoopEnd>& n, const std::vector<PortConnectorPtr>& inputs, const LinearIR& linear_ir);
    static ExpressionPtr create(const std::shared_ptr<ov::Node>& n, const std::vector<PortConnectorPtr>& inputs, const LinearIR& linear_ir);

    // Creates inputs for expression using parent output port connectors
    static void create_expression_inputs(const LinearIR& linear_ir, const ExpressionPtr& expr);
    // Creates new output port connectors
    static void create_expression_outputs(const ExpressionPtr& expr);
    // The method verifies of input port connectors to availability of the expression as consumer and add it if missed
    static void init_expression_inputs(const ExpressionPtr& expr, const std::vector<PortConnectorPtr>& inputs);
};

} // namespace lowered
} // namespace snippets
} // namespace ov
