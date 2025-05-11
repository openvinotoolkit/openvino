// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "expression.hpp"
#include "expressions/buffer_expression.hpp"

#include "snippets/op/loop.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/perf_count.hpp"
#include "snippets/op/reg_spill.hpp"

namespace ov {
namespace snippets {
namespace lowered {

class ExpressionFactory {
public:
    ExpressionFactory(std::shared_ptr<IShapeInferSnippetsFactory> shape_infer_factory)
        : m_shape_infer_factory(std::move(shape_infer_factory)) {}

    template <typename T = Expression, typename... Args,
              typename std::enable_if<std::is_base_of<Expression, T>::value, bool>::type = true>
    std::shared_ptr<T> build(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& inputs, Args... args) {
        return create<T>(n, inputs, m_shape_infer_factory, args...);
    }

private:
    static ExpressionPtr create(const std::shared_ptr<ov::op::v0::Parameter>& par, const std::vector<PortConnectorPtr>& inputs,
                                const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);
    static ExpressionPtr create(const std::shared_ptr<ov::op::v0::Result>& res, const std::vector<PortConnectorPtr>& inputs,
                                const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);
    static ExpressionPtr create(const std::shared_ptr<op::LoopBegin>& n, const std::vector<PortConnectorPtr>& inputs,
                                const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);
    static ExpressionPtr create(const std::shared_ptr<op::LoopEnd>& n, const std::vector<PortConnectorPtr>& inputs,
                                const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);
    static ExpressionPtr create(const std::shared_ptr<op::RegSpillBegin>& n, const std::vector<PortConnectorPtr>& inputs,
                                const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);
    static ExpressionPtr create(const std::shared_ptr<op::RegSpillEnd>& n, const std::vector<PortConnectorPtr>& inputs,
                                const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);

    // Note: PerfCountBegin nodes have a PerfCountEnd ov::Output, but corresponding expression should not have any outputs to avoid register allocation
#ifdef SNIPPETS_DEBUG_CAPS
    static ExpressionPtr create(const std::shared_ptr<op::PerfCountBeginBase>& n, const std::vector<PortConnectorPtr>& inputs,
                                const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);
    static ExpressionPtr create(const std::shared_ptr<op::PerfCountEndBase>& n, const std::vector<PortConnectorPtr>& inputs,
                                const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);
    static ExpressionPtr create_without_connections(const std::shared_ptr<ov::Node>& n, const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory);
#endif

    template <typename T = Expression, typename... Args,
              typename std::enable_if<std::is_base_of<Expression, T>::value, bool>::type = true>
    static std::shared_ptr<T> create(const std::shared_ptr<ov::Node>& n, const std::vector<PortConnectorPtr>& inputs,
                                     const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory, Args... args) {
        auto expr = std::shared_ptr<T>(new T(n, shape_infer_factory, args...));
        init_expression_inputs(expr, inputs);
        create_expression_outputs(expr);
        expr->validate();
        // todo: here we blindly synchronize input shapes from parent and child. Remove this when shapes will be stored in port connector itself
        if (shape_infer_factory)
            expr->updateShapes();
        return expr;
    }

    // Creates new output port connectors
    static void create_expression_outputs(const ExpressionPtr& expr);
    // The method verifies of input port connectors to availability of the expression as consumer and add it if missed
    static void init_expression_inputs(const ExpressionPtr& expr, const std::vector<PortConnectorPtr>& inputs);

    const std::shared_ptr<IShapeInferSnippetsFactory> m_shape_infer_factory = nullptr;
};
using ExpressionFactoryPtr = std::shared_ptr<ExpressionFactory>;

template<>
std::shared_ptr<Expression> ExpressionFactory::build(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& inputs);

} // namespace lowered
} // namespace snippets
} // namespace ov
