// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/function.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/rt_info.hpp>

#include "snippets/generator.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Subgraph
 * @brief An operation that is implemented by a function
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Subgraph : public ngraph::op::Op {
public:
    using BlockedShape = std::tuple<ngraph::Shape, ngraph::AxisVector, ngraph::element::Type>;
    using BlockedShapeVector = std::vector<BlockedShape>;

    NGRAPH_RTTI_DECLARATION;

    Subgraph(const OutputVector& args, std::shared_ptr<Function> body);

    Subgraph(const NodeVector& args, std::shared_ptr<Function> body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::shared_ptr<Function> get_body() const {
        return m_body;
    }

    std::shared_ptr<ngraph::snippets::Generator> get_generator() const {
        return m_generator;
    }

    std::shared_ptr<Subgraph> make_canonical_from_this();

    snippets::Schedule generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes);
    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;

    /// Set a new body for the op; body needs to satisfy requirements on inputs/outputs
    void set_body(std::shared_ptr<Function> body);

    // plugin sets generator for a snippet to some specific generator.
    // it's going to be replaced with Jitters table later
    void set_generator(std::shared_ptr<ngraph::snippets::Generator> generator);

    void print() const;
    void print_statistics(bool verbose);

    static auto wrap_node_as_subgraph(const std::shared_ptr<ngraph::Node>& node) -> std::shared_ptr<Subgraph>;

private:
    void canonicalize(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes);
    void convert_to_snippet_dialect();

    std::shared_ptr<Function> m_body;
    std::shared_ptr<ngraph::snippets::Generator> m_generator;
};

static inline std::ostream& operator<<(std::ostream& os, const op::Subgraph::BlockedShape& blocked_shape) {
    os << std::get<0>(blocked_shape) << " " << std::get<1>(blocked_shape) << " " << std::get<2>(blocked_shape);
    return os;
}

static inline auto is_scalar_constant(const std::shared_ptr<ngraph::Node>& source_output_node) -> bool {
    return !!ngraph::as_type_ptr<ngraph::opset1::Constant>(source_output_node) &&
        (source_output_node->get_shape() == ngraph::Shape() || ngraph::shape_size(source_output_node->get_shape()) == 1);
};

static inline auto create_body(std::string name, const ngraph::ResultVector& results, const ngraph::ParameterVector& parameters) ->
    std::shared_ptr<ngraph::Function> {
    auto body = std::make_shared<ngraph::Function>(results, parameters, name);
    return body;
};

static inline auto build_subgraph(const std::shared_ptr<ngraph::Node>& node, const ngraph::OutputVector& inputs, const std::shared_ptr<ngraph::Function>& body)
    -> std::shared_ptr<Subgraph>{
    auto subgraph = std::make_shared<Subgraph>(inputs, body);
    copy_runtime_info(node, subgraph);
    subgraph->set_friendly_name(node->get_friendly_name());
    return subgraph;
};

}  // namespace op
}  // namespace snippets
}  // namespace ngraph
