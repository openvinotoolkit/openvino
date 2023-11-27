// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>

#include "snippets/emitter.hpp"
#include "snippets/lowered/port_connector.hpp"
#include "snippets/lowered/expression_port.hpp"

#include "snippets/shape_inference/shape_inference.hpp"

namespace ov {
namespace snippets {
namespace lowered {

class LinearIR;
using ExpressionPtr = std::shared_ptr<Expression>;
using ExressionMap = std::unordered_map<Expression*, ExpressionPtr>;
class Expression : public std::enable_shared_from_this<Expression> {
    friend class LinearIR;
    friend class ExpressionPort;

public:
    Expression() = default;
    virtual ~Expression() = default;

    std::shared_ptr<Node> get_node() const;
    std::shared_ptr<Emitter> get_emitter() const;

    RegInfo get_reg_info() const;
    void set_reg_info(RegInfo rinfo);

    const PortConnectorPtr& get_input_port_connector(size_t i) const;
    const PortConnectorPtr& get_output_port_connector(size_t i) const;
    std::vector<PortConnectorPtr> get_input_port_connectors() const { return m_input_port_connectors; }
    std::vector<PortConnectorPtr> get_output_port_connectors() const { return m_output_port_connectors; }

    const PortDescriptorPtr& get_input_port_descriptor(size_t i) const;
    const PortDescriptorPtr& get_output_port_descriptor(size_t i) const;
    std::vector<PortDescriptorPtr> get_input_port_descriptors() const { return m_input_port_descriptors; }
    std::vector<PortDescriptorPtr> get_output_port_descriptors() const { return m_output_port_descriptors; }

    size_t get_input_count() const { return m_input_port_connectors.size(); }
    size_t get_output_count() const { return m_output_port_connectors.size(); }

    void validate() const;

    ExpressionPort get_input_port(size_t i);
    ExpressionPort get_output_port(size_t i);
    void updateShapes();
    virtual bool needShapeInfer() const {return true; }

    std::vector<size_t> get_loop_ids() const;
    void set_loop_ids(const std::vector<size_t>& loops);
    virtual ExpressionPtr clone_with_new_inputs(const std::vector<PortConnectorPtr>& new_inputs,
                                                const std::shared_ptr<Node>& new_node) const;
    ExpressionPtr clone_with_new_inputs(const ExressionMap& expr_map, const std::shared_ptr<Node>& new_node) const;

protected:
    Expression(const Expression& other);
    // Note: The constructor initialization is private since an expression can be created only by Linear IR.
    //       The method must be used only by Linear IR builder of expressions!
    Expression(const std::shared_ptr<Node>& n, const std::shared_ptr<IShapeInferSnippetsFactory>& factory);
    void update_node_and_connectors(const std::vector<PortConnectorPtr>& new_inputs, const std::shared_ptr<Node>& new_node);

    void replace_input(size_t port, PortConnectorPtr to);

    std::shared_ptr<Node> m_source_node{nullptr};
    std::shared_ptr<Emitter> m_emitter{nullptr};
    std::vector<PortConnectorPtr> m_input_port_connectors{};
    std::vector<PortConnectorPtr> m_output_port_connectors{};
    std::vector<PortDescriptorPtr> m_input_port_descriptors{};
    std::vector<PortDescriptorPtr> m_output_port_descriptors{};
    // The order Loops identifies: Outer ---> Inner
    // Note: The loops with the same dimension index (splitted dimension) should be successively nested
    std::vector<size_t> m_loop_ids{};
    std::shared_ptr<IShapeInferSnippets> m_shapeInference{nullptr};
};

class IOExpression : public Expression {
    friend class LinearIR;

public:
    enum class io_type {INPUT, OUTPUT, UNDEFINED};
    ExpressionPtr clone_with_new_inputs(const std::vector<PortConnectorPtr>& new_inputs,
                                        const std::shared_ptr<Node>& new_node) const override;
    int64_t get_index() const  { return m_index; }
    io_type get_type() const { return m_type; }
    // Result needs shapeInfer to copy shape from Parent's output to this expr input
    bool needShapeInfer() const override {return m_type == io_type::OUTPUT; }
private:
    IOExpression(const IOExpression& other) = default;
    explicit IOExpression(const std::shared_ptr<ov::opset1::Parameter>& n, int64_t index, const std::shared_ptr<IShapeInferSnippetsFactory>& factory);
    explicit IOExpression(const std::shared_ptr<ov::opset1::Result>& n, int64_t index, const std::shared_ptr<IShapeInferSnippetsFactory>& factory);

    int64_t m_index = -1;
    io_type m_type = io_type::UNDEFINED;
};

} // namespace lowered
} // namespace snippets
} // namespace ov
