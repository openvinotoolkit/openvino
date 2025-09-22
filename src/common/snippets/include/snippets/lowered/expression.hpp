// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <openvino/core/node.hpp>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/type.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/port_connector.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov::snippets::lowered {

class ExpressionFactory;
class LinearIR;
using ExpressionPtr = std::shared_ptr<Expression>;
using ExpressionMap = std::unordered_map<Expression*, ExpressionPtr>;
class Expression : public std::enable_shared_from_this<Expression> {
    friend class LinearIR;
    friend class ExpressionFactory;
    friend class ExpressionPort;

public:
    OPENVINO_RTTI_BASE("Expression")

    Expression() = default;
    virtual ~Expression() = default;

    std::shared_ptr<Node> get_node() const;
    std::shared_ptr<Emitter> get_emitter() const;

    RegInfo get_reg_info() const;
    void set_reg_info(const RegInfo& rinfo);
    const std::set<Reg>& get_live_regs() const {
        return m_live_regs;
    }
    void set_live_regs(std::set<Reg> live_regs) {
        m_live_regs = std::move(live_regs);
    }

    double get_exec_num() const {
        return m_exec_num;
    }

    const PortConnectorPtr& get_input_port_connector(size_t i) const;
    const PortConnectorPtr& get_output_port_connector(size_t i) const;
    const std::vector<PortConnectorPtr>& get_input_port_connectors() const {
        return m_input_port_connectors;
    }
    const std::vector<PortConnectorPtr>& get_output_port_connectors() const {
        return m_output_port_connectors;
    }

    const PortDescriptorPtr& get_input_port_descriptor(size_t i) const;
    const PortDescriptorPtr& get_output_port_descriptor(size_t i) const;
    const std::vector<PortDescriptorPtr>& get_input_port_descriptors() const {
        return m_input_port_descriptors;
    }
    const std::vector<PortDescriptorPtr>& get_output_port_descriptors() const {
        return m_output_port_descriptors;
    }

    ExpressionPtr get_input_expr_ptr(size_t i) const;

    size_t get_input_count() const {
        return m_input_port_connectors.size();
    }
    size_t get_output_count() const {
        return m_output_port_connectors.size();
    }

    void set_input_port_connector(size_t port, PortConnectorPtr to);

    // Attention! Cannot be called in ctor because this method validats port attributes (descs, connectors)
    virtual void validate() const;

    ExpressionPort get_input_port(size_t i);
    ExpressionPort get_output_port(size_t i);
    std::vector<ExpressionPort> get_input_ports();
    std::vector<ExpressionPort> get_output_ports();

    void updateShapes();
    bool needShapeInfer() const {
        return m_need_shape_infer;
    }
    const std::vector<size_t>& get_loop_ids() const;
    void set_loop_ids(const std::vector<size_t>& loops);

    /**
     * @brief Clone Expression with new node and input port attributes/
     *        Output port descriptors will be cloned from the current expression.
     *        Output port connecters will be created.
     * @param new_node new node
     * @param new_inputs new input port connectors
     * @param new_in_descs new input port descriptors. If this collection is empty,
     *                     descriptors will be copied from the current expression
     * @return the copy
     */
    ExpressionPtr clone_with_new_inputs(const std::shared_ptr<Node>& new_node,
                                        const std::vector<PortConnectorPtr>& new_inputs,
                                        const std::vector<PortDescriptorPtr>& new_in_descs = {}) const;
    /**
     * @brief Clone Expression with new node using `expr_map` to connect to new parent expressions.
     * @param expr_map the map with the original and cloned expressions
     * @param new_node new node
     * @return the copy
     */
    ExpressionPtr clone_with_new_inputs(const ExpressionMap& expr_map, const std::shared_ptr<Node>& new_node) const;

    virtual bool visit_attributes(AttributeVisitor& visitor);

    const char* get_type_name() const {
        return get_type_info().name;
    }

protected:
    // Note: The constructor initialization is private since an expression can be created only by Linear IR.
    //       The method must be used only by Linear IR builder of expressions!
    explicit Expression(const std::shared_ptr<Node>& n,
                        const std::shared_ptr<IShapeInferSnippetsFactory>& factory,
                        bool need_shape_infer = true);

    // Virtual clone method which is called in clone_with_new_inputs with common logic
    virtual ExpressionPtr clone() const;

    std::shared_ptr<Node> m_source_node{nullptr};
    std::shared_ptr<Emitter> m_emitter{nullptr};
    std::vector<PortConnectorPtr> m_input_port_connectors;
    std::vector<PortConnectorPtr> m_output_port_connectors;
    std::vector<PortDescriptorPtr> m_input_port_descriptors;
    std::vector<PortDescriptorPtr> m_output_port_descriptors;
    // The order Loops identifies: Outer ---> Inner
    // Note: The loops with the same dimension index (splitted dimension) should be successively nested
    std::vector<size_t> m_loop_ids;
    std::shared_ptr<IShapeInferSnippets> m_shapeInference{nullptr};
    const bool m_need_shape_infer = true;

    // The serial number of the execution in LinearIR
    // Attention!
    //   1. This number can be fractional to avoid frequent enumerations after each insertion to the Linear IR.
    //   2. This number can be changed and updated during whole pipeline, so its absolute values are meaningless.
    //   3. This number can be negative, positive and zero.
    double m_exec_num = 0;
    std::set<Reg> m_live_regs;
};

}  // namespace ov::snippets::lowered
