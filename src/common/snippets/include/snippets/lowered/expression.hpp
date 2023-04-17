// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>

#include "snippets/tensor_descriptor.hpp"
#include "snippets/emitter.hpp"
#include "snippets/target_machine.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

class LinearIR;
class Expression;
using ExpressionPtr = std::shared_ptr<Expression>;

class ExpressionPort {
    friend class Expression;

public:
    enum Type {
        Input,
        Output
    };

    ExpressionPort() = default;

    Type get_type() const { return m_type; }

    ExpressionPtr expr = nullptr;
    size_t port = 0;

private:
    ExpressionPort(const ExpressionPtr& expr, size_t port, Type type);

    Type m_type = Type::Input;
};

class Expression : public std::enable_shared_from_this<Expression> {
    friend class LinearIR;

public:
    static size_t LOOP_NULL_ID;

    Expression() = default;
    explicit Expression(const std::shared_ptr<Node>& n);
    // The ctor fills outputs automatically from rt_info and/or tensor shapes
    explicit Expression(const std::shared_ptr<Node>& n, std::vector<TensorDescriptorPtr> inputs);
    explicit Expression(const std::shared_ptr<Node>& n, std::vector<TensorDescriptorPtr> inputs, std::vector<TensorDescriptorPtr> outputs);

    virtual ~Expression() = default;

    std::shared_ptr<Node> get_node() const;
    std::shared_ptr<Emitter> get_emitter() const;

    RegInfo get_reg_info() const { return  m_reg_info; }
    void set_reg_info(RegInfo rinfo) { m_reg_info = std::move(rinfo); }

    const std::vector<TensorDescriptorPtr>& get_inputs() { return m_inputs; }
    const std::vector<TensorDescriptorPtr>& get_outputs() { return m_outputs; }

    std::vector<size_t> get_loop_ids() const { return m_loop_ids; }
    void set_loop_ids(const std::vector<size_t>& loops) { m_loop_ids = loops; }
    void set_loop_id(size_t id, size_t idx);
    void remove_loop_id(size_t id);
    bool is_outside_loop() const { return m_is_outside_loop; }

    void init_emitter(const std::shared_ptr<const TargetMachine>& target);

    ExpressionPort input_port(size_t i);
    ExpressionPort output_port(size_t i);

protected:
    void replace_input(size_t port, TensorDescriptorPtr to);
    void replace_output(size_t port, TensorDescriptorPtr to);

    std::shared_ptr<Node> m_source_node{nullptr};
    std::shared_ptr<Emitter> m_emitter{nullptr};
    std::vector<TensorDescriptorPtr> m_inputs;
    std::vector<TensorDescriptorPtr> m_outputs;
    RegInfo m_reg_info{{}, {}};
    // The order Loops identifies: Outer ---> Inner
    std::vector<size_t> m_loop_ids;
    bool m_is_outside_loop = false;
};

class IOExpression : public Expression {
public:
    enum class io_type {INPUT, OUTPUT, UNDEFINED};

    IOExpression(const std::shared_ptr<ov::opset1::Parameter>& n, int64_t index);
    IOExpression(const std::shared_ptr<ov::opset1::Result>& n, int64_t index, std::vector<TensorDescriptorPtr> inputs);

    int64_t get_index() const  { return m_index; }
    io_type get_type() const { return m_type; }

private:
    int64_t m_index = -1;
    io_type m_type = io_type::UNDEFINED;
};

bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs);
bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs);
bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs);

} // namespace lowered
} // namespace snippets
} // namespace ngraph
