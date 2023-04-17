// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include "expression.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {

class Config {
public:
    // True if the lowered Emitters need to be accessed during runtime. Normally they're destroyed after code emission.
    bool m_save_lowered_code = false;
    // True if we should check runtime info for nodes to call specific needed transformations
    bool m_need_fill_tail_register = false;
    bool m_explicit_loop_insertion = false;
    ov::PartialShape m_master_shape{};
    size_t m_loop_depth = 1;
};

class LinearIR {
public:
    using container = std::list<ExpressionPtr>;
    using io_container = std::list<std::shared_ptr<IOExpression>>;
    using exprIt = container::iterator;
    using constExprIt = container::const_iterator;

    LinearIR() = default;
    explicit LinearIR(const std::shared_ptr<ov::Model>& m, Config config = {});

    LinearIR deep_copy() const;
    static LinearIR::container deep_copy_range(LinearIR::container::const_iterator begin, LinearIR::container::const_iterator end);

    const container& get_ops() const {return m_lowered_ops; }
    const io_container& get_IO_ops() const {return m_io_lowered_ops; }
    Config get_config() {return m_config; }

    ExpressionPtr get_expr_by_node(const std::shared_ptr<Node>& n) const;
    ExpressionPort get_expr_by_output(const TensorDescriptorPtr& n) const;
    const std::set<ExpressionPort>& get_exprs_by_input(const TensorDescriptorPtr& n) const;

    void replace_input(const ExpressionPort& expr_port, const TensorDescriptorPtr& to);
    void replace_input(const ExpressionPtr& expr, size_t port, const TensorDescriptorPtr& to);
    void replace_output(const ExpressionPort& expr_port, const TensorDescriptorPtr& to);
    void replace_output(const ExpressionPtr& expr, size_t port, const TensorDescriptorPtr& to);

    /**
    * @brief Move an expression from the position "from" to the position immediately before "to".
     * Note: this method does NOT take care about data dependencies and no relevant checks are performed.
     *       and doesn't touch internal maps.
    */
    void move(constExprIt from, constExprIt to);

    bool empty() const noexcept {return m_lowered_ops.empty(); }
    void debug_print(bool tds_as_pointers = false) const;

    container::reference back() noexcept {return m_lowered_ops.back();}
    container::const_reference back() const noexcept {return m_lowered_ops.back();}
    container::reference front() noexcept {return m_lowered_ops.front();}
    container::const_reference front() const noexcept {return m_lowered_ops.front();}

    exprIt begin() noexcept {return m_lowered_ops.begin();}
    exprIt end() noexcept {return m_lowered_ops.end();}
    constExprIt begin() const noexcept {return cbegin();}
    constExprIt end() const noexcept {return cend();}
    constExprIt cbegin() const noexcept {return m_lowered_ops.cbegin();}
    constExprIt cend() const noexcept {return m_lowered_ops.cend();}
    container::reverse_iterator rbegin() noexcept {return m_lowered_ops.rbegin();}
    container::reverse_iterator rend() noexcept {return m_lowered_ops.rend();}
    container::const_reverse_iterator crbegin() const noexcept {return m_lowered_ops.crbegin();}
    container::const_reverse_iterator crend() const noexcept {return m_lowered_ops.crend();}

    exprIt insert(constExprIt pos, const ov::NodeVector& nodes);
    exprIt insert(constExprIt pos, const std::shared_ptr<Node>& n);
    exprIt insert(constExprIt pos, container::value_type&& value);
    exprIt insert(constExprIt pos, const container::value_type& value);
    exprIt insert(constExprIt pos, exprIt begin, exprIt end);
    exprIt insert(constExprIt pos, constExprIt begin, constExprIt end);

    exprIt erase(exprIt pos);
    exprIt erase(constExprIt pos);

    void init_emitters(const std::shared_ptr<TargetMachine>& target);
    void serialize(const std::string& xml, const std::string& bin);

    static ov::NodeVector get_ordered_ops(const std::shared_ptr<ov::Model>& model);

    class LoopManager;
    using LoopManagerPtr = std::shared_ptr<LoopManager>;

    const LoopManagerPtr& get_loop_manager() const { return m_loop_manager; }

private:
    void register_expression(const ExpressionPtr& expr);
    // Like register_expression, but doesn't allow Parameter or Result registration. You can do it only through ctor
    void register_regular_expression(const ExpressionPtr& expr);
    void unregister_expression(const ExpressionPtr& expr);

    container m_lowered_ops{};
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Expression>> m_node2expression_map;
    // Expression must be uniquely identified by an output, so there can't be expressions that have the same output
    std::unordered_map<TensorDescriptorPtr, ExpressionPort> m_output2expression_map;
    // At the same time, several expressions can have the same input if they are connected to the same parent
    // E.g. LoopEnd will always have the same input as a Load inside the loop (since it has to increment the same reg)
    std::unordered_map<TensorDescriptorPtr, std::set<ExpressionPort>> m_input2expression_map;
    io_container m_io_lowered_ops;
    Config m_config{};
    LoopManagerPtr m_loop_manager = nullptr;
};

} // namespace lowered
} // namespace snippets
} // namespace ngraph
