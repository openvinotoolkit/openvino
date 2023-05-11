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
    ov::PartialShape m_master_shape{};
    size_t m_loop_depth = 1;
};

class LinearIR {
    class ExpressionFactory;
public:
    using container = std::list<ExpressionPtr>;
    using io_container = std::list<std::shared_ptr<IOExpression>>;
    using exprIt = container::iterator;
    using constExprIt = container::const_iterator;

    LinearIR() = default;
    explicit LinearIR(const std::shared_ptr<ov::Model>& m, Config config = {});

    ExpressionPtr create_expression(const std::shared_ptr<Node>& n, const std::vector<TensorPtr>& inputs);

    static LinearIR::container deep_copy_range(LinearIR::container::const_iterator begin, LinearIR::container::const_iterator end);

    const container& get_ops() const {return m_lowered_ops; }
    const io_container& get_IO_ops() const {return m_io_lowered_ops; }
    Config get_config() {return m_config; }

    const ExpressionPtr& get_expr_by_node(const std::shared_ptr<Node>& n) const;

    void replace_input(const std::set<ExpressionPort>& consumers, const TensorPtr& to);
    void replace_input(const ExpressionPort& expr_port, const TensorPtr& to);

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

    class LoopManager;
    using LoopManagerPtr = std::shared_ptr<LoopManager>;

    const LoopManagerPtr& get_loop_manager() const { return m_loop_manager; }

private:
    static ov::NodeVector get_ordered_ops(const std::shared_ptr<ov::Model>& model);
    // Default ctor - can be called only from Linear IR initialization as default way
    ExpressionPtr create_expression(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model = nullptr);

    void register_expression(const ExpressionPtr& expr, bool io_allowed = false);
    void unregister_expression(const ExpressionPtr& expr);

    container m_lowered_ops{};
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Expression>> m_node2expression_map;
    io_container m_io_lowered_ops;
    Config m_config{};
    LoopManagerPtr m_loop_manager = nullptr;
};

} // namespace lowered
} // namespace snippets
} // namespace ngraph
