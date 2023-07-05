// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include "expression.hpp"

namespace ov {
namespace snippets {
namespace lowered {

class Config {
public:
    // True if the lowered Emitters need to be accessed during runtime. Normally they're destroyed after code emission.
    bool m_save_expressions = false;
    // True if we should check runtime info for nodes to call specific needed transformations
    bool m_need_fill_tail_register = false;
    size_t m_loop_depth = 1;
};

/* The control flow of Snippets is built on Linear Intermediate Representation (Linear IR).
 * The class diagram is described in the documentation `snippets/docs/snippets_design_guide.md`.
 */
class LinearIR {
    class ExpressionFactory;
public:
    using container = std::list<ExpressionPtr>;
    using io_container = std::list<std::shared_ptr<IOExpression>>;
    using exprIt = container::iterator;
    using constExprIt = container::const_iterator;
    using exprReverseIt = container::reverse_iterator;
    using constExprReverseIt = container::const_reverse_iterator;

    LinearIR() = default;
    explicit LinearIR(const std::shared_ptr<ov::Model>& m, Config config = {});

    ExpressionPtr create_expression(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& inputs);

    static LinearIR::container deep_copy_range(LinearIR::container::const_iterator begin, LinearIR::container::const_iterator end);

    const container& get_ops() const {return m_expressions; }
    const io_container& get_IO_ops() const {return m_io_expressions; }
    Config get_config() {return m_config; }

    const ExpressionPtr& get_expr_by_node(const std::shared_ptr<Node>& n) const;

    void replace_input(const std::set<ExpressionPort>& consumers, const PortConnectorPtr& to);
    void replace_input(const ExpressionPort& expr_port, const PortConnectorPtr& to);

    /**
    * @brief Move an expression from the position "from" to the position immediately before "to".
     * Note: this method does NOT take care about data dependencies and no relevant checks are performed.
     *       and doesn't touch internal maps.
    */
    void move(constExprIt from, constExprIt to);

    bool empty() const noexcept {return m_expressions.empty(); }
    void debug_print(bool tds_as_pointers = false) const;

    container::reference back() noexcept {return m_expressions.back();}
    container::const_reference back() const noexcept {return m_expressions.back();}
    container::reference front() noexcept {return m_expressions.front();}
    container::const_reference front() const noexcept {return m_expressions.front();}

    exprIt begin() noexcept {return m_expressions.begin();}
    exprIt end() noexcept {return m_expressions.end();}
    constExprIt begin() const noexcept {return cbegin();}
    constExprIt end() const noexcept {return cend();}
    constExprIt cbegin() const noexcept {return m_expressions.cbegin();}
    constExprIt cend() const noexcept {return m_expressions.cend();}
    exprReverseIt rbegin() noexcept {return m_expressions.rbegin();}
    exprReverseIt rend() noexcept {return m_expressions.rend();}
    constExprReverseIt crbegin() const noexcept {return m_expressions.crbegin();}
    constExprReverseIt crend() const noexcept {return m_expressions.crend();}

    exprIt insert(constExprIt pos, const ov::NodeVector& nodes);
    exprIt insert(constExprIt pos, const std::shared_ptr<Node>& n);
    exprIt insert(constExprIt pos, container::value_type&& value);
    exprIt insert(constExprIt pos, const container::value_type& value);
    exprIt insert(constExprIt pos, exprIt begin, exprIt end);
    exprIt insert(constExprIt pos, constExprIt begin, constExprIt end);

    exprIt erase(exprIt pos);
    exprIt erase(constExprIt pos);

    constExprIt find(const ExpressionPtr& target) const;
    template<typename iterator>
    iterator find(iterator begin, iterator end, const ExpressionPtr& target) const;
    template<typename iterator>
    iterator find_before(iterator it, const ExpressionPtr& target) const;
    template<typename iterator>
    iterator find_after(iterator it, const ExpressionPtr& target) const;

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

    container m_expressions{};
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Expression>> m_node2expression_map;
    io_container m_io_expressions;
    Config m_config{};
    LoopManagerPtr m_loop_manager = nullptr;
};

template<typename iterator>
iterator LinearIR::find(iterator begin, iterator end, const ExpressionPtr& target) const {
    auto found = std::find(begin, end, target);
    OPENVINO_ASSERT(found != end, "Expression has not been found");
    return found;
}
} // namespace lowered
} // namespace snippets
} // namespace ov
