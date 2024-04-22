// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include "snippets/lowered/expression.hpp"
#include "snippets/target_machine.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov {
namespace snippets {
namespace lowered {

#ifdef SNIPPETS_DEBUG_CAPS
// Snippets performance count mode
// Disabled - default, w/o perf count for snippets
// Chrono - perf count with chrono call. This is a universal method, and support multi-thread case to output perf count data for each thread.
// BackendSpecific - perf count provided by backend. This is for device specific requirment.
// For example, in sake of more light overhead and more accurate result, x86 CPU specific mode via read RDTSC register is implemented,
// which take ~50ns, while Chrono mode take 260ns for a pair of perf count start and perf count end execution, on ICX. This mode only support single thread.
enum PerfCountMode {
    Disabled,
    Chrono,
    BackendSpecific,
};
#endif

class Config {
public:
    // True if we should check runtime info for nodes to call specific needed transformations
    bool m_need_fill_tail_register = false;
    size_t m_loop_depth = 1;
#ifdef SNIPPETS_DEBUG_CAPS
    PerfCountMode perf_count_mode = PerfCountMode::Disabled;
#endif
    // Some Subgraphs doesn't support domain optimization due to operations' semantics
    bool m_enable_domain_optimization = false;
    // Minimal advised work amount for parallel execution.
    // Set by a backend, typically equals to the number of threads available on the machine.
    size_t m_min_parallel_work_amount = 8;
    // Minimal advised work amount that should be processed during one call of the executable produced by Subgraph::generate
    // Set by a backend, should be large enough to compensate for the kernel call overheads
    size_t m_min_kernel_work_amount = 256;
    // True if the Buffer scratchpad size of LinearIR will be optimized (all possible optimizations will be activated)
    // False if all Buffers will have uniqie ID and offsets in the Linear IR
    bool m_are_buffers_optimized = true;
};

class LoopManager;
using LoopManagerPtr = std::shared_ptr<LoopManager>;

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
    LinearIR(const std::shared_ptr<ov::Model>& m, const std::shared_ptr<IShapeInferSnippetsFactory>& factory, Config config = {});

    ExpressionPtr create_expression(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& inputs) const;

    std::shared_ptr<LinearIR> clone() const;
    static LinearIR::container deep_copy_range(LinearIR::container::const_iterator begin,
                                               LinearIR::container::const_iterator end,
                                               ExpressionMap& expression_map);

    const container& get_ops() const { return m_expressions; }
    const io_container& get_IO_ops() const { return m_io_expressions; }
    const Config& get_config() const { return m_config; }
    void set_loop_depth(size_t loop_depth) { m_config.m_loop_depth = loop_depth; }

    const ExpressionPtr& get_expr_by_node(const std::shared_ptr<Node>& n) const;

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

    const LoopManagerPtr& get_loop_manager() const { return m_loop_manager; }

    IShapeInferSnippets::Result shape_infer(const std::vector<VectorDimsRef>& input_shapes);
    const std::shared_ptr<ShapeInferSnippetsNode>& get_shape_infer_instance() const {return m_shape_infer; }
    VectorDims get_master_shape() const;
    VectorDims get_parallel_domain() const;

    bool is_dynamic() const;

    /* ------ Helpers for work with LinearIR ----- */
    /**
     * @brief Creates new Expression from `new_node` with inputs `inputs`,
     *        sets `loops_ids` as loop identifiers and inserts the expression on the `place` in LinearIR.
     *        Also connects output ports to `consumers`
     * @param new_node the target node
     * @param inputs template argument that might be:
     *        - vector of PortConnectorsPtr that will be inputs of the expression
     *        - vector of output ExpressionPort that will be source parent ports
     * @param loop_ids vector of loops ids that will be set for the expression
     * @param update_loop_ports true - the helpers updates the corresponding loop ports after insertion otherwise - skip
     * @param place before this place expression will be inserted
     * @param consumers vector of expression port sets. These expression ports will be consumers of the expression.
     *        The vector may be empty or size of vector must be equal to output port count
     * @return new expression iterator in LinearIR
     */
    template<typename T>
    exprIt insert_node(const std::shared_ptr<ov::Node>& new_node, const std::vector<T>& inputs, const std::vector<size_t>& loop_ids,
                       bool update_loop_ports, const constExprIt& place, const std::vector<std::set<ExpressionPort>>& consumers);
    /**
     * @brief The same helper as the helpers above but for case when new_node has only one output.
     * @param new_node the target node
     * @param inputs template argument that might be:
     *        - vector of PortConnectorsPtr that will be inputs of the expression
     *        - vector of output ExpressionPort that will be source parent ports
     * @param loop_ids vector of loops ids that will be set for the expression
     * @param update_loop_ports true - the helpers updates the corresponding loop ports after insertion otherwise - skip
     * @param place before this place expression will be inserted
     * @param consumers vector of expression port sets. These expression ports will be consumers of the expression.
     *        The vector may be empty or size of vector must be equal to output port count
     * @return new expression iterator in LinearIR
     */
    template<typename T>
    exprIt insert_node(const std::shared_ptr<ov::Node>& new_node, const std::vector<T>& inputs, const std::vector<size_t>& loop_ids,
                       bool update_loop_ports, const constExprIt& place, const std::set<ExpressionPort>& consumers = {}) {
        const auto consumers_py_port = consumers.empty() ? std::vector<std::set<ExpressionPort>>{} : std::vector<std::set<ExpressionPort>>{ consumers };
        return insert_node(new_node, inputs, loop_ids, update_loop_ports, place, consumers_py_port);
    }
    /**
     * @brief Replace the several existing expressions with the one new expression that contains `new_node`.
     *        Calls the helper `insert_node` and performs substitution: removes `old_exprs`.
     *        Also the helper move consumers from last expression in `old_exprs` to the new expression.
     *        Notes:
     *         - The helper supports only the sequence of `old_exprs`.
     *           It means that consumers of the expression in seq are expressions from this sequence (except of the last expr)
     *           and all sources of the `old_exprs` must be expression from this seq as well or must be source of `new_inputs`
     *        - The helper set output PortDescriptors - clones from last removable expression
     *        - The helpers updates LoopPorts of the corresponding loops using information about removable expressions
     * @param old_exprs the sequence of removable expressions
     * @param new_node the target node
     * @param loop_ids vector of loops ids that will be set for the expression
     * @param place before this place expression will be inserted
     * @return new expression iterator in LinearIR
     */
    exprIt replace_with_node(const std::vector<ExpressionPtr>& old_exprs, const std::shared_ptr<ov::Node>& new_node, const std::vector<size_t>& loop_ids,
                             const constExprIt& place);
    /**
     * @brief Replace the several existing expressions with the one new expression that contains `new_node` that are in the same loops.
     *        Insert new expression on the place of last `old_exprs`
     * @param old_exprs the sequence of removable expressions
     * @param new_node the target node
     * @return new expression iterator in LinearIR
     */
    exprIt replace_with_node(const std::vector<ExpressionPtr>& old_exprs, const std::shared_ptr<ov::Node>& new_node);
    /**
     * @brief Replace the several existing expressions with the one new expression.
     *        The helper move consumers from last expression in `old_exprs` to the new expression.
     *        Notes:
     *         - The helper supports only the sequence of `old_exprs`.
     *           It means that consumers of the expression in seq are expressions from this sequence (except of the last expr)
     *           and all sources of the `old_exprs` must be expression from this seq as well or must be source of `new_inputs`
     *        - The helpers updates LoopPorts of the corresponding loops using information about removable expressions
     * @param old_exprs the sequence of removable expressions
     * @param new_expr the new expr
     * @param place before this place expression will be inserted
     * @return new expression iterator in LinearIR
     */
    exprIt replace_with_expr(const std::vector<ExpressionPtr>& old_exprs, const ExpressionPtr& new_expr, const constExprIt& place);
    /**
     * @brief Replace the several existing expressions with the one new expression that are in the same loops.
     *        Insert new expression on the place of last `old_exprs`
     * @param old_exprs the sequence of removable expressions
     * @param new_expr the new expr
     * @return new expression iterator in LinearIR
     */
    exprIt replace_with_expr(const std::vector<ExpressionPtr>& old_exprs, const ExpressionPtr& new_expr);

private:
    std::shared_ptr<ShapeInferSnippetsNode> m_shape_infer = nullptr;

    class LIRShapeInfer : public ShapeInferSnippetsNode {
    public:
        using IOExpression = lowered::IOExpression;
        explicit LIRShapeInfer(container& body_exprs, io_container& io_exprs);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;

    private:
        const std::shared_ptr<container> m_exprs = nullptr;
        std::vector<std::shared_ptr<IOExpression>> m_input_exprs {};
        std::vector<std::shared_ptr<IOExpression>> m_output_exprs {};
    };

    static ov::NodeVector get_ordered_ops(const std::shared_ptr<ov::Model>& model);
    // Default ctor - can be called only from Linear IR initialization as default way
    ExpressionPtr create_expression(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model = nullptr);
    ExpressionPtr create_expression(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& new_inputs,
                                    const std::vector<size_t>& loop_ids, bool update_loop_ports, const std::vector<std::set<ExpressionPort>>& consumers = {});

    void register_expression(const ExpressionPtr& expr, bool io_allowed = false);
    void unregister_expression(const ExpressionPtr& expr);

    container m_expressions{};
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Expression>> m_node2expression_map;
    io_container m_io_expressions;
    Config m_config{};
    LoopManagerPtr m_loop_manager = nullptr;
    std::shared_ptr<IShapeInferSnippetsFactory> m_shape_infer_factory;
    bool m_is_dynamic = false;
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
