// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>
#include "emitter.hpp"
#include "target_machine.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {

using code = const uint8_t *;
using RegInfo = std::pair<std::vector<size_t>, std::vector<size_t>>;

class LoweringConfig {
public:
    // True if the lowered Emitters need to be accessed during runtime. Normally they're destroyed after code emission.
    bool m_save_lowered_code = false;
    // True if we can optimize tails for single evaluation during code generation
    // More details with optimization examples you can see in generate() method
    // For example, tails with Buffer ops doesn't support single evaluation optimizations
    //              because of that we should always reset memory pointer using finalization offsets
    //              after data storing to Buffer
    bool m_optimize_single_evaluation = true;
    // True if we should check runtime info for nodes to call specific needed transformations
    bool m_need_fill_tail_register = false;
    bool m_explicit_loop_insertion = false;
    ov::PartialShape m_master_shape{};
    size_t m_loop_depth = 1;
};

/**
 * @interface Emitter
 * @brief Base class for all target specific code emitters used by generator.
 * @ingroup snippets
 */
class LoweredExprIR;
class LoweredExpr {
    friend LoweredExprIR;

public:
    /**
     * @brief Default constructor
     */
    explicit LoweredExpr(const std::shared_ptr<Node>& n);
    explicit LoweredExpr(const std::shared_ptr<Node>& n, std::vector<TensorDescriptorPtr> inputs, std::vector<TensorDescriptorPtr> outputs = {});
    LoweredExpr() = default;
    virtual ~LoweredExpr() = default;
    std::shared_ptr<Node> get_node() const;
    std::shared_ptr<Emitter> get_emitter() const;
    void init_emitter(const std::shared_ptr<const TargetMachine>& target);
    RegInfo get_reg_info() const {return  m_reg_info;}
    void set_reg_info(RegInfo rinfo) {m_reg_info = std::move(rinfo);}
    const std::vector<TensorDescriptorPtr>& get_inputs() {return m_inputs; }
    const std::vector<TensorDescriptorPtr>& get_outputs() {return m_outputs; }

protected:
    void replace_input(const TensorDescriptorPtr& from, TensorDescriptorPtr to);
    void replace_output(const TensorDescriptorPtr& from, TensorDescriptorPtr to);
    std::shared_ptr<Node> m_source_node{nullptr};
    std::shared_ptr<Emitter> m_emitter{nullptr};
    std::vector<TensorDescriptorPtr> m_inputs;
    std::vector<TensorDescriptorPtr> m_outputs;
    RegInfo m_reg_info{{}, {}};
};

class IOLoweredExpr : public LoweredExpr {
public:
    enum class io_type {INPUT, OUTPUT, UNDEFINED};
    IOLoweredExpr(const std::shared_ptr<ov::opset1::Parameter>& n, int64_t index);
    IOLoweredExpr(const std::shared_ptr<ov::opset1::Result>& n, int64_t index, std::vector<TensorDescriptorPtr> inputs);
    int64_t get_index() const  {return m_index;}
    io_type get_type() const {return m_type; }
private:
    int64_t m_index = -1;
    io_type m_type = io_type::UNDEFINED;
};

using LoweredExprPtr = std::shared_ptr<LoweredExpr>;
class LoweredExprIR {
public:
    using container = std::list<LoweredExprPtr>;
    using io_container = std::list<std::shared_ptr<IOLoweredExpr>>;
    using exprIt = container::iterator;
    using constExprIt = container::const_iterator;
    explicit LoweredExprIR(const std::shared_ptr<ov::Model>& m, LoweringConfig config = {});
    LoweredExprIR() = default;
    LoweredExprIR deep_copy() const;
    static LoweredExprIR::container deep_copy_range(LoweredExprIR::container::const_iterator begin, LoweredExprIR::container::const_iterator end);
    const container& get_ops() const {return m_lowered_ops; }
    const io_container& get_IO_ops() const {return m_io_lowered_ops; }
    void init_emitters(const std::shared_ptr<TargetMachine>& target);
    LoweringConfig get_config() {return m_config; }
    LoweredExprPtr get_expr_by_node(const std::shared_ptr<Node>& n) const;
    LoweredExprPtr get_expr_by_output(const TensorDescriptorPtr& n) const;
    const std::set<LoweredExprPtr>& get_exprs_by_input(const TensorDescriptorPtr& n) const;
    void replace_input(const LoweredExprPtr& expr, const TensorDescriptorPtr& from, TensorDescriptorPtr to);
    void replace_output(const LoweredExprPtr& expr, const TensorDescriptorPtr& from, const TensorDescriptorPtr& to);
    exprIt insert(constExprIt pos, const ov::NodeVector& nodes);
    exprIt insert(constExprIt pos, const std::shared_ptr<Node>& n);
    exprIt insert(constExprIt pos, container::value_type&& value);
    exprIt insert(constExprIt pos, const container::value_type& value);
    exprIt insert(constExprIt pos, exprIt begin, exprIt end);
    exprIt insert(constExprIt pos, constExprIt begin, constExprIt end);
    /**
    * @brief Move an expression from the position "from" to the position immediately before "to".
     * Returns iterator to the element after "from" position. The behavior of this method is identical to calling
     * insert(to, *from) + erase(from), except that no unnecessary updates of internal maps are performed.
     * Note: this method does NOT take care about data dependencies and no relevant checks are performed
    */
    LoweredExprIR::exprIt move(exprIt from, constExprIt to);

    bool empty() const noexcept {return m_lowered_ops.empty(); }
    void debug_print(bool tds_as_pointers = false) const;

    container::reference back() noexcept {return m_lowered_ops.back();}
    container::const_reference back() const noexcept {return m_lowered_ops.back();}
    container::reference front() noexcept {return m_lowered_ops.front();}
    container::const_reference front() const noexcept {return m_lowered_ops.front();}
    exprIt erase(exprIt pos);
    exprIt erase(constExprIt pos);
    exprIt begin() noexcept {return m_lowered_ops.begin();}
    exprIt end() noexcept {return m_lowered_ops.end();}
    constExprIt begin() const noexcept {return cbegin();}
    constExprIt end() const noexcept {return cend();}
    constExprIt cbegin() const noexcept {return m_lowered_ops.cbegin();}
    constExprIt cend() const noexcept {return m_lowered_ops.cend();}
    container ::reverse_iterator rbegin() noexcept {return m_lowered_ops.rbegin();}
    container::reverse_iterator rend() noexcept {return m_lowered_ops.rend();}
    container::const_reverse_iterator crbegin() const noexcept {return m_lowered_ops.crbegin();}
    container::const_reverse_iterator crend() const noexcept {return m_lowered_ops.crend();}
    static ov::NodeVector get_ordered_ops(const std::shared_ptr<ov::Model>& model);
    void serialize(const std::string& xml, const std::string& bin);

private:
    void register_expression(const LoweredExprPtr& expr);
    // Like register_expression, but doesn't allow Parameter or Result registration. You can do it only through constructon
    void register_regular_expression(const LoweredExprPtr& expr);
    void unregister_expression(const LoweredExprPtr& expr);
    container m_lowered_ops{};
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<LoweredExpr>> m_node2expression_map;
    // Expression must be uniquely identified by an output, so there can't be expressions that have the same output
    std::unordered_map<TensorDescriptorPtr , LoweredExprPtr> m_output2expression_map;
    // At the same time, several expressions can have the same input if they are connected to the same parent
    // E.g. LoopEnd will always have the same input as a Load inside the loop (since it has to increment the same reg)
    std::unordered_map<TensorDescriptorPtr , std::set<LoweredExprPtr>> m_input2expression_map;
    io_container m_io_lowered_ops;
    LoweringConfig m_config{};
};

using AllocatedEmitter = std::pair<std::shared_ptr<Emitter>, RegInfo>;

} // namespace snippets
} // namespace ngraph