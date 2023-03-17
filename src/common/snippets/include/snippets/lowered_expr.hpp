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
    // True if we should check runtime info for nodes to call specific needed transformations
    bool m_need_fill_tail_register = false;
    bool m_explicit_loop_insertion = false;
    ov::PartialShape m_master_shape{};
    size_t m_loop_depth = 1;
};

class LoweredExprIR;
class LoweredExpr {
    friend LoweredExprIR;

public:
    static size_t LOOP_NULL_ID;

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
    std::vector<size_t> get_loop_ids() const { return m_loop_ids; }
    void set_loop_ids(const std::vector<size_t>& loops) { m_loop_ids = loops; }
    void set_loop_id(size_t id, size_t idx);
    void remove_loop_id(size_t id);

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

struct LoweredExprPort {
    enum Type {
        Input,
        Output
    };

    LoweredExprPort() = default;

    static LoweredExprPort make_input(const LoweredExprPtr& expr, size_t port);
    static LoweredExprPort make_output(const LoweredExprPtr& expr, size_t port);

    LoweredExprPtr expr = nullptr;
    size_t port = 0;
    Type type = Type::Input;

private:
    LoweredExprPort(const LoweredExprPtr& expr, size_t port, Type type);
};

bool operator==(const LoweredExprPort& lhs, const LoweredExprPort& rhs);
bool operator!=(const LoweredExprPort& lhs, const LoweredExprPort& rhs);
bool operator<(const LoweredExprPort& lhs, const LoweredExprPort& rhs);

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
    LoweredExprPort get_expr_by_output(const TensorDescriptorPtr& n) const;
    const std::set<LoweredExprPort>& get_exprs_by_input(const TensorDescriptorPtr& n) const;
    void replace_input(const LoweredExprPort& expr_port, const TensorDescriptorPtr& to);
    void replace_input(const LoweredExprPtr& expr, size_t port, const TensorDescriptorPtr& to);
    void replace_output(const LoweredExprPort& expr_port, const TensorDescriptorPtr& to);
    void replace_output(const LoweredExprPtr& expr, size_t port, const TensorDescriptorPtr& to);
    exprIt insert(constExprIt pos, const ov::NodeVector& nodes);
    exprIt insert(constExprIt pos, const std::shared_ptr<Node>& n);
    exprIt insert(constExprIt pos, container::value_type&& value);
    exprIt insert(constExprIt pos, const container::value_type& value);
    exprIt insert(constExprIt pos, exprIt begin, exprIt end);
    exprIt insert(constExprIt pos, constExprIt begin, constExprIt end);

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
    exprIt erase(exprIt pos);
    exprIt erase(constExprIt pos);
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
    static ov::NodeVector get_ordered_ops(const std::shared_ptr<ov::Model>& model);
    void serialize(const std::string& xml, const std::string& bin);

    class LoweredLoopManager {
    public:
        LoweredLoopManager() = default;

        class LoweredLoopInfo {
        public:
            LoweredLoopInfo() = default;
            LoweredLoopInfo(size_t work_amount, size_t increment,
                            const std::vector<LoweredExprPort>& entries,
                            const std::vector<LoweredExprPort>& exits)
                    : work_amount(work_amount), increment(increment), entry_exprs(entries), exit_exprs(exits) {}
            size_t work_amount = 0;
            size_t increment = 0;
            // The order of entry and exit expressions is important:
            //     - The position before first entry expr is Loop Begin position
            //     - The position after last exit expr is Loop End position
            // Note: Scalars aren't entry expressions but can be before first entry expr in Linear IR
            std::vector<LoweredExprPort> entry_exprs = {};
            std::vector<LoweredExprPort> exit_exprs = {};
        };
        using LoweredLoopInfoPtr = std::shared_ptr<LoweredLoopInfo>;

        size_t add_loop_info(const LoweredLoopInfoPtr& loop);
        void remove_loop_info(size_t index);
        LoweredLoopInfoPtr get_loop_info(size_t index) const;
        size_t get_loop_count() const { return m_map.size(); }
        const std::map<size_t, LoweredLoopInfoPtr>& get_map() const;

        static void skipped_mark(LoweredExprIR::constExprIt loop_begin_pos,
                                 LoweredExprIR::constExprIt loop_end_pos,
                                 size_t loop_depth);
        void mark_loop(LoweredExprIR& linear_ir,
                       LoweredExprIR::constExprIt loop_begin_pos,
                       LoweredExprIR::constExprIt loop_end_pos,
                       size_t loop_depth, size_t vector_size);
        void mark_loop(LoweredExprIR& linear_ir,
                       LoweredExprIR::constExprIt loop_begin_pos,
                       LoweredExprIR::constExprIt loop_end_pos,
                       size_t idx,
                       size_t work_amount,
                       size_t work_amount_increment,
                       const std::vector<LoweredExprPort>& entries,
                       const std::vector<LoweredExprPort>& exits);

        void get_loop_bounds(const LoweredExprIR& linear_ir,
                             size_t loop_id,
                             LoweredExprIR::constExprIt& loop_begin_pos,
                             LoweredExprIR::constExprIt& loop_end_pos) const;
        static void get_loop_bounds(const LoweredExprIR& linear_ir,
                                    const std::vector<LoweredExprPort>& entries,
                                    const std::vector<LoweredExprPort>& exits,
                                    LoweredExprIR::constExprIt& loop_begin_pos,
                                    LoweredExprIR::constExprIt& loop_end_pos,
                                    size_t loop_id = LoweredExpr::LOOP_NULL_ID);

    private:
        static void exprs_marking(LoweredExprIR::constExprIt loop_begin_pos,
                                  LoweredExprIR::constExprIt loop_end_pos,
                                  size_t loop_id, size_t idx);
        static void get_io_loop_ports(LoweredExprIR& linear_ir,
                                      LoweredExprIR::constExprIt loop_begin_pos,
                                      LoweredExprIR::constExprIt loop_end_pos,
                                      std::vector<LoweredExprPort>& entries,
                                      std::vector<LoweredExprPort>& exits);

        std::map<size_t, LoweredLoopInfoPtr> m_map = {};
        size_t next_id = 0;
    };
    using LoweredLoopManagerPtr = std::shared_ptr<LoweredLoopManager>;

    const LoweredLoopManagerPtr& get_loop_manager() const { return m_loop_manager; }

private:
    void register_expression(const LoweredExprPtr& expr);
    // Like register_expression, but doesn't allow Parameter or Result registration. You can do it only through constructon
    void register_regular_expression(const LoweredExprPtr& expr);
    void unregister_expression(const LoweredExprPtr& expr);
    container m_lowered_ops{};
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<LoweredExpr>> m_node2expression_map;
    // Expression must be uniquely identified by an output, so there can't be expressions that have the same output
    std::unordered_map<TensorDescriptorPtr, LoweredExprPort> m_output2expression_map;
    // At the same time, several expressions can have the same input if they are connected to the same parent
    // E.g. LoopEnd will always have the same input as a Load inside the loop (since it has to increment the same reg)
    std::unordered_map<TensorDescriptorPtr, std::set<LoweredExprPort>> m_input2expression_map;
    io_container m_io_lowered_ops;
    LoweringConfig m_config{};
    LoweredLoopManagerPtr m_loop_manager = nullptr;
};

using AllocatedEmitter = std::pair<std::shared_ptr<Emitter>, RegInfo>;

} // namespace snippets
} // namespace ngraph