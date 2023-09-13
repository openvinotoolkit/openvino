// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_ir.hpp"

#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>

#include "port_descriptor.hpp"

namespace ov {
namespace snippets {
namespace lowered {

class LinearIR::LoopManager {
public:
    LoopManager() = default;

    struct LoopPort {
        LoopPort() = default;
        LoopPort(const ExpressionPort& port, bool is_scheduled = true)
            : expr_port(std::make_shared<ExpressionPort>(port)), is_incremented(is_scheduled) {}

        friend bool operator==(const LoopPort& lhs, const LoopPort& rhs);
        friend bool operator!=(const LoopPort& lhs, const LoopPort& rhs);
        friend bool operator<(const LoopPort& lhs, const LoopPort& rhs);

        std::shared_ptr<ExpressionPort> expr_port = {};
        // True if after each Loop iteration the corresponding data pointer should be incremented.
        // Otherwise, the data pointer shift is skipped
        bool is_incremented = true;
        int64_t ptr_increment = 0;
        int64_t finalization_offset = 0;
        int64_t data_size = 0;
    };

    class LoopInfo {
    public:
        LoopInfo() = default;
        LoopInfo(size_t work_amount, size_t increment, size_t dim_idx,
                 const std::vector<LoopPort>& entries,
                 const std::vector<LoopPort>& exits)
            : work_amount(work_amount), increment(increment), dim_idx(dim_idx),
              entry_points(entries), exit_points(exits), outer_splited_loop(false) {}
        LoopInfo(size_t work_amount, size_t increment, size_t dim_idx,
                 const std::vector<ExpressionPort>& entries,
                 const std::vector<ExpressionPort>& exits);

        size_t work_amount = 0;
        size_t increment = 0;
        size_t dim_idx = 0;  // The numeration begins from the end (dim_idx = 0 -> is the most inner dimension)
        // The order of entry and exit expressions is important:
        //     - The position before first entry expr is Loop Begin position
        //     - The position after last exit expr is Loop End position
        // Note: Scalars aren't entry expressions but can be before first entry expr in Linear IR
        std::vector<LoopPort> entry_points = {};
        std::vector<LoopPort> exit_points = {};
        // True if this Loop is outer Loop for nested Loops that splits the same dimension
        bool outer_splited_loop = false;
    };
    using LoopInfoPtr = std::shared_ptr<LoopInfo>;

    size_t add_loop_info(const LoopInfoPtr& loop);
    void remove_loop_info(size_t index);
    LoopInfoPtr get_loop_info(size_t index) const;
    size_t get_loop_count() const { return m_map.size(); }
    const std::map<size_t, LoopInfoPtr>& get_map() const;

    // Return outer Loop IDs
    static std::vector<size_t> get_outer_expr_loops(const ExpressionPtr& expr, size_t loop_id);

    void mark_loop(LinearIR::constExprIt loop_begin_pos,
                   LinearIR::constExprIt loop_end_pos,
                   size_t loop_depth, size_t vector_size);
    // Return Loop ID
    template <typename T>
    size_t mark_loop(LinearIR::constExprIt loop_begin_pos,
                    LinearIR::constExprIt loop_end_pos,
                    size_t work_amount, size_t work_amount_increment, size_t dim_idx,
                    const std::vector<T>& entries,
                    const std::vector<T>& exits) {
        const auto loop_info = std::make_shared<LoopManager::LoopInfo>(work_amount, work_amount_increment, dim_idx, entries, exits);
        const auto loop_id = this->add_loop_info(loop_info);
        for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
            insert_loop_id(*expr_it, loop_id);
        }
        return loop_id;
    }

    void fuse_loops(const LinearIR& linear_ir, size_t loop_id_upper, size_t loop_id_lower, bool fuse_into_upper = true);
    void fuse_loops(LinearIR::constExprIt loop_begin_target, LinearIR::constExprIt loop_end_target,
                    size_t loop_id_upper, size_t loop_id_lower, bool fuse_into_upper = true);

    // The following methods update ports of LoopInfo. They save the order of ports!
    // Remainder: the order is important to find Loop bounds (the most first and the most last expressions)
    //   - Update LoopPort - insert new loop target ports instead of existing.
    //   - Update ExpressionPort in the LoopPort - with saving of port parameters. It's softer method since ExpressionPort may not be port of Loop
    template<typename T>
    void update_loop_port(size_t loop_id, const T& actual_port, const std::vector<T>& target_ports, bool is_entry = true);
    template<typename T>
    void update_loops_port(const std::vector<size_t>& loop_ids, const T& actual_port,
                           const std::vector<T>& target_ports, bool is_entry = true) {
        for (auto loop_id : loop_ids) {
            update_loop_port(loop_id, actual_port, target_ports, is_entry);
        }
    }
    // Sort Loop Ports by expression locations in Linear IR
    void sort_loop_ports(LinearIR::constExprIt& loop_begin_pos, LinearIR::constExprIt& loop_end_pos, size_t loop_id);

    // When the previous expression was replaced with new expressions (decomposition), the method updates the corresponding Loop.
    // If ports of decomposed expression were the Loop ports, these Loop ports may be updated by parameters `entries` and `exits`
    // Note: This method should be removed when Softmax decomposition will be moved on data flow pipeline since
    //       all decompositions should be call on this pipeline
    void expression_replacement(constExprIt new_expr_begin, constExprIt new_expr_end, const ExpressionPtr& decomposed_expr,
                                size_t loop_id, const std::vector<ExpressionPort>& new_entries, const std::vector<ExpressionPort>& exits);

    // Note: these methods find iterators of first entry loop point and last exit point (bounds of Loop)
    //       If there are already inserted LoopBegin and LoopEnd in Linear IR, the methods can find them as well if `loop_ops_inserted` = true
    void get_loop_bounds(const LinearIR& linear_ir,
                         size_t loop_id,
                         LinearIR::constExprIt& loop_begin_pos,
                         LinearIR::constExprIt& loop_end_pos,
                         bool loop_ops_inserted = false) const;
    static void get_loop_bounds(const LinearIR& linear_ir,
                                const std::vector<LoopPort>& entries,
                                const std::vector<LoopPort>& exits,
                                LinearIR::constExprIt& loop_begin_pos,
                                LinearIR::constExprIt& loop_end_pos,
                                size_t loop_id, bool loop_ops_inserted = false);

    LoopPort get_loop_port_by_expr_port(const ExpressionPort& expr_port, const size_t loop_id);

private:
    static void get_io_loop_ports(LinearIR::constExprIt loop_begin_pos,
                                  LinearIR::constExprIt loop_end_pos,
                                  std::vector<ExpressionPort>& entries,
                                  std::vector<ExpressionPort>& exits);

    static void fuse_loop_ports(std::vector<LinearIR::LoopManager::LoopPort>& exit_points,
                                std::vector<LinearIR::LoopManager::LoopPort>& entry_points,
                                size_t loop_id);

    /* ===== The methods for work with Loop IDs of Expression ===== */
    // Notes:
    //  - These methods don't update the corresponding LoopInfo
    void replace_loop_id(const ExpressionPtr& expr, size_t prev_id, size_t new_id);
    void remove_loop_id(const ExpressionPtr& expr, size_t id);
    // Insert loop ID before (as outer Loop) or after (as inner Loop) target ID in vector of identifiers
    // Before:                                 | After:
    //   loop_ids: [.., new_id, target_id, ..] |    loop_ids: [.., target_id, new_id, ..]
    // Default value of target ID - SIZE_MAX - for `after` the new Loop is the most inner Loop
    //                                         for `before` the new Loop is the most outer Loop
    void insert_loop_id(const ExpressionPtr& expr, size_t new_id, bool before = true, size_t target_id = SIZE_MAX);
    void insert_loop_ids(const ExpressionPtr& expr, const std::vector<size_t>& new_ids, bool before = true, size_t target_id = SIZE_MAX);

    std::map<size_t, LoopInfoPtr> m_map = {};
    size_t next_id = 0;
};

} // namespace lowered
} // namespace snippets
} // namespace ov
