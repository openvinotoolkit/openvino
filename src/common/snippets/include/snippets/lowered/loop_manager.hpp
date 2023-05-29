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
            : port(std::make_shared<ExpressionPort>(port)), is_incremented(is_scheduled) {}

        friend bool operator==(const LoopPort& lhs, const LoopPort& rhs);
        friend bool operator!=(const LoopPort& lhs, const LoopPort& rhs);
        friend bool operator<(const LoopPort& lhs, const LoopPort& rhs);

        std::shared_ptr<ExpressionPort> port = {};
        // True if after each Loop iteration the corresponding data pointer should be incremented.
        // Otherwise, the data pointer shift is skipped
        bool is_incremented = true;
    };

    class LoopInfo {
    public:
        LoopInfo() = default;
        LoopInfo(size_t work_amount, size_t increment, size_t dim_idx,
                 const std::vector<LoopPort>& entries,
                 const std::vector<LoopPort>& exits)
            : work_amount(work_amount), increment(increment), dim_idx(dim_idx), entry_points(entries), exit_points(exits) {}
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
    };
    using LoopInfoPtr = std::shared_ptr<LoopInfo>;

    size_t add_loop_info(const LoopInfoPtr& loop);
    void remove_loop_info(size_t index);
    LoopInfoPtr get_loop_info(size_t index) const;
    size_t get_loop_count() const { return m_map.size(); }
    const std::map<size_t, LoopInfoPtr>& get_map() const;

    void mark_loop(LinearIR::constExprIt loop_begin_pos,
                   LinearIR::constExprIt loop_end_pos,
                   size_t loop_depth, size_t vector_size);
    void mark_loop(LinearIR::constExprIt loop_begin_pos,
                   LinearIR::constExprIt loop_end_pos,
                   size_t work_amount,
                   size_t work_amount_increment,
                   size_t dim_idx,
                   const std::vector<ExpressionPort>& entries,
                   const std::vector<ExpressionPort>& exits);

    void get_loop_bounds(const LinearIR& linear_ir,
                         size_t loop_id,
                         LinearIR::constExprIt& loop_begin_pos,
                         LinearIR::constExprIt& loop_end_pos) const;
    static void get_loop_bounds(const LinearIR& linear_ir,
                                const std::vector<LoopPort>& entries,
                                const std::vector<LoopPort>& exits,
                                LinearIR::constExprIt& loop_begin_pos,
                                LinearIR::constExprIt& loop_end_pos,
                                size_t loop_id);

    /* ===== The methods for work with Loop IDs of Expression ===== */
    // Notes:
    //  - These methods don't update the corresponding LoopInfo
    //  - These methods should be private
    // TODO [112195] : fix these notes
    void replace_loop_id(const ExpressionPtr& expr, size_t prev_id, size_t new_id);
    void remove_loop_id(const ExpressionPtr& expr, size_t id);
    // Insert loop ID before (as outer Loop) or after (as inner Loop) target ID in vector of identifiers
    // Before:                                 | After:
    //   loop_ids: [.., new_id, target_id, ..] |    loop_ids: [.., target_id, new_id, ..]
    // Default value of target ID - SIZE_MAX - for `after` the new Loop is the most inner Loop
    //                                         for `before` the new Loop is the most outer Loop
    void insert_loop_id(const ExpressionPtr& expr, size_t new_id, bool before = true, size_t target_id = SIZE_MAX);
    void insert_loop_ids(const ExpressionPtr& expr, const std::vector<size_t>& new_ids, bool before = true, size_t target_id = SIZE_MAX);

private:
    static void get_io_loop_ports(LinearIR::constExprIt loop_begin_pos,
                                  LinearIR::constExprIt loop_end_pos,
                                  std::vector<ExpressionPort>& entries,
                                  std::vector<ExpressionPort>& exits);

    std::map<size_t, LoopInfoPtr> m_map = {};
    size_t next_id = 0;
};

} // namespace lowered
} // namespace snippets
} // namespace ov
