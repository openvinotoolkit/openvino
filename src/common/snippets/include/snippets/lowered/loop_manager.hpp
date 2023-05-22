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

    class LoopInfo {
    public:
        LoopInfo() = default;
        LoopInfo(size_t work_amount, size_t increment, size_t dim_idx,
                 const std::vector<ExpressionPort>& entries,
                 const std::vector<ExpressionPort>& exits)
            : work_amount(work_amount), increment(increment), dim_idx(dim_idx), entry_exprs(entries), exit_exprs(exits) {}
        size_t work_amount = 0;
        size_t increment = 0;
        size_t dim_idx = 0;  // The numeration begins from the end (dim_idx = 0 -> is the most inner dimension)
        // The order of entry and exit expressions is important:
        //     - The position before first entry expr is Loop Begin position
        //     - The position after last exit expr is Loop End position
        // Note: Scalars aren't entry expressions but can be before first entry expr in Linear IR
        std::vector<ExpressionPort> entry_exprs = {};
        std::vector<ExpressionPort> exit_exprs = {};
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
                                const std::vector<ExpressionPort>& entries,
                                const std::vector<ExpressionPort>& exits,
                                LinearIR::constExprIt& loop_begin_pos,
                                LinearIR::constExprIt& loop_end_pos,
                                size_t loop_id);

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
