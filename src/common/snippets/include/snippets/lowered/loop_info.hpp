// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"


namespace ov {
namespace snippets {
namespace lowered {

/* The structure contains the information about a Loop in Linear Intermediate IR (Linear IR):
 * work amount of the Loop, step of loop counter increment, entry and exit ports of the Loop,
 * passes for specific iterations.
 */
class LoopInfo {
public:
    enum {UNDEFINED_DIM_IDX = std::numeric_limits<size_t>::max()};

    LoopInfo() = default;
    LoopInfo(size_t work_amount, size_t increment,
             const std::vector<LoopPort>& entries,
             const std::vector<LoopPort>& exits,
             const SpecificIterationHandlers& handlers = SpecificIterationHandlers());
    LoopInfo(size_t work_amount, size_t increment,
             const std::vector<ExpressionPort>& entries,
             const std::vector<ExpressionPort>& exits,
             const SpecificIterationHandlers& handlers = SpecificIterationHandlers());

    /**
     * @brief Clone LoopInfo with new expressions
     * @param expr_map map of new and old expressions
     * @return the copy
     */
    std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map) const;

    /**
     * @brief Returns dimension index if dimension indices for all entry and exit points are equal.
     *        Otherwise returns UNDEFINED_DIM_IDX.
     * @return index
     */
    size_t get_dim_idx() const;
    /**
     * @brief Returns work amount of the Loop.
     * @return m_work_amount
     */
    size_t get_work_amount() const;
    /**
     * @brief Returns step of loop counter increment.
     * @return m_increment
     */
    size_t get_increment() const;
    /**
     * @brief Returns vector of loop input ports
     * @return m_entry_points
     */
    const std::vector<LoopPort>& get_entry_points() const;
    /**
     * @brief Returns vector of loop outputs ports
     * @return m_exit_points
     */
    const std::vector<LoopPort>& get_exit_points() const;
    /**
     * @brief Returns handlers of loop specific iterations
     * @return m_handlers
     */
    const SpecificIterationHandlers& get_handlers() const;

    /**
     * @brief Sets `dim_idx` to all entry and exit points
     * @param dim_idx - index
     */
    void set_dim_idx(size_t dim_idx);
    /**
     * @brief Set m_work_amount value
     * @param work_amount - work amount of the loop
     */
    void set_work_amount(size_t work_amount);
    /**
     * @brief Set m_increment value
     * @param increment - step of loop counter increment
     */
    void set_increment(size_t increment);
    /**
     * @brief Set m_entry_points value
     * @param entry_points - vector of loop input ports
     */
    void set_entry_points(std::vector<LoopPort> entry_points);
    /**
     * @brief Set m_exit_points value
     * @param exit_points - vector of loop output ports
     */
    void set_exit_points(std::vector<LoopPort> exit_points);
    /**
     * @brief Set m_handlers value
     * @param handlers - transformations for loop specific iterations
     */
    void set_handlers(SpecificIterationHandlers handlers);

    /**
     * @brief Register loop specific iteration handler
     * @param Type - type of specific iteration
     * @param T - transformation
     * @param args - arguments of the transformation
     */
    template <SpecificIterationHandlers::HandlerType Type, typename T, class... Args>
    void register_handler(Args&&... args) {
        m_handlers.register_handler<Type, T>(args...);
    }

    /**
     * @brief Update the parameters of existing loop input ports
     * @param updater - function that updates ports
     */
    void update_entry_points(const std::function<void(LoopPort&)>& updater);
    /**
     * @brief Update the parameters of existing loop output ports
     * @param updater - function that updates ports
     */
    void update_exit_points(const std::function<void(LoopPort&)>& updater);

private:
    size_t m_work_amount = 0;
    size_t m_increment = 0;
    // The order of entry and exit expressions is important:
    //     - The position before first entry expr is Loop Begin position
    //     - The position after last exit expr is Loop End position
    // Note: Scalars aren't entry expressions but can be before first entry expr in Linear IR
    std::vector<LoopPort> m_entry_points = {};
    std::vector<LoopPort> m_exit_points = {};
    SpecificIterationHandlers m_handlers = {};
};
using LoopInfoPtr = std::shared_ptr<LoopInfo>;

} // namespace lowered
} // namespace snippets
} // namespace ov
