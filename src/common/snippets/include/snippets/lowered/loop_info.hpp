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
    LoopInfo(size_t work_amount, size_t increment, const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits);
    LoopInfo(size_t work_amount, size_t increment, const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits);
    virtual ~LoopInfo() = default;

    /**
     * @brief Clone LoopInfo with new expressions
     * @param expr_map map of new and old expressions
     * @return the copy
     */
    virtual std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map) const = 0;

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
     * @return handlers
     */
    virtual const SpecificIterationHandlers& get_handlers() const = 0;

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
     * @brief Set handlers
     * @param handlers - transformations for loop specific iterations
     */
    virtual void set_handlers(SpecificIterationHandlers handlers) = 0;

    /**
     * @brief Update the parameters of existing loop input ports
     * @param updater - function that updates ports
     */
    inline void update_entry_points(const std::function<void(LoopPort&)>& updater) {
        std::for_each(m_entry_points.begin(), m_entry_points.end(), updater);
    }
    /**
     * @brief Update the parameters of existing loop output ports
     * @param updater - function that updates ports
     */
    inline void update_exit_points(const std::function<void(LoopPort&)>& updater) {
        std::for_each(m_exit_points.begin(), m_exit_points.end(), updater);
    }
    /**
     * @brief Calls `initializer` for each entry point
     * @param initializer - function that address to entry point
     */
    inline void init_using_entry_points(const std::function<void(const LoopPort&)>& initializer) const {
        std::for_each(m_entry_points.cbegin(), m_entry_points.cend(), initializer);
    }
    /**
     * @brief Calls `initializer` for each exit point
     * @param initializer - function that address to exit point
     */
    inline void init_using_exit_points(const std::function<void(const LoopPort&)>& initializer) const {
        std::for_each(m_exit_points.cbegin(), m_exit_points.cend(), initializer);
    }

    /**
     * @brief Returns vector with boolean values `is_incremented` of loop ports
     * @return vector with boolean values
     */
    std::vector<bool> get_is_incremented() const;
    /**
     * @brief Returns vector with pointer increments of loop ports
     * @return vector with ptr increments
     */
    std::vector<int64_t> get_ptr_increments() const;
    /**
     * @brief Returns vector with finalization offsets of loop ports
     * @return vector with finalization offsets
     */
    std::vector<int64_t> get_finalization_offsets() const;
    /**
     * @brief Returns vector with data sizes of loop ports
     * @return vector with data sizes
     */
    std::vector<int64_t> get_data_sizes() const;

protected:
    static std::vector<LoopPort> clone_loop_ports(const ExpressionMap& expr_map, const std::vector<LoopPort>& port_ports);

    size_t m_work_amount = 0;
    size_t m_increment = 0;
    // The order of entry and exit expressions is important:
    //     - The position before first entry expr is Loop Begin position
    //     - The position after last exit expr is Loop End position
    // Note: Scalars aren't entry expressions but can be before first entry expr in Linear IR
    std::vector<LoopPort> m_entry_points = {};
    std::vector<LoopPort> m_exit_points = {};
};
using LoopInfoPtr = std::shared_ptr<LoopInfo>;

class UnifiedLoopInfo : public LoopInfo {
public:
    UnifiedLoopInfo() = default;
    UnifiedLoopInfo(size_t work_amount, size_t increment,
                    const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                    const SpecificIterationHandlers& handlers = SpecificIterationHandlers());
    UnifiedLoopInfo(size_t work_amount, size_t increment,
                    const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits,
                    const SpecificIterationHandlers& handlers = SpecificIterationHandlers());

    /**
     * @brief Clone LoopInfo with new expressions
     * @param expr_map map of new and old expressions
     * @return the copy
     */
    std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map) const override;

    /**
     * @brief Returns handlers of loop specific iterations
     * @return m_handlers
     */
    const SpecificIterationHandlers& get_handlers() const override;
    /**
     * @brief Set m_handlers value
     * @param handlers - transformations for loop specific iterations
     */
    void set_handlers(SpecificIterationHandlers handlers) override;

    /**
     * @brief Register loop specific iteration handler
     * @param Type - type of specific iteration
     * @param T - transformation
     * @param args - arguments of the transformation
     */
    template <SpecificLoopIterType Type, typename T, class... Args>
    void register_handler(Args&&... args) {
        m_handlers.register_handler<Type, T>(args...);
    }

private:
    SpecificIterationHandlers m_handlers = {};
};
using UnifiedLoopInfoPtr = std::shared_ptr<UnifiedLoopInfo>;

class ExpandedLoopInfo : public LoopInfo {
public:
    ExpandedLoopInfo() = default;
    ExpandedLoopInfo(size_t work_amount, size_t increment,
                     const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                     SpecificLoopIterType type, std::shared_ptr<UnifiedLoopInfo> original_loop_info);
    ExpandedLoopInfo(size_t work_amount, size_t increment,
                     const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                     std::vector<int64_t> ptr_increments, std::vector<int64_t> final_offsets, std::vector<int64_t> data_sizes,
                     SpecificLoopIterType type, std::shared_ptr<UnifiedLoopInfo> unified_loop_info);
    /**
     * @brief Clone LoopInfo with new expressions
     * @param expr_map map of new and old expressions
     * @return the copy
     */
    std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map) const override;

    /**
     * @brief Returns handlers of loop specific iterations
     * @return handlers
     */
    const SpecificIterationHandlers& get_handlers() const override;
    /**
     * @brief Returns original unified LoopInfo from which this LoopInfo was created
     * @return const reference of m_unified_loop_info
     */
    const std::shared_ptr<UnifiedLoopInfo>& get_unified_loop_info() const;
    /**
     * @brief Returns type of loop iterations
     * @return SpecificLoopIterType
     */
    SpecificLoopIterType get_type() const;
    /**
     * @brief Returns passes of the corresponding halder
     * @return pass pipeline
     */
    const pass::PassPipeline& get_handlers_by_type() const;

    /**
     * @brief Set m_handlers value
     * @param handlers - transformations for loop specific iterations
     */
    void set_handlers(SpecificIterationHandlers handlers) override;

    /**
     * @brief Returns dense vector with pointer increments
     * @return ref of `m_ptr_increments`
     */
    std::vector<int64_t>& get_dense_ptr_increments();
    /**
     * @brief Returns dense vector with finalization offsets
     * @return ref of `m_finalization_offsets`
     */
    std::vector<int64_t>& get_dense_finalization_offsets();
    /**
     * @brief Returns dense vector with data sizes
     * @return const ref of `m_data_sizes`
     */
    const std::vector<int64_t>& get_dense_data_sizes() const;

private:
    // ExpandedLoopInfo has LoopPorts to have opportunity to work with Loops
    // in iter handlers in InsertSpecificIterations/ For example, in UpdateSubtensors.
    // However, for faster work with data ptr shifts ExpandedLoopInfo has the separate dense attributes.
    // Note: the first initialization of these attributes is in ctor from entry and exit loop ports
    std::vector<int64_t> m_ptr_increments = {};
    std::vector<int64_t> m_finalization_offsets = {};
    std::vector<int64_t> m_data_sizes = {};

    SpecificLoopIterType m_type = {};
    std::shared_ptr<UnifiedLoopInfo> m_unified_loop_info = {};
};
using ExpandedLoopInfoPtr = std::shared_ptr<ExpandedLoopInfo>;

} // namespace lowered
} // namespace snippets
} // namespace ov
