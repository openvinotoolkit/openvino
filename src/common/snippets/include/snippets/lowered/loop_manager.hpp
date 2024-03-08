// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>

#include "linear_ir.hpp"
#include "pass/iter_handler.hpp"
#include "pass/pass.hpp"
#include "port_descriptor.hpp"

namespace ov {
namespace snippets {
namespace lowered {

class LinearIR::LoopManager {
public:
    LoopManager() = default;

    struct LoopPort {
        LoopPort() = default;
        LoopPort(const ExpressionPort& port, bool is_incremented = true, size_t dim_idx = 0);
        std::shared_ptr<LoopPort> clone_with_new_expr(const ExpressionPtr& new_expr) const;

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
        size_t dim_idx = 0; // The numeration starts from the end (dim_idx = 0 -> is the most inner dimension)
    };

    class LoopInfo {
    public:
        enum {UNDEFINED_DIM_IDX = std::numeric_limits<size_t>::max()};
        class SpecificIterationHandlers {
        public:
            enum class HandlerType { FIRST_ITER, MAIN_BODY, LAST_ITER };
            SpecificIterationHandlers() = default;
            SpecificIterationHandlers(size_t loop_work_amount, size_t loop_increment);
            SpecificIterationHandlers(lowered::pass::PassPipeline first_iter_handlers,
                                      lowered::pass::PassPipeline main_body_handlers,
                                      lowered::pass::PassPipeline last_iter_handlers);

            const lowered::pass::PassPipeline& get_first_iter_handelrs() const;
            const lowered::pass::PassPipeline& get_main_iter_handelrs() const;
            const lowered::pass::PassPipeline& get_last_iter_handelrs() const;
            static SpecificIterationHandlers merge_handlers(const SpecificIterationHandlers& lhs, const SpecificIterationHandlers& rhs);

            template <HandlerType Type,
                      typename T,
                      class... Args,
                      typename std::enable_if<Type == HandlerType::FIRST_ITER, bool>::type = true>
            void register_handler(Args&&... args) {
                m_first_iter_handlers.register_pass<T>(args...);
            }

            template <HandlerType Type,
                      typename T,
                      class... Args,
                      typename std::enable_if<Type == HandlerType::MAIN_BODY, bool>::type = true>
            void register_handler(Args&&... args) {
                m_main_body_handlers.register_pass<T>(args...);
            }

            template <HandlerType Type,
                      typename T,
                      class... Args,
                      typename std::enable_if<Type == HandlerType::LAST_ITER, bool>::type = true>
            void register_handler(Args&&... args) {
                m_last_iter_handlers.register_pass<T>(args...);
            }

        private:
            lowered::pass::PassPipeline m_first_iter_handlers;
            lowered::pass::PassPipeline m_main_body_handlers;
            lowered::pass::PassPipeline m_last_iter_handlers;
        };

        LoopInfo() = default;
        LoopInfo(size_t work_amount, size_t increment,
                 const std::vector<LoopPort>& entries,
                 const std::vector<LoopPort>& exits,
                 const SpecificIterationHandlers& handlers = SpecificIterationHandlers());
        LoopInfo(size_t work_amount, size_t increment,
                 const std::vector<ExpressionPort>& entries,
                 const std::vector<ExpressionPort>& exits,
                 const SpecificIterationHandlers& handlers = SpecificIterationHandlers());

        std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map) const;

        // Returns dimension index if dimension indices for all entry and exit points are equal, and UNDEFINED_DIM_IDX otherwise
        size_t get_dim_idx() const;
        size_t get_work_amount() const;
        size_t get_increment() const;
        const std::vector<LoopPort>& get_entry_points() const;
        const std::vector<LoopPort>& get_exit_points() const;
        const SpecificIterationHandlers& get_handlers() const;

        // Sets dim_idx to all entry and exit points
        void set_dim_idx(size_t dim_idx);
        void set_work_amount(size_t work_amount);
        void set_increment(size_t increment);
        void set_entry_points(std::vector<LoopPort> entry_points);
        void set_exit_points(std::vector<LoopPort> exit_points);
        void set_handlers(SpecificIterationHandlers handlers);

        template <SpecificIterationHandlers::HandlerType Type, typename T, class... Args>
        void register_handler(Args&&... args) {
            m_handlers.register_handler<Type, T>(args...);
        }

        // Update the parameters of existing LoopPorts
        void update_entry_points(const std::function<void(LoopPort&)>& updater);
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

    /**
     * @brief Clone Loop Manager with new expressions
     * @param expr_map map of new and old expressions
     * @return the copy
     */
    std::shared_ptr<LoopManager> clone_with_new_expr(const ExpressionMap& expr_map) const;

    /**
     * @brief Get target Loop Info
     * @param index loop ID
     * @return the LoopInfo shared ptr
     */
    LoopInfoPtr get_loop_info(size_t index) const;
    /**
     * @brief Get count of loops
     * @return count of loops in the map
     */
    size_t get_loop_count() const { return m_map.size(); }
    /**
     * @brief Get loop map [loop_id -> loop info]
     * @return loop map
     */
    const std::map<size_t, LoopInfoPtr>& get_map() const;
    /**
     * @brief Get loop IDs of expression that are outer (upper) than `loop_id`
     * @param expr the target expression
     * @param loop_id the target loop ID
     * @return vector of outer loop IDs
     */
    static std::vector<size_t> get_outer_expr_loops(const ExpressionPtr& expr, size_t loop_id);
    /**
     * @brief Get loop IDs of expression that are common outer (upper) than `loop_id`
     * @param lhs the first expression
     * @param rhs the second expression
     * @return vector of common outer loop IDs
     */
    static std::vector<size_t> get_common_outer_loops(const ExpressionPtr& lhs, const ExpressionPtr& rhs);
    /**
     * @brief Get common outer loop IDs of expression set
     * @param exprs vector of expressions
     * @return vector of common outer loop IDs
     */
    static std::vector<size_t> get_common_outer_loops(const std::vector<ExpressionPtr>& exprs);

    /**
     * @brief Create new LoopInfo and mark all expressions in loop bounds by new loop ID
     * @param loop_begin_pos the first expression iterator
     * @param loop_end_pos the next iterator after last expression
     * @param loop_depth count of loops for marking
     * @param vector_size increment of loop
     */
    void mark_loop(LinearIR::constExprIt loop_begin_pos,
                   LinearIR::constExprIt loop_end_pos,
                   size_t loop_depth, size_t vector_size);
    /**
     * @brief Create new LoopInfo and mark all expressions in loop bounds by new loop ID with more manual parameters
     * @param loop_begin_pos the first expression iterator
     * @param loop_end_pos the next iterator after last expression
     * @param work_amount work amount of the loop
     * @param increment the step of loop counter increment
     * @param dim_idx loop iterates by this index of dimension
     * @param entries input loop ports
     * @param exits output loop ports
     * @param set_default_handlers flag defines whether it is needed to set default set of SpecificIterationHandlers or not
     * @return new loop ID
     */
    template <typename T>
    size_t mark_loop(LinearIR::constExprIt loop_begin_pos,
                     LinearIR::constExprIt loop_end_pos,
                     size_t work_amount,
                     size_t increment,
                     size_t dim_idx,
                     const std::vector<T>& entries,
                     const std::vector<T>& exits,
                     bool set_default_handlers = true) {
        const auto normalized_increment = std::min(increment, work_amount);
        const auto handlers = set_default_handlers
                                  ? LoopInfo::SpecificIterationHandlers(work_amount, normalized_increment)
                                  : LoopInfo::SpecificIterationHandlers();
        const auto loop_info = std::make_shared<LoopManager::LoopInfo>(work_amount, normalized_increment, entries, exits, handlers);
        loop_info->set_dim_idx(dim_idx);
        const auto loop_id = this->add_loop_info(loop_info);
        for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
            insert_loop_id(*expr_it, loop_id);
        }
        return loop_id;
    }
    /**
     * @brief Create new LoopInfo and mark all expressions in loop bounds by new loop ID with more manual parameters
     * @param loop_begin_pos the first expression iterator
     * @param loop_end_pos the next iterator after last expression
     * @param work_amount work amount of the loop
     * @param increment the step of loop counter increment
     * @param entries input loop ports
     * @param exits output loop ports
     * @param set_default_handlers flag defines whether it is needed to set default set of SpecificIterationHandlers or not
     * @return new loop ID
     */
    template <typename T>
    size_t mark_loop(LinearIR::constExprIt loop_begin_pos,
                     LinearIR::constExprIt loop_end_pos,
                     size_t work_amount,
                     size_t increment,
                     const std::vector<T>& entries,
                     const std::vector<T>& exits,
                     bool set_default_handlers = true) {
        const auto normalized_increment = std::min(increment, work_amount);
        const auto handlers = set_default_handlers
                                  ? LoopInfo::SpecificIterationHandlers(work_amount, normalized_increment)
                                  : LoopInfo::SpecificIterationHandlers();
        const auto loop_info = std::make_shared<LoopManager::LoopInfo>(work_amount, normalized_increment, entries, exits, handlers);
        const auto loop_id = this->add_loop_info(loop_info);
        for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
            insert_loop_id(*expr_it, loop_id);
        }
        return loop_id;
    }
    /**
     * @brief Create new Loop and replace with it: create new LoopInfo, update loop IDs in expressions and
     *        remove the old LoopInfo from the map if no one expression isn't mark by this `old_id`
     * @param linear_ir linear IR
     * @param loop_begin_pos the first expression iterator. Must be the iterator of the `linear_ir`.
     *                       If there is explicit LoopBegin expressions in IR, loop_begin_pos must be the iterator of the LoopBegin
     * @param loop_end_pos the next iterator after last expression iterator. Must be the iterator of the `linear_ir`.
     *                     If there is explicit LoopEnd expressions in IR, loop_end_pos must be the next iterator of the LoopEnd
     * @param work_amount work amount of the loop
     * @param work_amount_increment increment of the loop
     * @param entries input loop ports
     * @param exits output loop ports
     * @param old_id loop ID of the previos loop
     * @return new loop ID
     */
    size_t replace_with_new_loop(const LinearIR& linear_ir,
                                 LinearIR::constExprIt loop_begin_pos,
                                 LinearIR::constExprIt loop_end_pos,
                                 size_t work_amount,
                                 size_t increment,
                                 const std::vector<LoopPort>& entries,
                                 const std::vector<LoopPort>& exits,
                                 const size_t old_id);
    /**
     * @brief Fuse two LoopInfos: fuse their informations to the one
     * @param linear_ir linear IR
     * @param loop_id_upper Loop ID of the Loop that is earlier in the linear IR
     * @param loop_id_lower Loop ID of the Loop that is later in the linear IR
     * @param fuse_into_upper True if the Loop with `loop_id_upper` ID is left and the Loop with `loop_id_lower` is removed
     */
    void fuse_loops(const LinearIR& linear_ir, size_t loop_id_upper, size_t loop_id_lower, bool fuse_into_upper = true);
    /**
     * @brief Fuse two loops: fuse their LoopInfos and update loop IDs in expressions
     * @param loop_begin_target the first expression iterator of the Loop that will be left after fusion
     * @param loop_end_target the next ietartor for the last expression of the Loop that will be left after fusion
     * @param loop_id_upper Loop ID of the Loop that is earlier in the linear IR
     * @param loop_id_lower Loop ID of the Loop that is later in the linear IR
     * @param fuse_into_upper True if the Loop with `loop_id_upper` ID is left and the Loop with `loop_id_lower` is removed
     */
    void fuse_loops(LinearIR::constExprIt loop_begin_target, LinearIR::constExprIt loop_end_target,
                    size_t loop_id_upper, size_t loop_id_lower, bool fuse_into_upper = true);
    /**
     * @brief Update Loop ports for one Loop. The method saves the order of ports since
     *        the order of expression defines Loop bounds before explicit loop insertion (the most first and the most last expressions).
     *        Note:
     *         - Update LoopPort - insert new loop target ports instead of existing.
     *         - Update ExpressionPort in the LoopPort - with saving of port parameters. It's softer method since ExpressionPort may not be port of Loop
     * @param loop_id the target Loop ID
     * @param actual_port the current port
     * @param target_ports vector of the new ports (the order is important!)
     * @param is_entry True if these ports are input, otherwise - output
     */
    template<typename T>
    void update_loop_port(size_t loop_id, const T& actual_port, const std::vector<T>& target_ports, bool is_entry = true);
    /**
     * @brief Update Loop ports for several Loops.
     * @param loop_ids the target Loop IDs
     * @param actual_port the current port
     * @param target_ports vector of the new ports (the order is important!)
     * @param is_entry True if these ports are input, otherwise - output
     */
    template<typename T>
    void update_loops_port(const std::vector<size_t>& loop_ids, const T& actual_port,
                           const std::vector<T>& target_ports, bool is_entry = true) {
        for (auto loop_id : loop_ids) {
            update_loop_port(loop_id, actual_port, target_ports, is_entry);
        }
    }
    /**
     * @brief The method checks the loops (LoopInfo) that the target expression is marked with and update the corresponding loop ports if needed:
     *           - If parent of the target expression and this expression are marked by one Loop and the parent is an exit port of this Loop,
     *             the method replace parent output port with the target expression output ports as new exit LoopPorts.
     *             If there are other consumers of parent output port that are not by the same Loop (like in the example below),
     *             the method just adds inserted expression output ports to existing parent output port as new exit LoopPorts.
     *                     Parent [1, 0]
     *                    /              \                        <- Adds the target expression outputs to the existing LoopPort (parent output) of Loop[1]
     *               Another expr [2]   Target expression [1, 3]     (If Another expr is marked by Loop [1] too, the method will replace loop ports (not add))
     *           - If the target expression and its consumers have the same outer loop ids and some of consumers are entry ports of these Loops,
     *             the method just replace the existing entry loop ports (that contains consumer input ports) with the target expression input ports.
     * @param expr the target expression
     */
    void update_loop_ports(const ExpressionPtr& expr);
    /**
     * @brief Sort Loop Ports by expression locations in Linear IR
     * @param loop_begin_pos the first expression iterator of the Loop
     * @param loop_end_pos the next iterator after the last expression
     * @param loop_id target Loop ID
     */
    void sort_loop_ports(LinearIR::constExprIt& loop_begin_pos, LinearIR::constExprIt& loop_end_pos, size_t loop_id);
    /**
     * @brief When the previous expression was replaced with new expressions (decomposition), the method updates the corresponding Loop.
     *        If ports of decomposed expression were the Loop ports, these Loop ports may be updated by parameters `entries` and `exits`
     * @param new_expr_begin the first expression iterator
     * @param new_expr_end the next iterator after the last expression
     * @param decomposed_expr the expression that is decomposed into several other exprs
     * @param loop_id the new expression will be marked by this loop ID
     * @param new_entries new input loop ports
     * @param new_exits new output loop ports
     */
    void expression_replacement(constExprIt new_expr_begin, constExprIt new_expr_end, const ExpressionPtr& decomposed_expr,
                                size_t loop_id, const std::vector<ExpressionPort>& new_entries, const std::vector<ExpressionPort>& exits);
    /**
     * @brief Find bounds of Loop:
     *        - If the explicit Loop exprs with the target `loop_id` have been inserted,
     *          Loop bounds are these iterators of the corresponding LoopBegin and LoopEnd.
     *        - Otherwise Loop bounds are iterators of the first entry loop port (or Scalar, VectorBuffer and another LoopBegin that
     *          are in this Loop but have another `loop_id`) and the next iterator of the last exit loop port (or another LoopEnd that
     *          are in this Loop but have another `loop_id`).
     * @param linear_ir linear IR
     * @param loop_id target Loop ID
     * @return the pair of loop_begin_pos and loop_end_pos iterators
     */
    std::pair<LinearIR::constExprIt, LinearIR::constExprIt> get_loop_bounds(const LinearIR& linear_ir, size_t loop_id) const;
    /**
     * @brief Find bounds of Loop:
     *        - If the explicit Loop exprs with the target `loop_id` have been inserted,
     *          Loop bounds are these iterators of the corresponding LoopBegin and LoopEnd.
     *        - Otherwise Loop bounds are iterators of the first entry loop port (or Scalar, VectorBuffer and another LoopBegin that
     *          are in this Loop but have another `loop_id`) and the next iterator of the last exit loop port (or another LoopEnd that
     *          are in this Loop but have another `loop_id`).
     * @param linear_ir linear IR
     * @param loop_id target Loop ID
     * @param entries input loop ports
     * @param exits output loop ports
     * @return the pair of loop_begin_pos and loop_end_pos iterators
     */
    static std::pair<LinearIR::constExprIt, LinearIR::constExprIt> get_loop_bounds(const LinearIR& linear_ir, size_t loop_id,
                                                                                   const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits);

    /**
     * @brief Get LoopPort by ExpressionPort
     * @param expr_port the target ExpressionPort
     * @param loop_id target Loop ID
     * @return loop port
     */
    LoopPort get_loop_port_by_expr_port(const ExpressionPort& expr_port, const size_t loop_id);

private:
    /**
     * @brief Add new Loop Info to the map
     * @param loop target loop info
     * @return the loop ID
     */
    size_t add_loop_info(const LoopInfoPtr& loop);
    /**
     * @brief Remove LoopInfo from the map
     * @param index the target index of Loop
     */
    void remove_loop_info(size_t index);
    /**
     * @brief Find expression ports in bounds that are connected to consumers or parent that aren't in these bounds
     * @param loop_begin_pos the first expression iterator of the Loop
     * @param loop_end_pos the next iterator after the last expression
     * @param entries found input expression ports
     * @param exits found output expression ports
     */
    static void get_io_loop_ports(LinearIR::constExprIt loop_begin_pos,
                                  LinearIR::constExprIt loop_end_pos,
                                  std::vector<ExpressionPort>& entries,
                                  std::vector<ExpressionPort>& exits);
    /**
     * @brief Fuse exit LoopPorts of upper Loop and entry LoopPorts of lower Loop
     * @param exit_points ref of exit LoopPorts of upper Loop
     * @param entry_points ref of entry LoopPorts of lower Loop
     * @param loop_id ID of the new Loop after fusion
     */
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
    static bool is_loop_id_found(const ExpressionPtr& expr, size_t id);

    std::map<size_t, LoopInfoPtr> m_map = {};
    size_t next_id = 0;
};

} // namespace lowered
} // namespace snippets
} // namespace ov
