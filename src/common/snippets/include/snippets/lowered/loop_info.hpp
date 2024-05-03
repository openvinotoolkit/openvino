// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"


namespace ov {
namespace snippets {
namespace lowered {

/**
 * @interface LoopInfo
 * @brief The base class that contains the common information about a Loop in Linear Intermediate Representation (Linear IR):
 *        work amount of the Loop, step of loop counter increment, input and exit ports of the Loop.
 * @ingroup snippets
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
     * @brief Returns count of input ports
     * @return count
     */
    size_t get_input_count() const;
    /**
     * @brief Returns count of exit ports
     * @return count
     */
    size_t get_output_count() const;

    /**
     * @brief Returns dimension index if dimension indices for all input and exit ports are equal.
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
     * @brief Returns vector with boolean values `is_incremented` of loop ports
     * @return vector with boolean values
     */
    std::vector<bool> get_is_incremented() const;
    /**
     * @brief Returns vector of loop input ports
     * @return m_input_ports
     */
    const std::vector<LoopPort>& get_input_ports() const;
    /**
     * @brief Returns vector of loop outputs ports
     * @return m_output_ports
     */
    const std::vector<LoopPort>& get_output_ports() const;

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
     * @brief Sets `dim_idx` to all input and exit ports
     * @param dim_idx - index
     */
    void set_dim_idx(size_t dim_idx);

    /**
     * @brief Replace the current LoopPort `actual_port` with new `target_ports`
     * @param actual_port actual port
     * @param target_ports new ports. Ther order of ports is important. Can contains `actual_port`
     */
    virtual void replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports);
    /**
     * @brief Replace the current LoopPort that contains ExpressionPort `actual_port` with new `target_ports`
     *        Note: If there is no LoopPort with this ExpressionPort `actual_port`, does nothing
     * @param actual_port actual port
     * @param target_ports new ports. Ther order of ports is important. Can contains `actual_port`
     */
    virtual void replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports);

    /**
     * @brief Iterates through all loop ports and call `caller` for each of them
     * @param caller - function that called for each loop port
     */
    inline void iterate_through_ports(const std::function<void(LoopPort&)>& caller) {
        std::for_each(m_input_ports.begin(), m_input_ports.end(), caller);
        std::for_each(m_output_ports.begin(), m_output_ports.end(), caller);
    }
    /**
     * @brief Iterates through all loop ports and call `caller` for each of them
     * @param caller - function that called for each loop port
     */
    inline void iterate_through_ports(const std::function<void(const LoopPort&)>& caller) const {
        std::for_each(m_input_ports.cbegin(), m_input_ports.cend(), caller);
        std::for_each(m_output_ports.cbegin(), m_output_ports.cend(), caller);
    }

    // Note that get_type_info_static and get_type_info are needed to mimic OPENVINO_RTTI interface,
    // so the standard OPENVINO_RTTI(...) macros could be used in derived classes.
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static ::ov::DiscreteTypeInfo type_info_static {"PassBase"};
        type_info_static.hash();
        return type_info_static;
    }

    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }

    const char* get_type_name() const {
        return get_type_info().name;
    }

protected:
    /**
     * @brief Helper to clone Loop ports using `ExpressionMap`
     * @param expr_map expression map [the current expr -> the new expr]
     * @param loop_ports the loop ports that will be cloned
     * @return vector with new cloned loop ports
     */
    static std::vector<LoopPort> clone_loop_ports(const ExpressionMap& expr_map, const std::vector<LoopPort>& loop_ports);
    /**
     * @brief Find LoopPort in input and exit ports
     * @param loop_port target port
     * @return iterator of the corresponding collection
     */
    template<typename T>
    std::vector<LoopPort>::iterator find_loop_port(const T& loop_port);

    size_t m_work_amount = 0;
    size_t m_increment = 0;
    // The order of input and exit expressions is important:
    //     - The position before first input expr is Loop Begin position
    //     - The position after last exit expr is Loop End position
    // Note: Scalars aren't input expressions but can be before first input expr in Linear IR
    std::vector<LoopPort> m_input_ports = {};
    std::vector<LoopPort> m_output_ports = {};
};
using LoopInfoPtr = std::shared_ptr<LoopInfo>;

/**
 * @interface UnifiedLoopInfo
 * @brief The structure describes unified (entire) Loop before decomposition into specific loop iterations.
 *        Contains passes for specific loop iterations that will be called for each iteration during the decomposition stage (`InsertSpecificIterations` pass).
 * @ingroup snippets
 */
class UnifiedLoopInfo : public LoopInfo {
public:
    OPENVINO_RTTI("UnifiedLoopInfo", "0", LoopInfo)
    // The structure describes data pointer shift parameters:
    // pointer increment, finalization offset, element size of the port
    struct LoopPortDesc {
        LoopPortDesc(int64_t inc = 0, int64_t fo = 0, int64_t ds = 0)
            : ptr_increment(inc), finalization_offset(fo), data_size(ds) {}
        int64_t ptr_increment = 0;
        int64_t finalization_offset = 0;
        int64_t data_size = 0;
    };
    using LoopPortInfo = std::pair<LoopPort, LoopPortDesc>;

    UnifiedLoopInfo() = default;
    UnifiedLoopInfo(size_t work_amount, size_t increment,
                    const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                    const std::vector<LoopPortDesc>& in_descs, const std::vector<LoopPortDesc>& out_descs,
                    const SpecificIterationHandlers& handlers = SpecificIterationHandlers());
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
    const SpecificIterationHandlers& get_handlers() const;
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
    /**
     * @brief Returns vector with data pointer shift params of input loop ports
     * @return vector with params
     */
    const std::vector<LoopPortDesc>& get_input_port_descs() const;
    /**
     * @brief Returns vector with data pointer shift params of exit loop ports
     * @return vector with params
     */
    const std::vector<LoopPortDesc>& get_output_port_descs() const;
    /**
     * @brief Returns vector with full LoopPort info [Port and descriptor] of input loop ports
     * @return vector with port information
     */
    std::vector<LoopPortInfo> get_input_ports_info() const;
    /**
     * @brief Returns vector with full LoopPort info [Port and descriptor] of input loop ports
     * @return vector with port information
     */
    std::vector<LoopPortInfo> get_output_ports_info() const;

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
    template <SpecificLoopIterType Type, typename T, class... Args>
    void register_pass_to_handler(Args&&... args) {
        m_handlers.register_pass<Type, T>(args...);
    }

    /**
     * @brief Sort ALL input Loop Ports by `new_order`: `m_input_ports[new_order[i]] = m_input_ports[i]`
     * @param new_order vector of new indexes
     */
    void sort_entry_ports(const std::vector<size_t>& new_order);
    /**
     * @brief Sort ALL exit Loop Ports by `new_order`: `m_output_ports[new_order[i]] = m_output_ports[i]`
     * @param new_order vector of new indexes
     */
    void sort_exit_ports(const std::vector<size_t>& new_order);

    /**
     * @brief Replace the current LoopPort `actual_port` with new `target_ports`
     * @param actual_port actual port
     * @param target_ports new ports. Ther order of ports is important. Can contains `actual_port`
     */
    void replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) override;
    /**
     * @brief Replace the current LoopPort that contains ExpressionPort `actual_port` with new `target_ports`
     *        Note: If there is no LoopPort with this ExpressionPort `actual_port`, does nothing
     * @param actual_port actual port
     * @param target_ports new ports. Ther order of ports is important. Can contains `actual_port`
     */
    void replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) override;

    /**
     * @brief Iterates through all LoopPortDesc and call `caller` for each of them
     * @param caller - function that called for each LoopPortDesc
     */
    inline void iterate_through_descs(const std::function<void(LoopPortDesc&)>& caller) {
        std::for_each(m_entry_port_descs.begin(), m_entry_port_descs.end(), caller);
        std::for_each(m_exit_port_descs.begin(), m_exit_port_descs.end(), caller);
    }
    /**
     * @brief Iterates through all loop ports and call `caller` for each of them
     * @param caller - function that called for each loop port
     */
    inline void iterate_through_descs(const std::function<void(const LoopPortDesc&)>& caller) const {
        std::for_each(m_entry_port_descs.cbegin(), m_entry_port_descs.cend(), caller);
        std::for_each(m_exit_port_descs.cbegin(), m_exit_port_descs.cend(), caller);
    }
    /**
     * @brief Iterates through all pairs <LoopPort, LoopPortDesc> and call `caller` for each of them
     * @param caller - function that called for each pair
     */
    inline void iterate_through_infos(const std::function<void(LoopPort&, LoopPortDesc&)>& caller) {
        OPENVINO_ASSERT(m_input_ports.size() == m_entry_port_descs.size(), "Incompatible count of input port and descs");
        OPENVINO_ASSERT(m_output_ports.size() == m_exit_port_descs.size(), "Incompatible count of exit port and descs");
        for (size_t i = 0; i < get_input_count(); ++i)
            caller(m_input_ports[i], m_entry_port_descs[i]);
        for (size_t i = 0; i < get_output_count(); ++i)
            caller(m_output_ports[i], m_exit_port_descs[i]);
    }
    /**
     * @brief Iterates through all pairs <LoopPort, LoopPortDesc> and call `caller` for each of them
     * @param caller - function that called for each pair
     */
    inline void iterate_through_infos(const std::function<void(const LoopPort&, const LoopPortDesc&)>& caller) const {
        OPENVINO_ASSERT(m_input_ports.size() == m_entry_port_descs.size(), "Incompatible count of input port and descs");
        OPENVINO_ASSERT(m_output_ports.size() == m_exit_port_descs.size(), "Incompatible count of exit port and descs");
        for (size_t i = 0; i < get_input_count(); ++i)
            caller(m_input_ports[i], m_entry_port_descs[i]);
        for (size_t i = 0; i < get_output_count(); ++i)
            caller(m_output_ports[i], m_exit_port_descs[i]);
    }

private:
    template<typename T>
    void replace_with_new_port_descs(const T& actual_port, size_t actual_port_idx, const std::vector<T>& target_ports, bool is_entry);

    SpecificIterationHandlers m_handlers = {};
    std::vector<LoopPortDesc> m_entry_port_descs = {};
    std::vector<LoopPortDesc> m_exit_port_descs = {};
};
using UnifiedLoopInfoPtr = std::shared_ptr<UnifiedLoopInfo>;

/**
 * @interface ExpandedLoopInfo
 * @brief The structure describes expanded Loop (specific iterations) after unified loop decomposition into specific loop iterations.
 *        Contains type of specific iteration, pointer to the original unified loop and data pointer shifts for quick recalculation.
 * @ingroup snippets
 */
class ExpandedLoopInfo : public LoopInfo {
public:
    OPENVINO_RTTI("ExpandedLoopInfo", "0", LoopInfo)
    ExpandedLoopInfo() = default;
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
     * @brief Returns passes of the corresponding handler
     * @return pass pipeline
     */
    const pass::PassPipeline& get_handler_passes() const;
    /**
     * @brief Returns dense vector with pointer increments
     * @return const ref of `m_ptr_increments`
     */
    const std::vector<int64_t>& get_ptr_increments() const;
    /**
     * @brief Returns dense vector with finalization offsets
     * @return const ref of `m_finalization_offsets`
     */
    const std::vector<int64_t>& get_finalization_offsets() const;
    /**
     * @brief Returns dense vector with data sizes
     * @return const ref of `m_data_sizes`
     */
    const std::vector<int64_t>& get_data_sizes() const;

    /**
     * @brief Replace the current LoopPort `actual_port` with new `target_ports`
     *        Attention: ExpandedLoopInfo supports only replace one port with one port!
     * @param actual_port actual port
     * @param target_ports vector with the single target port!
     */
    void replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) override;
    /**
     * @briefReplace the current LoopPort `actual_port` with new `target_ports`
     *        Note: If there is no LoopPort with this ExpressionPort `actual_port`, does nothing
     *        Attention: ExpandedLoopInfo supports only replace one port with one port!
     * @param actual_port actual port
     * @param target_ports vector with the single target port!
     */
    void replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) override;

private:
    // ExpandedLoopInfo has LoopPorts to have opportunity to work with Loops
    // in iter handlers in InsertSpecificIterations. For example, in UpdateSubtensors.
    // However, for faster work with data ptr shifts ExpandedLoopInfo has the separate dense attributes.
    // Thus, LoopPorts of ExpandedLoopInfo are interpreted as input and exit ports of specific Loop iterations.
    // All needed informations about data pointer shifts are stored in attributes below!
    // Note: the first initialization of these attributes is in ctor from input and exit loop ports
    std::vector<int64_t> m_ptr_increments = {};
    std::vector<int64_t> m_finalization_offsets = {};
    std::vector<int64_t> m_data_sizes = {};

    const SpecificLoopIterType m_type = {};
    std::shared_ptr<UnifiedLoopInfo> m_unified_loop_info = {};
};
using ExpandedLoopInfoPtr = std::shared_ptr<ExpandedLoopInfo>;

template<typename T>
void UnifiedLoopInfo::replace_with_new_port_descs(const T& actual_port, size_t actual_port_idx, const std::vector<T>& target_ports, bool is_entry) {
    auto& data_ptr_shifts = is_entry ? m_entry_port_descs : m_exit_port_descs;
    // Create LoopPortDesc for `target_ports`
    std::vector<LoopPortDesc> target_shifts(target_ports.size(), data_ptr_shifts[actual_port_idx]);
    // Update LoopPortDesc
    auto shift_it = data_ptr_shifts.erase(data_ptr_shifts.begin() + actual_port_idx);
    data_ptr_shifts.insert(shift_it, target_shifts.cbegin(), target_shifts.cend());
}

} // namespace lowered
} // namespace snippets
} // namespace ov
