// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"


namespace ov {
namespace snippets {
namespace lowered {

class LoopInfo;
using LoopInfoMap = std::unordered_map<const LoopInfo*, std::shared_ptr<LoopInfo>>;
using LoopInfoSet = std::unordered_set<const LoopInfo*>;
using LoopInfoPtr = std::shared_ptr<LoopInfo>;

/**
 * @interface LoopInfo
 * @brief The base class that contains the common information about a Loop in Linear Intermediate Representation (Linear IR):
 *        work amount of the Loop, step of loop counter increment, input and output ports of the Loop.
 * @ingroup snippets
 */
class LoopInfo : public std::enable_shared_from_this<LoopInfo> {
public:
    LoopInfo() = default;
    LoopInfo(size_t work_amount, size_t increment, const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits);
    LoopInfo(size_t work_amount, size_t increment, const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits);
    virtual ~LoopInfo() = default;

    /**
     * @brief Clone LoopInfo with new Expressions
     * @param expr_map map of new and old expressions
     * @param loop_map map of new and old LoopInfo.
     * @return the copy
     */
    virtual std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map, LoopInfoMap& loop_map) const = 0;

    /**
     * @brief Apply the passed function to the current LoopInfo
     * @param func function for applying
     * @param applied_loops set of already updated loops
     */
    virtual void apply(const std::function<void(const LoopInfoPtr&)>& func, LoopInfoSet& applied_loops) = 0;

    /**
     * @brief Check if some parameters of Loop are dynamic (undefined)
     * @return True if some parameters of Loop are unknown, False if all parameters are static
     */
    virtual bool is_dynamic() const;

    /**
     * @brief Returns count of input ports
     * @return count
     */
    size_t get_input_count() const;
    /**
     * @brief Returns count of output ports
     * @return count
     */
    size_t get_output_count() const;

    /**
     * @brief Returns dimension index if dimension indices for all input and output ports are equal.
     *        Otherwise returns LoopPort::UNDEFINED_DIM_IDX.
     * @return index
     */
    size_t get_dim_idx() const;
    /**
     * @brief Returns work amount of the Loop.
     * @return m_work_amount
     */
    virtual size_t get_work_amount() const;
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
    virtual void set_work_amount(size_t work_amount);
    /**
     * @brief Set m_increment value
     * @param increment - step of loop counter increment
     */
    void set_increment(size_t increment);
    /**
     * @brief Sets `dim_idx` to all input and output ports
     * @param dim_idx - index
     */
    void set_dim_idx(size_t dim_idx);

    /**
     * @brief Replace the current LoopPort `actual_port` with new `target_ports`
     * @param actual_port actual port
     * @param target_ports new ports. The ports order is important. Can contain `actual_port`
     */
    virtual void replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports);
    /**
     * @brief Replace the current LoopPort that contains ExpressionPort `actual_port` with new `target_ports`
     *        Note: If there is no LoopPort with this ExpressionPort `actual_port`, does nothing
     * @param actual_port actual port
     * @param target_ports new ports. The ports order is important. Can contain `actual_port`
     */
    virtual void replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports);

    /**
     * @brief Iterates through all loop ports and call `caller` for each of them
     * @param caller - function that called for each loop port
     */
    inline void iterate_through_ports(const std::function<void(const LoopPort&)>& caller) const {
        std::for_each(m_input_ports.cbegin(), m_input_ports.cend(), caller);
        std::for_each(m_output_ports.cbegin(), m_output_ports.cend(), caller);
    }

    /**
     * @brief Sort Loop Ports according to the execution order of underlying expressions
     */
    virtual void sort_ports() = 0;

    // Note that get_type_info_static and get_type_info are needed to mimic OPENVINO_RTTI interface,
    // so the standard OPENVINO_RTTI(...) macros could be used in derived classes.
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static ::ov::DiscreteTypeInfo type_info_static {"LoopInfoBase"};
        type_info_static.hash();
        return type_info_static;
    }

    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }

    const char* get_type_name() const {
        return get_type_info().name;
    }
    /**
     * @brief Return true if expression port is a loop port
     * @param expr_port - expression port to check
     */
    bool is_loop_port(const ExpressionPort& expr_port);
    /**
     * @brief Return loop port of an expression port
     * @param expr_port - expression port.
     */
    const LoopPort& get_loop_port(const ExpressionPort& expr_port);

protected:
    /**
     * @brief Helper to clone Loop ports using `ExpressionMap`
     * @param expr_map expression map [the current expr -> the new expr]
     * @param loop_ports the loop ports that will be cloned
     * @return vector with new cloned loop ports
     */
    static std::vector<LoopPort> clone_loop_ports(const ExpressionMap& expr_map, const std::vector<LoopPort>& loop_ports);
    /**
     * @brief Find LoopPort in input and output ports
     * @param loop_port target port
     * @return iterator of the corresponding collection
     */
    template<typename T>
    std::vector<LoopPort>::iterator find_loop_port(const T& loop_port);

    size_t m_work_amount = 0;
    size_t m_increment = 0;
    // The order of input and output expressions is important:
    //     - The position before first input expr is Loop Begin position
    //     - The position after last output expr is Loop End position
    // Note: Scalars aren't input expressions but can be before first input expr in Linear IR
    std::vector<LoopPort> m_input_ports = {};
    std::vector<LoopPort> m_output_ports = {};
};

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

        bool is_dynamic() const;
        bool is_static() const;

        friend bool operator==(const LoopPortDesc& lhs, const LoopPortDesc& rhs);
        friend bool operator!=(const LoopPortDesc& lhs, const LoopPortDesc& rhs);
    };
    // The structure describes full information about port
    // - TODO [140365] : UnifiedLoopInfo should have the map of LoopPorts and LoopDesc as class field
    //                   instead of the separate vectors with descriptors.
    struct LoopPortInfo {
        LoopPortInfo() = default;
        LoopPortInfo(LoopPort port_, LoopPortDesc desc_) : port(std::move(port_)), desc(std::move(desc_)) {}

        LoopPort port = {};
        LoopPortDesc desc = {};
    };

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
     * @brief Clone LoopInfo with new Expressions
     * @param expr_map map of new and old expressions
     * @param loop_map map of new and old LoopInfo.
     * @return the copy
     */
    std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map, LoopInfoMap& loop_map) const override;

    /**
     * @brief Apply the passed function on the current LoopInfo.
     * @param func function for applying
     * @param applied_loops set of already updated loops
     */
    void apply(const std::function<void(const LoopInfoPtr&)>& func, LoopInfoSet& applied_loops) override;

    /**
     * @brief Check if some parameters of Loop are dynamic (undefined)
     * @return True if some parameters of Loop are unknown, False if all parameters are static
     */
    bool is_dynamic() const override;

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
     * @brief Returns vector with data pointer shift params of output loop ports
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
     * @brief Sort Loop Ports according to the execution order of underlying expressions
     */
    void sort_ports() override;

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
        std::for_each(m_input_port_descs.begin(), m_input_port_descs.end(), caller);
        std::for_each(m_output_port_descs.begin(), m_output_port_descs.end(), caller);
    }
    /**
     * @brief Iterates through all loop ports and call `caller` for each of them
     * @param caller - function that called for each loop port
     */
    inline void iterate_through_descs(const std::function<void(const LoopPortDesc&)>& caller) const {
        std::for_each(m_input_port_descs.cbegin(), m_input_port_descs.cend(), caller);
        std::for_each(m_output_port_descs.cbegin(), m_output_port_descs.cend(), caller);
    }
    /**
     * @brief Iterates through all pairs <LoopPort, LoopPortDesc> and call `caller` for each of them
     * @param caller - function that called for each pair
     */
    inline void iterate_through_infos(const std::function<void(LoopPort&, LoopPortDesc&)>& caller) {
        for (size_t i = 0; i < get_input_count(); ++i)
            caller(m_input_ports[i], m_input_port_descs[i]);
        for (size_t i = 0; i < get_output_count(); ++i)
            caller(m_output_ports[i], m_output_port_descs[i]);
    }
    /**
     * @brief Iterates through all pairs <LoopPort, LoopPortDesc> and call `caller` for each of them
     * @param caller - function that called for each pair
     */
    inline void iterate_through_infos(const std::function<void(const LoopPort&, const LoopPortDesc&)>& caller) const {
        for (size_t i = 0; i < get_input_count(); ++i)
            caller(m_input_ports[i], m_input_port_descs[i]);
        for (size_t i = 0; i < get_output_count(); ++i)
            caller(m_output_ports[i], m_output_port_descs[i]);
    }

    /**
     * @brief Return loop port info of an expression port
     * @param expr_port - expression port.
     */
    LoopPortInfo get_loop_port_info(const ExpressionPort& expr_port);

protected:
    /**
     * @brief Clone LoopPortDesc[actual_port_idx] `new_count` times and insert on the place of current desc
     * @param actual_port_idx index of the current descriptor/port
     * @param new_count count of cloned descriptors
     * @param is_input true if descriptor is of input port. Otherwise, false - of output Loop port
     */
    void replace_with_cloned_descs(size_t actual_port_idx, size_t new_count, bool is_input);

    SpecificIterationHandlers m_handlers = {};
    std::vector<LoopPortDesc> m_input_port_descs = {};
    std::vector<LoopPortDesc> m_output_port_descs = {};
};
using UnifiedLoopInfoPtr = std::shared_ptr<UnifiedLoopInfo>;

/**
 * @interface InnerSplittedUnifiedLoopInfo
 * @brief The structure describes inner splitted Loop after `SplitLoops`.
 *        Contains pointer to outer splitted loop info. WorkAmount is equal to increment of outer splitted loop info.
 * @ingroup snippets
 */
class InnerSplittedUnifiedLoopInfo : public UnifiedLoopInfo {
public:
    OPENVINO_RTTI("InnerSplittedUnifiedLoopInfo", "0", UnifiedLoopInfo)

    InnerSplittedUnifiedLoopInfo() = default;
    InnerSplittedUnifiedLoopInfo(size_t increment, const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                 const std::vector<LoopPortDesc>& in_descs, const std::vector<LoopPortDesc>& out_descs,
                                 const SpecificIterationHandlers& handlers, LoopInfoPtr outer_splitted_loop_info);

    /**
     * @brief Clone LoopInfo with new Expressions
     * @param expr_map map of new and old expressions
     * @param loop_map map of new and old LoopInfo.
     *        If `loop_map` contains cloned outer splitted loop -info, we take it from there.
     *        Otherwise we manually clone it and add to this map.
     * @return the copy
     */
    std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map, LoopInfoMap& loop_map) const override;

    /**
     * @brief Apply the passed function on OuterSplittedLoopInfo and then on the current LoopInfo.
     * @param func function for applying
     * @param applied_loops set of already updated loops
     */
    void apply(const std::function<void(const LoopInfoPtr&)>& func, LoopInfoSet& applied_loops) override;

    /**
     * @brief Returns work amount of the Loop.
     * @return m_work_amount
     */
    size_t get_work_amount() const override;
    /**
     * @brief Returns OuterSplittedLoopInfo
     * @return m_outer_splitted_loop_info
     */
    LoopInfoPtr get_outer_splitted_loop_info() const;

    /**
     * @brief Set m_work_amount value
     * @param work_amount - work amount of the loop
     */
    void set_work_amount(size_t work_amount) override;
    /**
     * @brief Set m_outer_splitted_loop_info value
     * @param outer - OuterSplittedLoopInfo
     */
    void set_outer_splitted_loop_info(LoopInfoPtr outer);

private:
    LoopInfoPtr m_outer_splitted_loop_info = nullptr;
};
using InnerSplittedUnifiedLoopInfoPtr = std::shared_ptr<InnerSplittedUnifiedLoopInfo>;

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
                     SpecificLoopIterType type, UnifiedLoopInfoPtr unified_loop_info, bool evaluate_once = false);
    /**
     * @brief Clone LoopInfo with new Expressions
     * @param expr_map map of new and old expressions
     * @param loop_map map of new and old LoopInfo.
     *        If `loop_map` contains cloned unified loop -info, we take it from there.
     *        Otherwise we manually clone it and add to this map.
     * @return the copy
     */
    std::shared_ptr<LoopInfo> clone_with_new_expr(const ExpressionMap& expr_map, LoopInfoMap& loop_map) const override;

    /**
     * @brief Apply the passed function on UnifiedLoopInfo and then on the current LoopInfo.
     * @param func function for applying
     * @param applied_loops set of already updated loops
     */
    void apply(const std::function<void(const LoopInfoPtr&)>& func, LoopInfoSet& applied_loops) override;

    /**
     * @brief Check if some parameters of Loop are dynamic (undefined)
     * @return True if some parameters of Loop are unknown, False if all parameters are static
     */
    bool is_dynamic() const override;

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
     * @brief Returns True if the current Loop should be executed once
     *        Otherwise, returns False
     * @return `m_evaluance_once`
     */
    bool is_evaluate_once() const;

    /**
     * @brief Set value to `m_evaluance_once`
     * @param value - new value of `m_evaluance_once`
     */
    void set_evaluate_once(bool value);
    /**
     * @brief Update `m_ptr_increments` using copy values from `new_values`.
     *        The count of new values must be equal to the count of current increments.
     * @param new_values vector of new pointer increments
     */
    void update_ptr_increments(const std::vector<int64_t>& new_values);
    /**
     * @brief Update `m_finalization_offsets` using copy values from `new_values`.
     *        The count of new values must be equal to the count of current offsets.
     * @param new_values vector of new finalization offsets
     */
    void update_finalization_offsets(const std::vector<int64_t>& new_values);

    /**
     * @brief Replace the current LoopPort `actual_port` with new `target_ports`
     *        Attention: ExpandedLoopInfo supports only replace one port with one port!
     * @param actual_port actual port
     * @param target_ports vector with the single target port!
     */
    void replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) override;
    /**
     * @brief Replace the current LoopPort `actual_port` with new `target_ports`
     *        Note: If there is no LoopPort with this ExpressionPort `actual_port`, does nothing
     *        Attention: ExpandedLoopInfo supports only replace one port with one port!
     * @param actual_port actual port
     * @param target_ports vector with the single target port!
     */
    void replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) override;

    /**
     * @brief Sort Loop Ports according to the execution order of underlying expressions
     */
    void sort_ports() override;

private:
    std::vector<int64_t> m_ptr_increments = {};
    std::vector<int64_t> m_finalization_offsets = {};
    std::vector<int64_t> m_data_sizes = {};

    const SpecificLoopIterType m_type = {};
    std::shared_ptr<UnifiedLoopInfo> m_unified_loop_info = {};

    bool m_evaluate_once = false;
};
using ExpandedLoopInfoPtr = std::shared_ptr<ExpandedLoopInfo>;

} // namespace lowered
} // namespace snippets
} // namespace ov
