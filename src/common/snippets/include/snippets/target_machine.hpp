// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public interface for target independent code generator.
 * @file generator.hpp
 */
#pragma once

#include "emitter.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov {
namespace snippets {

struct CompiledSnippet {
    virtual const uint8_t* get_code() const = 0;
    virtual size_t get_code_size() const = 0;
    virtual bool empty() const = 0;
    virtual ~CompiledSnippet() = default;
};
using CompiledSnippetPtr = std::shared_ptr<CompiledSnippet>;

typedef std::pair<std::function<std::shared_ptr<Emitter>(const lowered::ExpressionPtr&)>,
        std::function<std::set<ov::element::TypeVector>(const std::shared_ptr<ov::Node>&)>> jitters_value;

class RuntimeConfigurator;

/**
 * @interface TargetMachine
 * @brief Base class Target machine representation. Target derives from this class to provide generator information about supported emitters
 * @ingroup snippets
 */
class TargetMachine {
public:
    TargetMachine(const std::shared_ptr<RuntimeConfigurator>& c) : configurator(c) {}

    virtual ~TargetMachine() = default;

    /**
     * @brief checks if target is natively supported
     * @return true, if supported
     */
    virtual bool is_supported() const = 0;

    /**
     * @brief finalizes code generation
     * @return generated kernel binary
     */
    virtual CompiledSnippetPtr get_snippet() = 0;

    /**
     * @brief gets number of lanes supported by target's vector ISA
     * @return number of lanes
     */
    virtual size_t get_lanes() const = 0;

    /**
     * @brief Get all possible ABI argument registers.
     * The number of actually used register depends on the signature of the called binary.
     * @return vector os snippets::Reg
     */
    virtual std::vector<snippets::Reg> get_abi_arg_regs() const = 0;

    /**
     * @brief Get all available general-purpose registers.
     * Returns only registers that are not reserved for special purposes (e.g. stack pointer or instruction address).
     * @return  vector os snippets::Reg
     */
    virtual std::vector<snippets::Reg> get_gp_reg_pool() const = 0;
    /**
    * @brief Get all available vector registers.
     * Returns only registers that are not reserved for special purposes
     * @return  vector os snippets::Reg
    */
    virtual std::vector<snippets::Reg> get_vec_reg_pool() const = 0;

    /**
     * @brief called by generator to all the emitter for a target machine
     * @return a map by node's type info with callbacks to create an instance of emitter for corresponding operation type
     */
    std::function<std::shared_ptr<Emitter>(const lowered::ExpressionPtr&)> get(const ov::DiscreteTypeInfo& type) const;
    /**
     * @brief called by generator to all the emitter for a target machine
     * @return a map by node's type info with callbacks to create an set of supported precisions for corresponding operation type
     */
    std::function<std::set<ov::element::TypeVector>(const std::shared_ptr<ov::Node>&)> get_supported_precisions(const ov::DiscreteTypeInfo& type) const;

    /**
     * @brief checks if emitter for a specific operation is supported
     * @return true, if supported
     */
    bool has(const ov::DiscreteTypeInfo& type) const;

    /**
     * @brief clone the current state
     * @return shared pointer with cloned instance
     */
    virtual std::shared_ptr<TargetMachine> clone() const = 0;

    /**
     * @brief gets runtime configurator
     * @return shared pointer with runtime configurator
     */
    const std::shared_ptr<RuntimeConfigurator>& get_runtime_configurator() const;

protected:
    std::map<const ov::DiscreteTypeInfo, jitters_value> jitters;
    std::shared_ptr<RuntimeConfigurator> configurator;
};

} // namespace snippets
} // namespace ov