// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public interface for target indepenent code generator.
 * @file generator.hpp
 */
#pragma once

#include <transformations_visibility.hpp>
#include "snippets_isa.hpp"

namespace ngraph {
namespace snippets {

using code = const uint8_t *;
using RegInfo = std::pair<std::vector<size_t>, std::vector<size_t>>;

TRANSFORMATIONS_API auto getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo;

/**
 * @interface Emitter
 * @brief Base class for all target specific code emitters used by generator.
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Emitter {
public:
    /**
     * @brief Default constructor
     */
    Emitter(const std::shared_ptr<ngraph::Node>& n) {
    }

    /**
     * @brief called by generator to generate code to produce target code for a specific operation
     * @param in vector of vector argument registers
     * @param out vector of vector resulting registers
     * @param pool optional vector of free vector registers which might be used inside method
     * @param gpr vector of free generam puproce registers which might be used inside method
     * @return void
     */
    virtual void emit_code(const std::vector<size_t>& in,
                           const std::vector<size_t>& out,
                           const std::vector<size_t>& pool = {},
                           const std::vector<size_t>& gpr  = {}) const = 0;

    /**
     * @brief called by generator to generate data section, if needed for a specific operation
     * @return void
     */
    virtual void emit_data() const {
    }
};

/**
 * @interface TargetMachine
 * @brief Base class Target machine representation. Target derives from this class to provide generator information about supported emittors
 * @ingroup snippets
 */
class TRANSFORMATIONS_API TargetMachine {
public:
    /**
     * @brief called by generator to all the emittors available for a target machine
     * @return a map by node's type info with callbacks to create an instance of emmitter for corresponding operation type
     */
    virtual auto getJitters() -> std::map<const ngraph::DiscreteTypeInfo, std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)>>{
        return {};
    }
};

/**
 * @interface Schedule
 * @brief Return scheduling information and pointer to generated kernel code
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Schedule {
public:
    /**
     * @brief Default constructor
     */
    Schedule() : work_size({}), is_flat(false), ptr(nullptr) {}
    /**
     * @brief Default to create schedule out of specific parameters
     * @param ws work size for kernel execution
     * @param f can this kernel be linearided to 1D range
     * @param p pointer to generated code
     */
    Schedule(const Shape& ws, bool f, code p) : work_size(ws), is_flat(f), ptr(p) {}

    Shape work_size {};
    bool is_flat {false};
    code ptr {nullptr};
};

/**
 * @interface Generator
 * @brief Target independent code generator interface
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Generator {
public:
    /**
     * @brief Default constructor
     */
    Generator() = default;
    /**
     * @brief Default destructor
     */
    virtual ~Generator() = default;
    /**
     * @brief virtual method any specific implementation should implement
     * @param f runction in canonical for for table-based code generation
     * @return pointer to generated code
     */
    virtual code generate(std::shared_ptr<Function>& f) const = 0;

protected:
    mutable std::map<const ngraph::DiscreteTypeInfo, std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)>> jitters;
};

} // namespace snippets
} // namespace ngraph