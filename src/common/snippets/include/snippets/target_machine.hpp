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
typedef std::pair<std::function<std::shared_ptr<Emitter>(const std::shared_ptr<ov::Node>&)>,
        std::function<std::set<ov::element::TypeVector>(const std::shared_ptr<ov::Node>&)>> jitters_value;

/**
 * @interface TargetMachine
 * @brief Base class Target machine representation. Target derives from this class to provide generator information about supported emitters
 * @ingroup snippets
 */
class TargetMachine {
public:
    /**
     * @brief checks if target is natively supported
     * @return true, if supported
     */
    virtual bool is_supported() const = 0;

    /**
     * @brief finalizes code generation
     * @return generated kernel binary
     */
    virtual code get_snippet() const = 0;

    /**
     * @brief gets number of lanes supported by target's vector ISA
     * @return number of lanes
     */
    virtual size_t get_lanes() const = 0;


    /**
     * @brief called by generator to all the emitter for a target machine
     * @return a map by node's type info with callbacks to create an instance of emitter for corresponding operation type
     */
    std::function<std::shared_ptr<Emitter>(const std::shared_ptr<ov::Node>)> get(const ov::DiscreteTypeInfo& type) const;
    std::function<std::set<ov::element::TypeVector>(const std::shared_ptr<ov::Node>&)> get_supported_precisions(const ov::DiscreteTypeInfo type) const;

    /**
     * @brief checks if emitter for a specific operation is supported
     * @return true, if supported
     */
    bool has(const ov::DiscreteTypeInfo type) const;
    virtual ~TargetMachine() = default;

protected:
    std::map<const ov::DiscreteTypeInfo, jitters_value> jitters;
};

} // namespace snippets
} // namespace ov