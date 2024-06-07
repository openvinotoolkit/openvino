// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cstdint>

#include "openvino/core/node.hpp"

namespace ov {
namespace snippets {

/**
 * @interface RegType
 * @brief Register type of input and output operations
 */
enum class RegType { gpr, vec, undefined };
/**
 * @interface Reg
 * @brief Register representation: type of register and index
 */
struct Reg {
    Reg() = default;
    Reg(RegType type_, size_t idx_) : type(type_), idx(idx_) {}

    RegType type = RegType::gpr;
    size_t idx = 0;

    friend bool operator==(const Reg& lhs, const Reg& rhs);
    friend bool operator!=(const Reg& lhs, const Reg& rhs);
};
using RegInfo = std::pair<std::vector<Reg>, std::vector<Reg>>;

std::string regTypeToStr(const RegType& type);

/**
 * @interface Emitter
 * @brief Base class for all target specific code emitters used by generator.
 * @ingroup snippets
 */
class Emitter {
public:
    /**
     * @brief Default constructor
     */
    Emitter() {}

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
    virtual void emit_data() const {}

    virtual ~Emitter() = default;
};

} // namespace snippets
} // namespace ov
