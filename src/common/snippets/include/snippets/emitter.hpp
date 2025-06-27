// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <utility>
#include <vector>

namespace ov::snippets {

/**
 * @interface RegType
 * @brief Register type of input and output operations
 */
enum class RegType : uint8_t {
    gpr,
    vec,
    mask,
    // Ticket: 166071
    // Need to move this type to a separate class
    address,  // address type should be ignored by the code generation logic, as it is handled outside the snippets
              // pipeline.
    undefined
};
/**
 * @interface Reg
 * @brief Register representation: type of register and index
 */
struct Reg {
    enum { UNDEFINED_IDX = std::numeric_limits<size_t>::max() };
    Reg() = default;
    Reg(RegType type_, size_t idx_) : type(type_), idx(idx_) {}

    [[nodiscard]] bool is_address() const {
        return type == RegType::address;
    }
    [[nodiscard]] bool is_defined() const {
        return is_address() || (type != RegType::undefined && idx != UNDEFINED_IDX);
    }
    RegType type = RegType::undefined;
    size_t idx = UNDEFINED_IDX;

    friend bool operator==(const Reg& lhs, const Reg& rhs);
    friend bool operator<(const Reg& lhs, const Reg& rhs);
    friend bool operator>(const Reg& lhs, const Reg& rhs);
    friend bool operator!=(const Reg& lhs, const Reg& rhs);
    friend std::ostream& operator<<(std::ostream& s, const Reg& r);
};
using RegInfo = std::pair<std::vector<Reg>, std::vector<Reg>>;

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
    Emitter() = default;

    /**
     * @brief called by generator to generate code to produce target code for a specific operation
     * @details
     *   Avoid passing default arguments to virtual function, but still allow user to call
     *   emit_code function without "pool" or "gpr"
     * @param in vector of vector argument registers
     * @param out vector of vector resulting registers
     * @param pool optional vector of free vector registers which might be used inside method
     * @param gpr vector of free general purpose registers which might be used inside method
     * @return void
     */
    void emit_code(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool = {},
                   const std::vector<size_t>& gpr = {}) const {
        emit_code_impl(in, out, pool, gpr);
    }

    /**
     * @brief called by generator to generate data section, if needed for a specific operation
     * @return void
     */
    virtual void emit_data() const {}

    virtual ~Emitter() = default;

private:
    /**
     * @brief called by generator to generate code to produce target code for a specific operation
     * @param in vector of vector argument registers
     * @param out vector of vector resulting registers
     * @param pool optional vector of free vector registers which might be used inside method
     * @param gpr vector of free general purpose registers which might be used inside method
     * @return void
     */
    virtual void emit_code_impl(const std::vector<size_t>& in,
                                const std::vector<size_t>& out,
                                const std::vector<size_t>& pool,
                                const std::vector<size_t>& gpr) const = 0;
};

}  // namespace ov::snippets
