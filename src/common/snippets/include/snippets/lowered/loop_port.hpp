// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <type_traits>

#include "openvino/core/except.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::lowered {

/* The class describes port of Loop: expression port that connected to Expressions from other Loops.
 */
class LoopPort {
public:
    static constexpr auto UNDEFINED_DIM_IDX = std::numeric_limits<size_t>::max();
    enum class Type : uint8_t {
        Incremented,     // Loop port which data ptr should be incremented after each Loop iteration
        NotIncremented,  // Loop port which data ptr should not be incremented (for example, to avoid double increment)
        NotProcessed,    // LoopPort which doesn't process the dim by `dim_idx` (UNDEFINED_DIM_IDX) and is used only for
                         // Loop bound definition
    };

    LoopPort() = default;

    template <LoopPort::Type T,
              std::enable_if_t<utils::any_of(T, Type::Incremented, Type::NotIncremented), bool> = true>
    static LoopPort create(const ExpressionPort& port, size_t dim_idx = 0) {
        return {port, dim_idx, T};
    }

    template <LoopPort::Type T, std::enable_if_t<T == Type::NotProcessed, bool> = true>
    static LoopPort create(const ExpressionPort& port) {
        return {port, UNDEFINED_DIM_IDX, Type::NotProcessed};
    }

    [[nodiscard]] std::shared_ptr<LoopPort> clone_with_new_expr(const ExpressionPtr& new_expr) const;

    friend bool operator==(const LoopPort& lhs, const LoopPort& rhs);
    friend bool operator!=(const LoopPort& lhs, const LoopPort& rhs);
    friend bool operator<(const LoopPort& lhs, const LoopPort& rhs);

    [[nodiscard]] const std::shared_ptr<ExpressionPort>& get_expr_port() const {
        return m_expr_port;
    }
    [[nodiscard]] Type get_type() const {
        return m_type;
    }
    [[nodiscard]] size_t get_dim_idx() const;

    void set_expr_port(std::shared_ptr<ExpressionPort> p);
    void set_dim_idx(size_t idx);

    template <LoopPort::Type T,
              std::enable_if_t<utils::any_of(T, Type::Incremented, Type::NotIncremented), bool> = true>
    void convert_to_type() {
        OPENVINO_ASSERT(is_processed(), "NotProcessed LoopPort cannot change type!");
        m_type = T;
    }

    [[nodiscard]] bool is_processed() const;
    [[nodiscard]] bool is_incremented() const;

private:
    LoopPort(const ExpressionPort& port, size_t dim_idx, Type type);

    std::shared_ptr<ExpressionPort> m_expr_port;
    size_t m_dim_idx = 0;  // The numeration starts from the end (dim_idx = 0 -> is the most inner dimension)
    Type m_type = Type::Incremented;
};

std::ostream& operator<<(std::ostream& out, const LoopPort::Type& type);

}  // namespace ov::snippets::lowered
