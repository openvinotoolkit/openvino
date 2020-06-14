// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <utility>

#include "error.hpp"
#include <vpu/utils/optional.hpp>
#include <vpu/utils/small_vector.hpp>

//
// Simple integer arithmetics to be used for the work sizes calculation.
// Supported operations : +,-,*,/,%,(,)
// no unary -,+
// Variables defined as single chars and should not include one of the ops, whitespaces or 0-9
//

namespace vpu {

template <typename T>
Optional<int> parseNumber(const std::string& s) {
    auto value = T{};
    if ((std::istringstream(s) >> value >> std::ws).eof()) {
        return {value};
    }
    return {};
}

namespace details {

#define OPERATOR(OP)                                                           \
  IntOrFloat operator OP(const IntOrFloat &other) const {                      \
    if (isInt && other.isInt) {                                                \
      return IntOrFloat{value.i OP other.value.i};                             \
    }                                                                          \
    const float lhs = isInt ? value.i : value.f;                               \
    const float rhs = other.isInt ? other.value.i : other.value.f;             \
    return IntOrFloat{lhs OP rhs};                                             \
  }

class IntOrFloat final {
    union {
        int i;
        float f;
    } value{};
    bool isInt = true;

public:
    explicit IntOrFloat(int x) : isInt{true} {
        value.i = x;
    }
    explicit IntOrFloat(float x) : isInt{false} {
        value.f = x;
    }
    explicit IntOrFloat(const std::string& x) {
        const auto integer = parseNumber<int>(x);
        if (integer.hasValue()) {
            *this = IntOrFloat(integer.get());
            return;
        }
        const auto fp = parseNumber<float>(x);
        if (fp.hasValue()) {
            *this = IntOrFloat(fp.get());
            return;
        }
        VPU_THROW_FORMAT("Failed to convert string to number: '%s'", x);
    }

    explicit operator std::string() const {
        return isInt ? std::to_string(value.i) : std::to_string(value.f);
    }

    float toFloat() const { return isInt ? static_cast<float>(value.i) : value.f; }

    OPERATOR(+)
    OPERATOR(-)
    OPERATOR(*)
    OPERATOR(/)

    IntOrFloat operator %(const IntOrFloat & other) const {
        if (isInt && other.isInt) {
            return IntOrFloat{value.i % other.value.i};
        }
        THROW_IE_EXCEPTION << "Can't apply modulus operation to floating point value";
    }
};

} // namespace details

class MathExpression final {
public:
    void setVariables(const std::map<std::string, std::string>& variables) {
        for (const auto& var : variables) {
            // if string converts to float, it also will be able to convert to int
            if (parseNumber<float>(var.second).hasValue()) {
                _vars.emplace(var.first, details::IntOrFloat{var.second});
            }
        }
    }

    void parse(const std::string& expression);
    int evaluate() const;

private:
    enum class TokenType {
        Value,
        Operator,
        Function
    };

    struct Token {
        TokenType type;
        details::IntOrFloat value;
        std::string opName;

        explicit Token(TokenType type, details::IntOrFloat value, std::string name)
            : type(type), value(value), opName(std::move(name)) {}
    };

    std::map<std::string, details::IntOrFloat> _vars;
    SmallVector<Token> _parsedTokens;
};

}  // namespace vpu
