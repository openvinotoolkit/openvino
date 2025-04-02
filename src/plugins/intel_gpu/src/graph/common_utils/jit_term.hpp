// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"

namespace ov::intel_gpu {

template <typename T>
std::string to_code_string(T val) {
    std::stringstream ss;
    ss.imbue(std::locale("C"));
    ss << val;
    return ss.str();
}

class JitTerm {
public:
    JitTerm() = default;
    template <typename T,
              std::enable_if_t<!std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view> && !std::is_same_v<T, const char*>, bool> = true>
    explicit JitTerm(const T& v) : text(to_code_string(v)) {}

    explicit JitTerm(std::string v) : text(std::move(v)) {}

    [[nodiscard]] const std::string& str() const {
        return text;
    }
    [[nodiscard]] JitTerm gt(const JitTerm& rhs) const {
        return JitTerm{"(" + text + ">" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm ge(const JitTerm& rhs) const {
        return JitTerm{"(" + text + ">=" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm le(const JitTerm& rhs) const {
        return JitTerm{"(" + text + "<=" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm lt(const JitTerm& rhs) const {
        return JitTerm{"(" + text + "<" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm eq(const JitTerm& rhs) const {
        return JitTerm{"(" + text + "==" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm ne(const JitTerm& rhs) const {
        return JitTerm{"(" + text + "!=" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm assign(const JitTerm& rhs) const {
        return JitTerm{text + " = " + rhs.str()};
    }
    [[nodiscard]] JitTerm body(const JitTerm& rhs) const {
        return JitTerm{text + "{\n" + rhs.str() + "\n}"};
    }

    template <typename... Args>
    JitTerm operator()(Args&&... args) const {
        return JitTerm{text + "(" + concat(",", std::forward<Args>(args)...).str() + ")"};
    }

    JitTerm operator[](const JitTerm& idx) const {
        return JitTerm{text + "[" + idx.str() + "]"};
    }
    JitTerm operator[](size_t idx) const {
        return JitTerm{text + "[" + to_code_string(idx) + "]"};
    }

    template <typename T1, typename... Args>
    [[nodiscard]] static JitTerm concat(const std::string& separator, const T1& first, const Args&... args) {
        std::ostringstream oss;
        oss << first;
        ((oss << separator << args), ...);
        return JitTerm{oss.str()};
    }

private:
    std::string text;
};

template <typename... Args>
inline JitTerm concat(Args&&... args) {
    return JitTerm::concat("", std::forward<Args>(args)...);
}

inline std::ostream& operator<<(std::ostream& os, const JitTerm& t) {
    return os << t.str();
}

inline bool is_number(const JitTerm& s) {
    return !s.str().empty() && std::all_of(s.str().begin(), s.str().end(), ::isdigit);
}
template <typename T>
inline T as_number(const JitTerm& s) {
    T val;
    std::stringstream ss(s.str());
    ss >> val;
    return val;
}

inline JitTerm neg(const JitTerm& arg) {
    return JitTerm{"(-" + arg.str() + ")"};
}
inline JitTerm operator+(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0") {
        return rhs;
    }
    if (rhs.str() == "0") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) + as_number<int64_t>(rhs))};
    }

    return JitTerm{"(" + lhs.str() + " + " + rhs.str() + ")"};
}

inline JitTerm operator-(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0") {
        return neg(rhs);
    }
    if (rhs.str() == "0") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) - as_number<int64_t>(rhs))};
    }

    return JitTerm{"(" + lhs.str() + " - " + rhs.str() + ")"};
}

inline JitTerm operator*(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0" || rhs.str() == "0") {
        return JitTerm{"0"};
    }
    if (lhs.str() == "1") {
        return rhs;
    }
    if (rhs.str() == "1") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) * as_number<int64_t>(rhs))};
    }
    return JitTerm{"(" + lhs.str() + " * " + rhs.str() + ")"};
}
inline JitTerm operator/(const JitTerm& lhs, const JitTerm& rhs) {
    OPENVINO_ASSERT(rhs.str() != "0");
    if (rhs.str() == "1") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) / as_number<int64_t>(rhs))};
    }
    return JitTerm{"(" + lhs.str() + " / " + rhs.str() + ")"};
}
inline JitTerm operator%(const JitTerm& lhs, const JitTerm& rhs) {
    OPENVINO_ASSERT(rhs.str() != "0");
    if (rhs.str() == "1") {
        return JitTerm{"0"};
    }

    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) % as_number<int64_t>(rhs))};
    }

    return JitTerm{"(" + lhs.str() + " % " + rhs.str() + ")"};
}
inline JitTerm operator++(JitTerm& t, int) {
    return JitTerm{t.str() + "++"};
}
inline JitTerm operator--(JitTerm& t, int) {
    return JitTerm{t.str() + "--"};
}
inline JitTerm operator-=(const JitTerm& a, const JitTerm& b) {
    return concat(a, " -= ", b);
}

inline JitTerm ternary(const JitTerm& condition, const JitTerm& true_expr, const JitTerm& false_expr) {
    return JitTerm{"(" + condition.str() + " ? " + true_expr.str() + " : " + false_expr.str() + ")"};
}
inline JitTerm isinf(const JitTerm& arg) {
    return JitTerm{"isinf(" + arg.str() + ")"};
}
inline JitTerm exp(const JitTerm& arg) {
    return JitTerm{"exp(" + arg.str() + ")"};
}
inline JitTerm erf(const JitTerm& arg) {
    return JitTerm{"erf(" + arg.str() + ")"};
}
inline JitTerm sin(const JitTerm& arg) {
    return JitTerm{"sin(" + arg.str() + ")"};
}
inline JitTerm asin(const JitTerm& arg) {
    return JitTerm{"asin(" + arg.str() + ")"};
}
inline JitTerm sinh(const JitTerm& arg) {
    return JitTerm{"sinh(" + arg.str() + ")"};
}
inline JitTerm asinh(const JitTerm& arg) {
    return JitTerm{"asinh(" + arg.str() + ")"};
}
inline JitTerm cos(const JitTerm& arg) {
    return JitTerm{"cos(" + arg.str() + ")"};
}
inline JitTerm acos(const JitTerm& arg) {
    return JitTerm{"acos(" + arg.str() + ")"};
}
inline JitTerm cosh(const JitTerm& arg) {
    return JitTerm{"cosh(" + arg.str() + ")"};
}
inline JitTerm acosh(const JitTerm& arg) {
    return JitTerm{"acosh(" + arg.str() + ")"};
}
inline JitTerm tan(const JitTerm& arg) {
    return JitTerm{"tan(" + arg.str() + ")"};
}
inline JitTerm atan(const JitTerm& arg) {
    return JitTerm{"atan(" + arg.str() + ")"};
}
inline JitTerm tanh(const JitTerm& arg) {
    return JitTerm{"tanh(" + arg.str() + ")"};
}
inline JitTerm atanh(const JitTerm& arg) {
    return JitTerm{"atanh(" + arg.str() + ")"};
}
inline JitTerm log(const JitTerm& arg) {
    return JitTerm{"log(" + arg.str() + ")"};
}
inline JitTerm log2(const JitTerm& arg) {
    return JitTerm{"log2(" + arg.str() + ")"};
}
inline JitTerm round(const JitTerm& arg) {
    return JitTerm{"round(" + arg.str() + ")"};
}
inline JitTerm rint(const JitTerm& arg) {
    return JitTerm{"rint(" + arg.str() + ")"};
}
inline JitTerm floor(const JitTerm& arg) {
    return JitTerm{"floor(" + arg.str() + ")"};
}
inline JitTerm ceil(const JitTerm& arg) {
    return JitTerm{"ceil(" + arg.str() + ")"};
}
inline JitTerm sqrt(const JitTerm& arg) {
    return JitTerm{"sqrt(" + arg.str() + ")"};
}
inline JitTerm abs(const JitTerm& arg) {
    return JitTerm{"abs(" + arg.str() + ")"};
}
inline JitTerm fabs(const JitTerm& arg) {
    return JitTerm{"fabs(" + arg.str() + ")"};
}
inline JitTerm pow(const JitTerm& arg, const JitTerm& power) {
    return JitTerm{"pow(" + arg.str() + "," + power.str() + ")"};
}
inline JitTerm logical_and(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"(" + lhs.str() + " && " + rhs.str() + ")"};
}
inline JitTerm logical_or(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"(" + lhs.str() + " || " + rhs.str() + ")"};
}
inline JitTerm max(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"max(" + lhs.str() + ", " + rhs.str() + ")"};
}
inline JitTerm min(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"min(" + lhs.str() + ", " + rhs.str() + ")"};
}
inline JitTerm clamp(const JitTerm& val, const JitTerm& low, const JitTerm& high) {
    return JitTerm{"clamp(" + val.str() + ", " + low.str() + ", " + high.str() + ")"};
}
inline JitTerm for_loop(const JitTerm& init, const JitTerm& condition, const JitTerm& expression) {
    const JitTerm _for("for");
    return _for(JitTerm::concat("; ", init, condition, expression));
}
inline JitTerm operator"" _jit(const char* str, size_t /*unused*/) {
    return JitTerm{str};
}

}  // namespace ov::intel_gpu
