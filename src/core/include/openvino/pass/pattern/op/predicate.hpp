// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>
#include <unordered_map>

#include "openvino/core/any.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/symbol.hpp"

namespace ov::pass::pattern {

/// \brief Wrapper to uniformly store and access Matcher symbol information.
class OPENVINO_API PatternSymbolValue {
public:
    PatternSymbolValue();
    PatternSymbolValue(const std::shared_ptr<ov::Symbol>& s);
    PatternSymbolValue(const int64_t& i);
    PatternSymbolValue(const double& d);
    PatternSymbolValue(const std::vector<PatternSymbolValue>& g);

    bool is_dynamic() const;
    bool is_static() const;

    bool is_group() const;

    bool is_integer() const;
    bool is_double() const;

    int64_t i() const;
    double d() const;
    std::shared_ptr<ov::Symbol> s() const;
    std::vector<PatternSymbolValue> g() const;

private:
    bool is_valid() const;

    ov::Any m_value;
};

using PatternSymbolMap = std::unordered_map<std::string, PatternSymbolValue>;

namespace op {
using NodePredicate = std::function<bool(std::shared_ptr<Node>)>;
using ValuePredicate = std::function<bool(const Output<Node>&)>;
const std::string no_name = "no_name";

/// \brief Wrapper over different types of predicates. It is used to add restrictions to the match
/// Predicate types:
///   - Value Predicate -- function<bool(Output<Node>)>                      // most popular version of predicate
///   - Node Predicate  -- function<bool(shared_ptr<Node>)>                  // legacy version, should be used with care
///   - Symbol Predicate -- function<bool(PatternSymbolMap&,Output<Node>)>   // new version, collects / checks symbols
///
class OPENVINO_API Predicate {
public:
    Predicate();
    Predicate(nullptr_t);

    template <typename Fn, typename std::enable_if_t<std::is_invocable_r_v<bool, Fn, const Output<Node>&>>* = nullptr>
    explicit Predicate(Fn predicate, std::string name = no_name) {
        m_pred = [=](PatternSymbolMap& m, const Output<Node>& out) {
            return predicate(out);
        };
        m_name = name;
    }

    template <typename Fn,
    typename std::enable_if_t<std::is_invocable_r_v<bool, Fn, const std::shared_ptr<Node>&> &&
    !std::is_invocable_r_v<bool, Fn, const Output<Node>&>>* = nullptr>
    explicit Predicate(Fn predicate, std::string name = no_name) {
        m_pred = [=](PatternSymbolMap& m, const Output<Node>& out) {
            return predicate(out.get_node_shared_ptr());
        };
        m_name = name;
    }

    template <typename F,
    typename std::enable_if_t<std::is_invocable_r_v<bool, F, PatternSymbolMap&, Output<Node>>>* = nullptr>
    explicit Predicate(const F& predicate, const std::string& name = no_name) {
        m_pred = {predicate};
        m_name = name;
        m_requires_map = true;
    }

    bool operator()(PatternSymbolMap& m, const Output<Node>& output) const;

    template <typename T>
    bool operator()(const T& arg) const {
        OPENVINO_ASSERT(!m_requires_map,
                        "Predicate " + m_name + " called with unexpected argument: " + std::string(typeid(T).name()));
        if constexpr (std::is_convertible_v<T, Output<Node>>) {
            PatternSymbolMap dummy_map;
            return m_pred(dummy_map, arg);
        }
        OPENVINO_ASSERT(false,
                        "Predicate " + m_name + " called with unexpected argument: " + std::string(typeid(T).name()));
    }

    template <typename F>
    Predicate operator||(const F& other) const {
        return *this || Predicate(other);
    }

    template <typename F>
    Predicate operator&&(const F& other) const {
        return *this && Predicate(other);
    }

    template <typename F = Predicate>
    Predicate operator||(const Predicate& other) const {
        auto pred = m_pred, other_pred = other.m_pred;
        auto result = Predicate(
                [=](PatternSymbolMap& m, const Output<Node>& out) -> bool {
                    return pred(m, out) || other_pred(m, out);
                },
                m_name + " || " + other.m_name);
        result.m_requires_map = m_requires_map || other.m_requires_map;
        return result;
    }

    template <typename F = Predicate>
    Predicate operator&&(const Predicate& other) const {
        auto pred = m_pred, other_pred = other.m_pred;
        auto result = Predicate(
                [=](PatternSymbolMap& m, const Output<Node>& out) -> bool {
                    return pred(m, out) && other_pred(m, out);
                },
                m_name + " && " + other.m_name);
        result.m_requires_map = m_requires_map || other.m_requires_map;
        return result;
    }

private:
    bool m_requires_map = false;
    std::string m_name = no_name;

    std::function<bool(PatternSymbolMap&, const Output<Node>&)> m_pred;
};

template <
        typename Fn,
        typename std::enable_if_t<std::is_constructible_v<Predicate, Fn> && !std::is_same_v<Predicate, Fn>>* = nullptr>
Predicate operator&&(const Fn& lhs, const Predicate& rhs) {
    return Predicate(lhs) && rhs;
}

template <
        typename Fn,
        typename std::enable_if_t<std::is_constructible_v<Predicate, Fn> && !std::is_same_v<Predicate, Fn>>* = nullptr>
Predicate operator||(const Fn& lhs, const Predicate& rhs) {
    return Predicate(lhs) || rhs;
}
}  // namespace op
}  // namespace ov::pass::pattern
