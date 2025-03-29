// Copyright (C) 2018-2025 Intel Corporation
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
    PatternSymbolValue(int64_t i);
    PatternSymbolValue(double d);
    PatternSymbolValue(const std::vector<PatternSymbolValue>& g);

    bool is_dynamic() const;
    bool is_static() const;

    bool is_group() const;

    bool is_integer() const;
    bool is_double() const;

    int64_t i() const;
    double d() const;
    std::shared_ptr<ov::Symbol> s() const;
    const std::vector<PatternSymbolValue>& g() const;

    bool operator==(const PatternSymbolValue& other) const;

private:
    bool is_valid() const;

    ov::Any m_value;
};

using PatternSymbolMap = std::unordered_map<std::string, PatternSymbolValue>;

namespace op {
using NodePredicate = std::function<bool(std::shared_ptr<Node>)>;
using ValuePredicate = std::function<bool(const Output<Node>&)>;

/// \brief Wrapper over different types of predicates. It is used to add restrictions to the match
/// Predicate types:
///   - Value Predicate -- function<bool(Output<Node>)>                      // most popular version of predicate
///   - Node Predicate  -- function<bool(shared_ptr<Node>)>                  // legacy version, should be used with care
///   - Symbol Predicate -- function<bool(PatternSymbolMap&,Output<Node>)>   // new version, collects / checks symbols
///
class OPENVINO_API Predicate {
public:
    Predicate();
    Predicate(std::nullptr_t);

    template <typename TPredicate,
              typename std::enable_if_t<std::is_invocable_r_v<bool, TPredicate, const Output<Node>&>>* = nullptr>
    explicit Predicate(const TPredicate& predicate) {
        m_pred = [=](PatternSymbolMap&, const Output<Node>& out) {
            return predicate(out);
        };
    }

    template <typename TPredicate,
              typename std::enable_if_t<std::is_invocable_r_v<bool, TPredicate, const std::shared_ptr<Node>&> &&
                                        !std::is_invocable_r_v<bool, TPredicate, const Output<Node>&>>* = nullptr>
    explicit Predicate(const TPredicate& predicate) {
        m_pred = [=](PatternSymbolMap&, const Output<Node>& out) {
            return predicate(out.get_node_shared_ptr());
        };
    }

    template <
        typename TPredicate,
        typename std::enable_if_t<std::is_invocable_r_v<bool, TPredicate, PatternSymbolMap&, Output<Node>>>* = nullptr>
    explicit Predicate(const TPredicate& predicate) {
        m_pred = predicate;
        m_requires_map = true;
    }

    template <typename TPredicate>
    explicit Predicate(const TPredicate& predicate, std::string name) : Predicate(predicate) {
        if (!name.empty())
            m_name = std::move(name);
    }

    bool operator()(PatternSymbolMap& m, const Output<Node>& output) const;
    bool operator()(const std::shared_ptr<Node>& node) const;
    bool operator()(const Output<Node>& output) const;

    template <typename TPredicate>
    Predicate operator||(const TPredicate& other) const {
        return *this || Predicate(other);
    }

    template <typename TPredicate>
    Predicate operator&&(const TPredicate& other) const {
        return *this && Predicate(other);
    }

    Predicate operator||(const Predicate& other) const {
        auto result = Predicate(
            [pred = m_pred, other_pred = other.m_pred](PatternSymbolMap& m, const Output<Node>& out) -> bool {
                return pred(m, out) || other_pred(m, out);
            },
            m_name + " || " + other.m_name);
        result.m_requires_map = m_requires_map || other.m_requires_map;
        return result;
    }

    Predicate operator&&(const Predicate& other) const {
        auto result = Predicate(
            [pred = m_pred, other_pred = other.m_pred](PatternSymbolMap& m, const Output<Node>& out) -> bool {
                return pred(m, out) && other_pred(m, out);
            },
            m_name + " && " + other.m_name);
        result.m_requires_map = m_requires_map || other.m_requires_map;
        return result;
    }

private:
    bool m_requires_map = false;
    std::string m_name = "no_name";

    std::function<bool(PatternSymbolMap&, const Output<Node>&)> m_pred;
};

template <typename TPredicate,
          typename std::enable_if_t<std::is_constructible_v<Predicate, TPredicate> &&
                                    !std::is_same_v<Predicate, TPredicate>>* = nullptr>
Predicate operator&&(const TPredicate& lhs, const Predicate& rhs) {
    return Predicate(lhs) && rhs;
}

template <typename TPredicate,
          typename std::enable_if_t<std::is_constructible_v<Predicate, TPredicate> &&
                                    !std::is_same_v<Predicate, TPredicate>>* = nullptr>
Predicate operator||(const TPredicate& lhs, const Predicate& rhs) {
    return Predicate(lhs) || rhs;
}
}  // namespace op
}  // namespace ov::pass::pattern
