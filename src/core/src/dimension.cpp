// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/dimension.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

#include "openvino/util/common_util.hpp"

using namespace ov;

namespace {
/**
 * \brief Merges two symbols.
 *
 *  | lhs       | rhs       | result    |
 *  |-----------|-----------|-----------|
 *  | X         | X         | X         |
 *  | X         | no symbol | X         |
 *  | no symbol | X         | X         |
 *  | X         | Y         | Y         | (if merge_unequal == true)
 *  | X         | Y         | no symbol | (if merge_unequal == false)
 *
 * \param lhs  First input symbol.
 * \param lhs  Second input symbol.
 *
 * \return Merged symbol shared pointer
 */
std::shared_ptr<ov::Symbol> merge_symbols(std::shared_ptr<ov::Symbol> lhs,
                                          std::shared_ptr<ov::Symbol> rhs,
                                          bool merge_unequal = true) {
    if (ov::symbol::are_equal(lhs, rhs) || rhs == nullptr)
        return lhs;
    else if (merge_unequal || lhs == nullptr)
        return rhs;
    else
        return nullptr;
}

int64_t stringToInt64(const std::string& valStr) {
    int64_t ret{0};
    std::istringstream ss(valStr);
    if (!ss.eof()) {
        ss >> ret;
    }
    return ret;
}

bool check_all_digits(const std::string& value) {
    auto val = ov::util::trim(value);
    for (const auto& c : val) {
        if (!std::isdigit(c) || c == '-')
            return false;
    }
    return true;
}
Dimension::value_type dimension_length(Interval::value_type vt) {
    return vt == Interval::s_max ? -1 : vt;
}
}  // namespace

std::ostream& ov::operator<<(std::ostream& str, const Dimension& dimension) {
    if (dimension.is_static()) {
        return str << dimension.get_length();
    } else if (dimension.get_min_length() > 0) {
        str << dimension.get_min_length() << "..";
        if (dimension.get_interval().has_upper_bound())
            return str << dimension.get_max_length();
        else
            return str;
    } else if (dimension.get_interval().has_upper_bound()) {
        return str << ".." << dimension.get_max_length();
    } else {
        return str << "?";
    }
}

Dimension::Dimension(value_type dimension)
    : m_dimension(dimension == -1 ? 0 : dimension, dimension == -1 ? Interval::s_max : dimension) {}

Dimension::Dimension(value_type min_dimension, value_type max_dimension)
    : m_dimension(min_dimension == -1 ? 0 : min_dimension, max_dimension == -1 ? Interval::s_max : max_dimension) {}

Dimension::Dimension(const std::string& value) {
    auto val = ov::util::trim(value);
    if (val == "?" || val == "-1") {
        m_dimension = {0, Interval::s_max};
        return;
    }
    if (val.find("..") == std::string::npos) {
        OPENVINO_ASSERT(check_all_digits(val), "Cannot parse dimension: \"" + val + "\"");
        m_dimension = {stringToInt64(val)};
        return;
    }

    std::string min_value_str = val.substr(0, val.find(".."));
    min_value_str = ov::util::trim(min_value_str);

    int64_t min_value;
    if (min_value_str.empty())
        min_value = 0;
    else {
        OPENVINO_ASSERT(check_all_digits(min_value_str), "Cannot parse min bound: \"" + min_value_str + "\"");
        min_value = stringToInt64(min_value_str);
    }

    std::string max_value_str = val.substr(val.find("..") + 2);
    max_value_str = ov::util::trim(max_value_str);

    int64_t max_value;
    if (max_value_str.empty())
        max_value = Interval::s_max;
    else {
        OPENVINO_ASSERT(check_all_digits(max_value_str), "Cannot parse max bound: \"" + max_value_str + "\"");
        max_value = stringToInt64(max_value_str);
    }
    m_dimension = Interval(min_value, max_value);
}

std::string Dimension::to_string() const {
    std::stringstream dim_str_stream;
    dim_str_stream << Dimension(m_dimension);
    return dim_str_stream.str();
}

Dimension Dimension::operator+(const Dimension& dim) const {
    if (dim.m_dimension == 0)
        return *this;
    else if (m_dimension == 0)
        return dim;
    return Dimension(m_dimension + dim.m_dimension);
}

Dimension Dimension::operator-(const Dimension& dim) const {
    if (dim.m_dimension == 0)
        return *this;
    return Dimension(m_dimension - dim.m_dimension);
}

Dimension Dimension::operator/(const value_type divisor) const {
    OPENVINO_ASSERT(divisor >= 0, "divisor must be greater than 0");
    if (divisor == 1)
        return *this;
    if (m_dimension.get_max_val() == Interval::s_max && m_dimension.get_min_val() == 0)
        return Dimension::dynamic();
    const auto& lower_bound = static_cast<int64_t>(ceil(static_cast<double>(m_dimension.get_min_val()) / divisor));
    const auto& upper_bound = static_cast<int64_t>(floor(static_cast<double>(m_dimension.get_max_val()) / divisor));
    return Dimension(lower_bound, upper_bound);
}

Dimension Dimension::operator*(const Dimension& dim) const {
    if (dim.m_dimension == 1)
        return *this;
    else if (m_dimension == 1)
        return dim;
    return Dimension(m_dimension * dim.m_dimension);
}

Dimension Dimension::operator&(const Dimension& dim) const {
    return Dimension(m_dimension & dim.m_dimension);
}

Dimension& Dimension::operator&=(const Dimension& dim) {
    m_dimension &= dim.m_dimension;
    return *this;
}

bool Dimension::compatible(const Dimension& d) const {
    return !(m_dimension & d.m_dimension).empty();
}

bool Dimension::relaxes(const Dimension& d) const {
    return m_dimension.contains(d.m_dimension);
}

bool Dimension::refines(const Dimension& d) const {
    return d.m_dimension.contains(m_dimension);
}

bool Dimension::same_scheme(const Dimension& dim) const {
    return (m_dimension == dim.m_dimension) || (m_dimension.size() > 1 && dim.m_dimension.size() > 1);
}

bool Dimension::merge(Dimension& dst, const Dimension& d1, const Dimension& d2) {
    const auto result_interval = d1.m_dimension & d2.m_dimension;

    if (result_interval.empty()) {
        return false;
    } else if ((&dst == &d1) || (&dst == &d2)) {
        // If dst is one of inputs object change interval only.
        dst.m_dimension = result_interval;
    } else {
        dst = Dimension(result_interval);
    }
    ov::symbol::set_equal(d1.m_symbol, d2.m_symbol);
    dst.m_symbol = merge_symbols(d1.m_symbol, d2.m_symbol);
    return true;
}

bool Dimension::broadcast_merge(Dimension& dst, const Dimension& d1, const Dimension& d2) {
    bool d1_has_1 = d1.m_dimension.contains(1);
    bool d2_has_1 = d2.m_dimension.contains(1);
    if (d1_has_1 && d2_has_1) {
        auto result = ov::Interval(std::min(d1.m_dimension.get_min_val(), d2.m_dimension.get_min_val()),
                                   std::max(d1.m_dimension.get_max_val(), d2.m_dimension.get_max_val()));
        if (result.empty())
            return false;
        dst = Dimension(result);
        dst.m_symbol = merge_symbols(d1.m_symbol, d2.m_symbol, dst.is_static());
        return true;
    } else if (d1_has_1) {
        dst = d2;
    } else if (d2_has_1) {
        dst = d1;
    } else {
        return merge(dst, d1, d2);
    }
    return true;
}

Dimension::value_type Dimension::get_length() const {
    if (is_dynamic()) {
        OPENVINO_THROW("Cannot get length of dynamic dimension");
    }
    return m_dimension.get_min_val();
}

Dimension::value_type Dimension::get_max_length() const {
    return dimension_length(m_dimension.get_max_val());
}

Dimension::value_type Dimension::get_min_length() const {
    return dimension_length(m_dimension.get_min_val());
}

bool Dimension::has_symbol() const {
    return m_symbol != nullptr;
}

std::shared_ptr<ov::Symbol> Dimension::get_symbol() const {
    return m_symbol;
}

void Dimension::set_symbol(const std::shared_ptr<ov::Symbol>& s) {
    m_symbol = s;
}

AttributeAdapter<Dimension>::~AttributeAdapter() = default;
