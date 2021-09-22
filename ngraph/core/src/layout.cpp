// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/layout.hpp"

#include <algorithm>
#include <cctype>

#include "ngraph/except.hpp"
#include "ngraph/util.hpp"

using namespace ov;

/////////////////////////////////////////////////////////////////////////////////

static constexpr char BATCH[] = "N";
static constexpr char CHANNELS[] = "C";
static constexpr char WIDTH[] = "W";
static constexpr char HEIGHT[] = "H";
static constexpr char DEPTH[] = "D";
static constexpr char SCALAR[] = "**SCALAR**";
static constexpr char ELLIPSIS[] = "...";
static constexpr int ELLIPSIS_LEN = 3;

static const std::map<std::string, std::string>& dim_aliases() {
    static const std::map<std::string, std::string> DIM_ALIASES = {{BATCH, BATCH},
                                                                   {"BATCH", BATCH},
                                                                   {"B", BATCH},
                                                                   {"CHANNELS", CHANNELS},
                                                                   {"CHANNEL", CHANNELS},
                                                                   {"HEIGHT", HEIGHT},
                                                                   {"WIDTH", WIDTH},
                                                                   {"DEPTH", DEPTH}};
    return DIM_ALIASES;
}

static std::string to_internal_name(const std::string& dim_name) {
    auto name = ngraph::to_upper(dim_name);
    auto it = dim_aliases().find(name);
    if (it != dim_aliases().end()) {
        name = it->second;
    }
    return name;
}

static void validate_name(const std::string& dim_name) {
    OPENVINO_ASSERT(!dim_name.empty(), "Layout dimension name can't be empty");
    bool has_alphanumeric = false;
    for (const auto& c : dim_name) {
        bool is_alnum = std::isalnum(c);
        has_alphanumeric |= is_alnum;
        OPENVINO_ASSERT(is_alnum || c == '_',
                        "Layout name is invalid (" + dim_name + "). Only english letters, digits and _ is allowed");
    }
    OPENVINO_ASSERT(has_alphanumeric,
                    "Layout name is invalid (" + dim_name + "). Name shall have alphanumeric characters");
}

Layout::Layout() : m_dynamic(true), m_left_size(0), m_right_size(0) {}

Layout Layout::scalar() {
    return SCALAR;
}

// 1. only order of dimensions "adbc" (0312)
// 2. can define order and meaning for dimensions "NCHW"
// 3. partial layout specialization "NC?"
Layout::Layout(const std::string& layout_str) {
    auto layout = ngraph::trim(layout_str);
    OPENVINO_ASSERT(layout.length() > 0, "Cannot parse ov::Layout from an empty string");
    if (layout == SCALAR) {
        m_scalar = true;
        m_dynamic = false;
        return;
    }
    auto is_advanced_syntax = [](const std::string& layout) {
        return layout.length() >= 2 && layout.front() == '[' && layout.back() == ']';
    };

    auto assign_name = [&](const std::string& name, int64_t index) {
        auto dim_name = to_internal_name(name);
        validate_name(name);
        OPENVINO_ASSERT(m_names.count(dim_name) == 0,
                        "Dimension (" + dim_name + ") is defined multiple times in layout");
        m_names[dim_name] = index;
        m_index_map[index] = dim_name;
    };

    if (is_advanced_syntax(layout)) {
        OPENVINO_ASSERT(layout.length() > 2, "Cannot parse ov::Layout from an empty string");
        auto parse_commas = [&](const std::string& sub_name, int64_t index = 0) -> int64_t {
            OPENVINO_ASSERT(!sub_name.empty(), "Empty sub-string detected while parsing layout");
            std::istringstream ss(sub_name);
            std::string name;
            while (std::getline(ss, name, ',')) {
                name = ngraph::trim(name);
                if (name != "?") {
                    assign_name(name, index);
                }
                index++;
            }
            return index;
        };
        layout = layout.substr(1, layout.length() - 2);  // remove []
        auto ellipsis = layout.find(ELLIPSIS);
        if (ellipsis == std::string::npos) {
            auto last_index = parse_commas(layout);
            m_dynamic = false;
            m_left_size = last_index;
        } else {
            int64_t left_index = 0, right_index = 0;
            // Parse left and right parts
            auto left_layout = ngraph::trim(layout.substr(0, ellipsis));
            if (!left_layout.empty()) {
                OPENVINO_ASSERT(left_layout.at(left_layout.length() - 1) == ',',
                                "Layout: Invalid left side (" + layout + ")");
                left_layout = left_layout.substr(0, left_layout.length() - 1);
                left_index = parse_commas(left_layout);
            }
            auto right_layout = ngraph::trim(layout.substr(ellipsis + ELLIPSIS_LEN));
            if (!right_layout.empty()) {
                OPENVINO_ASSERT(right_layout.at(0) == ',', "Layout: Invalid right side (" + layout + ")");
                right_layout = right_layout.substr(1, right_layout.length() - 1);
                right_index = std::count(right_layout.begin(), right_layout.end(), ',') + 1;
                parse_commas(right_layout, -right_index);
            }
            m_dynamic = true;
            m_left_size = left_index;
            m_right_size = right_index;
        }
        return;
    }
    auto dynamic_start = layout.find(ELLIPSIS);
    bool backward = false;
    int64_t index = -1;
    for (auto i = 0; i < layout.length(); i++) {
        index++;
        auto c = std::toupper(layout[i]);
        if (c == '?') {
            continue;
        } else if (c == '.') {
            OPENVINO_ASSERT(!backward, std::string("Multiple ") + ELLIPSIS + " are not allowed");
            OPENVINO_ASSERT(i == dynamic_start, "Undefined number of dimensions shall have ...");
            // check next characters
            i += ELLIPSIS_LEN - 1;
            index += ELLIPSIS_LEN - 1;
            // undefined middle dimension
            backward = true;
            index = index - static_cast<int64_t>(layout.length());
            continue;
        }
        assign_name(std::string(1, static_cast<char>(c)), index);
    }
    if (dynamic_start != std::string::npos) {
        m_dynamic = true;
        m_left_size = static_cast<int64_t>(dynamic_start);
        m_right_size = static_cast<int64_t>(layout.length() - dynamic_start - ELLIPSIS_LEN);
    } else {
        m_dynamic = false;
        m_left_size = static_cast<int64_t>(layout.length());
    }
}

bool Layout::operator==(const Layout& rhs) const {
    if (m_scalar != rhs.m_scalar || m_dynamic != rhs.m_dynamic || m_left_size != rhs.m_left_size ||
        m_right_size != rhs.m_right_size) {
        return false;
    }
    for (const auto& item : m_names) {
        auto it = rhs.m_names.find(item.first);
        if (it == rhs.m_names.end()) {
            return false;
        }
        if (it->second != item.second) {
            return false;
        }
    }
    return std::all_of(rhs.m_names.begin(), rhs.m_names.end(), [&](const std::pair<std::string, int64_t>& item) {
        return m_names.count(item.first);
    });
}

bool Layout::operator!=(const Layout& rhs) const {
    return !(*this == rhs);
}

bool Layout::has_name(const std::string& dimension_name) const {
    auto name = to_internal_name(dimension_name);
    return m_names.count(name) > 0;
}

std::int64_t Layout::get_index_by_name(const std::string& dimension_name) const {
    auto name = to_internal_name(dimension_name);
    auto it = m_names.find(name);
    OPENVINO_ASSERT(it != m_names.end(), dimension_name + " dimension index is not defined");
    return it->second;
}

std::string Layout::to_string() const {
    if (m_scalar) {
        return SCALAR;
    }
    std::stringstream res;
    res << "[";
    auto add_dim = [&](int64_t index) {
        auto it = m_index_map.find(index);
        if (it == m_index_map.end()) {
            res << "?";
        } else {
            res << it->second;
        }
    };

    if (m_left_size > 0) {
        add_dim(0);
    }
    for (int64_t i = 1; i < m_left_size; i++) {
        res << ",";
        add_dim(i);
    }
    if (m_dynamic) {
        if (m_left_size > 0) {
            res << ",";
        }
        res << "...";
        for (int64_t i = -m_right_size; i < 0; i++) {
            res << ",";
            add_dim(i);
        }
    }
    res << "]";
    return res.str();
}

#define DEFINE_NAMED_DIMENSION(NAME, name)            \
    bool layout::has_##name(const Layout& layout) {   \
        return layout.has_name(NAME);                 \
    }                                                 \
                                                      \
    std::int64_t layout::name(const Layout& layout) { \
        return layout.get_index_by_name(NAME);        \
    }

DEFINE_NAMED_DIMENSION(BATCH, batch)

DEFINE_NAMED_DIMENSION(CHANNELS, channels)

DEFINE_NAMED_DIMENSION(DEPTH, depth)

DEFINE_NAMED_DIMENSION(HEIGHT, height)

DEFINE_NAMED_DIMENSION(WIDTH, width)

constexpr DiscreteTypeInfo AttributeAdapter<ov::Layout>::type_info;

const std::string& AttributeAdapter<ov::Layout>::get() {
    m_dump = m_ref.to_string();
    return m_dump;
}

void AttributeAdapter<ov::Layout>::set(const std::string& value) {
    m_ref = Layout(value);
}

constexpr VariantTypeInfo VariantWrapper<ov::Layout>::type_info;
