// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/layout.hpp"

#include <algorithm>

#include "ngraph/except.hpp"
#include "ngraph/util.hpp"

using namespace ov;

/////////////////////////////////////////////////////////////////////////////////

static constexpr char BATCH[] = "N";
static constexpr char CHANNELS[] = "C";
static constexpr char WIDTH[] = "W";
static constexpr char HEIGHT[] = "H";
static constexpr char DEPTH[] = "D";
static constexpr char ELLIPSIS[] = "...";
static constexpr int ELLIPSIS_LEN = 3;

static const std::map<std::string, layouts::PredefinedDim>& dim_aliases() {
    static const std::map<std::string, layouts::PredefinedDim> DIM_ALIASES = {
            {BATCH, layouts::PredefinedDim::BATCH},
            {"BATCH", layouts::PredefinedDim::BATCH},
            {"B", layouts::PredefinedDim::BATCH},
            {CHANNELS, layouts::PredefinedDim::CHANNELS},
            {"CHANNELS", layouts::PredefinedDim::CHANNELS},
            {"CHANNEL", layouts::PredefinedDim::CHANNELS},
            {HEIGHT, layouts::PredefinedDim::HEIGHT},
            {"HEIGHT", layouts::PredefinedDim::HEIGHT},
            {WIDTH, layouts::PredefinedDim::WIDTH},
            {"WIDTH", layouts::PredefinedDim::WIDTH},
            {DEPTH, layouts::PredefinedDim::DEPTH},
            {"DEPTH", layouts::PredefinedDim::DEPTH}
    };
    return DIM_ALIASES;
}

static std::string to_internal_name(const std::string& dim_name) {
    auto name = ngraph::to_upper(dim_name);
    auto it = dim_aliases().find(name);
    if (it != dim_aliases().end()) {
        name = layouts::predefined_name(it->second);
    }
    return name;
}

static void validate_name(const std::string& dim_name) {
    OV_CHECK(!dim_name.empty(), "Layout name can't be empty");
    bool has_alphanumeric = false;
    for (const auto& c: dim_name) {
        bool is_alnum = std::isalnum(c);
        has_alphanumeric |= is_alnum;
        OV_CHECK(is_alnum || c == '_' ,
                 "Layout name is invalid (" + dim_name + "). Only english letters, digits and _ is allowed");
    }
    OV_CHECK(has_alphanumeric, "Layout name is invalid (" + dim_name + "). Name shall have alphanumeric characters");
}

Layout Layout::scalar() {
    Layout l;
    l.m_scalar = true;
    return l;
}

// 1. only order of dimensions "adbc" (0312)
// 2. can define order and meaning for dimensions "NCHW"
// 3. partial layout specialization "NC?"
Layout::Layout(const std::string& layout_str) {
    if (layout_str.empty()) {
        throw ngraph::ngraph_error("Cannot parse ov::Layout from an empty string");
    }

    auto layout = ngraph::trim(layout_str);

    auto is_serialized = [](const std::string& layout) {
        return layout.length() >= 2 && layout.front() == '[' && layout.back() == '[';
    };

    if (is_serialized(layout)) {
        // TODO: parse from serialized string with format like "[batch, channels, height, width]"
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
            OV_CHECK(!backward, std::string("Multiple ") + ELLIPSIS + " are not allowed");
            OV_CHECK(i == dynamic_start, "Undefined number of dimensions shall have ...");
            // check next characters
            i += ELLIPSIS_LEN - 1;
            index += ELLIPSIS_LEN - 1;
            // undefined middle dimension
            backward = true;
            index = index - static_cast<int64_t>(layout.length());
            continue;
        }
        // Only letters and digits are allowed
        std::string dim_name = std::string(1, static_cast<char>(c));
        validate_name(dim_name);
        dim_name = to_internal_name(dim_name);
        OV_CHECK(m_names.count(dim_name) == 0, "Dimension (" + dim_name + ") is defined multiple times in layout");
        m_names[dim_name] = index;
    }
    if (dynamic_start != std::string::npos) {
        m_rank = LayoutRank::create_dynamic(dynamic_start, layout.length() - dynamic_start - ELLIPSIS_LEN);
    } else {
        m_rank = LayoutRank::create_static(layout.length());
    }
}

bool Layout::operator==(const Layout& rhs) const {
    return true;
}

bool Layout::operator!=(const Layout& rhs) const {
    return false;
}

bool Layout::has_name(const std::string& dimension_name) const {
    auto name = to_internal_name(dimension_name);
    return m_names.count(name) > 0;
}

std::int64_t Layout::get_index_by_name(const std::string& dimension_name) const {
    auto name = to_internal_name(dimension_name);
    auto it = m_names.find(name);
    if (it == m_names.end()) {
        throw ngraph::ngraph_error(dimension_name + " dimension index is not defined");
    }
    return it->second;
}

void Layout::set_name_for_index(const std::string& dimension_name, std::int64_t index) {
    auto name = to_internal_name(dimension_name);
    validate_name(name);
    auto it = m_names.find(name);
    if (it != m_names.end() && it->second != index) {
        throw ngraph::ngraph_error("Cannot change " + dimension_name + " dimension index");
    } else if (it == m_names.end()) {
        m_names[name] = index;
    }
}

bool Layout::is_scalar() const {
    return m_scalar;
}

std::string layouts::predefined_name(layouts::PredefinedDim dim) {
    switch (dim) {
        case PredefinedDim::BATCH:
            return BATCH;
        case PredefinedDim::CHANNELS:
            return CHANNELS;
        case PredefinedDim::WIDTH:
            return WIDTH;
        case PredefinedDim::HEIGHT:
            return HEIGHT;
        case PredefinedDim::DEPTH:
            return DEPTH;
        case PredefinedDim::UNDEFINED:
            break;
    }
    return {};
}

OPENVINO_API layouts::PredefinedDim layouts::to_predefined_dim(const std::string& predef_name) {
    auto name = ngraph::to_upper(predef_name);
    auto it = dim_aliases().find(name);
    if (it != dim_aliases().end()) {
        return it->second;
    }
    return PredefinedDim::UNDEFINED;
}

#define DEFINE_NAMED_DIMENSION(NAME, name)                         \
    bool layouts::has_##name(const Layout& layout) {               \
        return layout.has_name(NAME);                              \
    }                                                              \
                                                                   \
    std::int64_t layouts::name(const Layout& layout) {             \
        return layout.get_index_by_name(NAME);                     \
    }                                                              \
                                                                   \
    void layouts::set_##name(Layout& layout, std::int64_t index) { \
        layout.set_name_for_index(NAME, index);                    \
    }

DEFINE_NAMED_DIMENSION(BATCH, batch)

DEFINE_NAMED_DIMENSION(CHANNELS, channels)

DEFINE_NAMED_DIMENSION(DEPTH, depth)

DEFINE_NAMED_DIMENSION(HEIGHT, height)

DEFINE_NAMED_DIMENSION(WIDTH, width)

constexpr DiscreteTypeInfo AttributeAdapter<ov::Layout>::type_info;

const std::string& AttributeAdapter<ov::Layout>::get() {
    throw ngraph::ngraph_error("not implemented");
}
void AttributeAdapter<ov::Layout>::set(const std::string& value) {
    m_ref = Layout(value);
}
