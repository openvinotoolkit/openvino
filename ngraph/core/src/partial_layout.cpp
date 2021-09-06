// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_layout.hpp"

#include <algorithm>

#include "ngraph/except.hpp"

using namespace ov;

/////////////////////////////////////////////////////////////////////////////////

static constexpr char BATCH[] = "BATCH";
static constexpr char CHANNELS[] = "CHANNELS";
static constexpr char WIDTH[] = "WIDTH";
static constexpr char HEIGHT[] = "HEIGHT";
static constexpr char DEPTH[] = "DEPTH";
static constexpr char SCALAR[] = "SCALAR";

// 1. only order of dimensions "adbc" (0312)
// 2. can define order and meaning for dimensions "NCHW"
// 3. partial layout specialization "NC?"
Layout::Layout(const std::string& layoutStr) {
    if (layoutStr.empty()) {
        throw ngraph::ngraph_error("Cannot parse ov::Layout from an empty string");
    }

    // TODO: ilavreno
    // details::trim(layoutStr);

    // special case
    if (layoutStr == ::SCALAR) {
        set_dim_by_name(::SCALAR, 0);
        return;
    }

    const size_t numDims = layoutStr.length();
    // if it's NCDHW-like variations
    const bool ncdhwLikeLayout = std::all_of(layoutStr.cbegin(), layoutStr.cend(), [](char c) -> bool {
        return c == 'C' || c == 'H' || c == 'W' || c == 'N' || c == 'D' || c == '?';
    });

    auto setDimensionNames = [&layoutStr, numDims, this]() {
        // fill dimension names
        for (size_t i = 0; i < numDims; ++i) {
            if (layoutStr[i] == 'N')
                set_dim_by_name(BATCH, i);
            else if (layoutStr[i] == 'C')
                set_dim_by_name(CHANNELS, i);
            else if (layoutStr[i] == 'D')
                set_dim_by_name(DEPTH, i);
            else if (layoutStr[i] == 'H')
                set_dim_by_name(HEIGHT, i);
            else if (layoutStr[i] == 'W')
                set_dim_by_name(WIDTH, i);
        }
    };

    if (ncdhwLikeLayout) {
        // set only names for dimensions
        setDimensionNames();
    }
}

bool Layout::has_dim(const std::string& dimensionName) const {
    return _dimensionNames.find(dimensionName) != _dimensionNames.end();
}

std::int64_t Layout::get_index_by_name(const std::string& name) const {
    auto it = _dimensionNames.find(name);
    if (it == _dimensionNames.end()) {
        throw ngraph::ngraph_error(name + " dimension index is not defined");
    }
    return it->second;
}

void Layout::set_dim_by_name(const std::string& dimensionName, std::int64_t index) {
    auto it = _dimensionNames.find(dimensionName);

    // we cannot change dimension index
    if (it != _dimensionNames.end() && it->second != index) {
        throw ngraph::ngraph_error("Cannot change " + dimensionName + " dimension index");
    }

    _dimensionNames[dimensionName] = index;
}

#define DEFINE_NAMED_DIMENSION(NAME, name)                         \
    bool layouts::has_##name(const Layout& layout) {               \
        return layout.has_dim(NAME);                               \
    }                                                              \
                                                                   \
    std::int64_t layouts::name(const Layout& layout) {             \
        return layout.get_index_by_name(NAME);                     \
    }                                                              \
                                                                   \
    void layouts::set_##name(Layout& layout, std::int64_t index) { \
        layout.set_dim_by_name(NAME, index);                       \
    }

DEFINE_NAMED_DIMENSION(BATCH, batch)
DEFINE_NAMED_DIMENSION(CHANNELS, channels)
DEFINE_NAMED_DIMENSION(DEPTH, depth)
DEFINE_NAMED_DIMENSION(HEIGHT, height)
DEFINE_NAMED_DIMENSION(WIDTH, width)

bool Layout::is_scalar() const {
    return _dimensionNames.find(::SCALAR) != _dimensionNames.end();
}

constexpr DiscreteTypeInfo AttributeAdapter<ov::Layout>::type_info;

const std::string& AttributeAdapter<ov::Layout>::get() {
    throw ngraph::ngraph_error("not implemented");
}
void AttributeAdapter<ov::Layout>::set(const std::string& value) {
    m_ref = Layout(value);
}
