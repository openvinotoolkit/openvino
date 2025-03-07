// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/layout.hpp"

#include <algorithm>
#include <cctype>

#include "layout_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {

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
    auto name = ov::util::to_upper(dim_name);
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
    if (layout_str.empty()) {
        m_dynamic = true;
        m_left_size = m_right_size = 0;
        return;
    }
    auto layout = ov::util::trim(layout_str);
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
        m_index_map[index] = std::move(dim_name);
    };

    if (is_advanced_syntax(layout)) {
        OPENVINO_ASSERT(layout.length() > 2, "Cannot parse ov::Layout from an empty string");
        auto parse_commas = [&](const std::string& sub_name, int64_t index = 0) -> int64_t {
            OPENVINO_ASSERT(!sub_name.empty(), "Empty sub-string detected while parsing layout");
            std::istringstream ss(sub_name);
            std::string name;
            while (std::getline(ss, name, ',')) {
                name = ov::util::trim(name);
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
            auto left_layout = ov::util::trim(layout.substr(0, ellipsis));
            if (!left_layout.empty()) {
                OPENVINO_ASSERT(left_layout.at(left_layout.length() - 1) == ',',
                                "Layout: Invalid left side (" + layout + ")");
                left_layout = left_layout.substr(0, left_layout.length() - 1);
                left_index = parse_commas(left_layout);
            }
            auto right_layout = ov::util::trim(layout.substr(ellipsis + ELLIPSIS_LEN));
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
    for (size_t i = 0; i < layout.length(); i++) {
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

class LayoutUtils {
public:
    static Layout apply_permutation(const Layout& src_layout, const std::vector<uint64_t>& dims);
    static std::vector<int64_t> find_permutation(const Layout& src_layout,
                                                 const PartialShape& src_shape,
                                                 const Layout& dst_layout);
    static std::tuple<PartialShape, Layout> find_squeeze(const Layout& src_layout,
                                                         const PartialShape& src_shape,
                                                         const Layout& dst_layout);
    static std::tuple<PartialShape, Layout, size_t> find_unsqueeze(const Layout& src_layout,
                                                                   const PartialShape& src_shape,
                                                                   const Layout& dst_layout);
    static bool is_compatible(const Layout& layout, const PartialShape& shape);
};

Layout LayoutUtils::apply_permutation(const Layout& src_layout, const std::vector<uint64_t>& dims) {
    {  // Validate dims
        std::vector<bool> used(dims.size(), false);
        for (size_t i = 0; i < dims.size(); i++) {
            auto dim = dims[i];
            OPENVINO_ASSERT(dim < dims.size(), "Convert layout: dimension ", dim, " is out of bounds");
            OPENVINO_ASSERT(!used[dim],
                            "Convert layout: dimension ",
                            dim,
                            " is used more than once in convert arguments");
            used[dim] = true;
        }
    }
    if (src_layout.empty()) {
        return src_layout;  // Can return immediately
    }
    // No way to calculate layout from [N...C] with permutation {0, 3, 1, 2}
    OPENVINO_ASSERT(!src_layout.m_dynamic,
                    "Layout conversion by indexes is not supported for dynamic layout: ",
                    src_layout.to_string());
    Layout res;
    res.m_dynamic = false;
    res.m_left_size = src_layout.m_left_size;
    for (size_t i = 0; i < dims.size(); i++) {
        auto it = src_layout.m_index_map.find(static_cast<int64_t>(dims[i]));
        if (it == src_layout.m_index_map.end()) {
            continue;
        }
        res.m_index_map[static_cast<int64_t>(i)] = it->second;
        res.m_names[it->second] = static_cast<int64_t>(i);
    }
    return res;
}

std::vector<int64_t> LayoutUtils::find_permutation(const Layout& src_layout,
                                                   const PartialShape& src_shape,
                                                   const Layout& dst) {
    auto rank = src_shape.rank();
    auto check_trivial = [](std::vector<int64_t>& res) -> std::vector<int64_t>& {
        size_t i = 0;
        while (i < res.size() && res[i] == static_cast<int64_t>(i)) {
            i++;
        }
        if (i == res.size()) {
            // Array is [0,1,2,...,n], so permutation is not needed at all
            res = {};
        }
        return res;
    };
    auto to_static = [](const Layout& layout, const Rank& rank) -> Layout {
        OPENVINO_ASSERT(!layout.m_dynamic || !rank.is_dynamic(),
                        "Conversion is not supported for dynamic layouts with fully dynamic shapes");

        if (!layout.m_dynamic) {
            return layout;
        }
        Layout res = layout;
        auto len = rank.get_length();
        res.m_dynamic = false;
        res.m_left_size = rank.get_length();
        res.m_right_size = 0;
        for (auto& item : res.m_names) {
            if (item.second < 0) {
                item.second += len;
            }
        }
        std::unordered_map<std::int64_t, std::string> new_index_map;
        for (const auto& item : res.m_index_map) {
            auto new_ind = item.first;
            if (new_ind < 0) {
                new_ind += len;
            }
            new_index_map[new_ind] = item.second;
        }
        res.m_index_map = std::move(new_index_map);
        return res;
    };
    // Basic implementation so far, can support partially-specified layouts later (shape rank will be needed for dynamic
    // layouts)
    if (src_layout == dst) {
        return {};  // No permutation is needed
    }
    if (src_layout.empty() || dst.empty()) {
        return {};
    }
    auto src_static = to_static(src_layout, rank);
    auto dst_static = to_static(dst, rank);
    OPENVINO_ASSERT(src_static.m_left_size == dst_static.m_left_size,
                    "Conversion is not supported for layouts with different sizes, ",
                    src_layout.to_string(),
                    " <-> ",
                    dst.to_string());
    OPENVINO_ASSERT(rank.is_dynamic() || src_static.m_left_size == rank.get_length(),
                    "Conversion layout ",
                    src_layout.to_string(),
                    " <-> ",
                    dst.to_string(),
                    " failure. Layout is not consistent with input shape ",
                    src_shape,
                    ". Layout length ",
                    src_static.m_left_size,
                    " shall match with input shape rank ",
                    rank.get_length());
    std::vector<int64_t> res(src_static.m_left_size, -1);
    if (src_static.m_names.size() > dst_static.m_names.size()) {
        // find inverted permutation from least specified layout to most one
        auto inverted = find_permutation(dst_static, src_shape, src_static);
        if (inverted.empty()) {
            return {};
        }
        for (size_t i = 0; i < inverted.size(); i++) {
            res[inverted[i]] = static_cast<int64_t>(i);
        }
        return check_trivial(res);
    }
    std::vector<bool> mapped(src_static.m_left_size, false);
    // Fill known names (??c? -> nc??) will produce res=[-1,2,-1,-1], mapped=[false,false,true,false]
    for (const auto& src_item : src_static.m_index_map) {
        OPENVINO_ASSERT(dst.has_name(src_item.second),
                        "Dimension name '",
                        src_item.second,
                        "' is not found in layout: ",
                        dst_static.to_string());
        auto dst_ind = dst_static.get_index_by_name(src_item.second);
        res[dst_ind] = src_item.first;
        mapped[src_item.first] = true;
    }
    // Fill the rest
    int dst_pos = 0;
    auto find_free_pos = [&]() {
        while (mapped[dst_pos] && dst_pos < src_static.m_left_size) {
            dst_pos++;
        }
        OPENVINO_ASSERT(dst_pos < src_static.m_left_size,
                        "Internal unexpected error: can't map layout ",
                        src_static.to_string(),
                        " to ",
                        dst_static.to_string());
        mapped[dst_pos] = true;
        return dst_pos;
    };
    for (int64_t i = 0; i < src_static.m_left_size; i++) {
        if (res[i] < 0) {
            res[i] = find_free_pos();
        }
    }
    return check_trivial(res);
}

std::tuple<PartialShape, Layout> LayoutUtils::find_squeeze(const Layout& src_layout,
                                                           const PartialShape& src_shape,
                                                           const Layout& dst_layout) {
    if (src_layout.m_dynamic || dst_layout.m_dynamic || src_layout.m_left_size <= dst_layout.m_left_size) {
        return {src_shape, src_layout};
    }

    // Don't allow conversions like model_layout=NC??, tensor_layout=HWC
    // Though in future such conversions may be possible to implement
    OPENVINO_ASSERT(src_layout.m_left_size == static_cast<int64_t>(src_layout.m_index_map.size()),
                    "Layout conversion ",
                    dst_layout.to_string(),
                    " <-> ",
                    src_layout.to_string(),
                    " is not supported. Please use fully specified model layout, current is ",
                    src_layout.to_string());

    // Don't allow conversions like model_layout=NCHW, tensor_layout=?HW
    OPENVINO_ASSERT(dst_layout.m_left_size == static_cast<int64_t>(dst_layout.m_index_map.size()),
                    "Layout conversion ",
                    dst_layout.to_string(),
                    " <-> ",
                    src_layout.to_string(),
                    " is not supported. Please use fully specified tensor layout, current is ",
                    dst_layout.to_string());

    bool rank_dynamic = src_shape.rank().is_dynamic();
    OPENVINO_ASSERT(rank_dynamic || src_shape.rank().get_length() == src_layout.m_left_size,
                    "Model input layout ",
                    src_layout.to_string(),
                    " is inconsistent with input shape ",
                    src_shape,
                    ". Layout and shape shall have same rank, got ",
                    src_layout.m_left_size,
                    " != ",
                    src_shape.rank().get_length());
    // At this point src_layout and dst_layout don't have '...' or '?'
    auto res_dims =
        rank_dynamic ? PartialShape::dynamic() : PartialShape(std::vector<Dimension>(dst_layout.m_left_size));
    Layout res;
    res.m_dynamic = false;
    res.m_left_size = dst_layout.m_left_size;
    int64_t dst_idx = 0;
    for (int64_t src_idx = 0; src_idx < src_layout.m_left_size; src_idx++) {
        const auto& src_dim_name = src_layout.m_index_map.at(src_idx);
        if (dst_layout.has_name(src_dim_name)) {
            if (!rank_dynamic) {
                res_dims[dst_idx] = src_shape[src_idx];
            }
            res.m_index_map[dst_idx] = src_dim_name;
            res.m_names[src_dim_name] = dst_idx;
            dst_idx++;
        }
    }
    if (dst_idx != dst_layout.m_left_size) {
        std::stringstream missing_names;
        missing_names << "( ";
        for (const auto& dst_item : dst_layout.m_names) {
            const auto& key = dst_item.first;
            if (!res.m_names.count(key)) {
                missing_names << "'" << key << "' ";
            }
        }
        missing_names << ")";
        OPENVINO_ASSERT(dst_idx == dst_layout.m_left_size,
                        "Layout conversion failed. Tensor layout",
                        dst_layout.to_string(),
                        " has dimensions missing in model layout ",
                        src_layout.to_string(),
                        ". Missing dimensions are ",
                        missing_names.str());
    }
    return {std::move(res_dims), std::move(res)};
}

std::tuple<PartialShape, Layout, size_t> LayoutUtils::find_unsqueeze(const Layout& src_layout,
                                                                     const PartialShape& src_shape,
                                                                     const Layout& dst_layout) {
    if (src_layout.m_dynamic || dst_layout.m_dynamic || src_layout.m_left_size >= dst_layout.m_left_size) {
        return {src_shape, src_layout, {}};
    }

    // find_squeeze already performed necessary validation, no need to repeat here
    bool rank_dynamic = src_shape.rank().is_dynamic();
    auto dims_cnt = dst_layout.m_left_size - src_layout.m_left_size;
    auto res_dims =
        rank_dynamic ? PartialShape::dynamic() : PartialShape(std::vector<Dimension>(dst_layout.m_left_size, 1));
    Layout res;
    res.m_dynamic = false;
    res.m_left_size = dst_layout.m_left_size;
    int64_t unset_idx = 0;
    for (auto i = 0; i < dst_layout.m_left_size; i++) {
        const auto& dim_name = dst_layout.m_index_map.at(i);
        if (src_layout.has_name(dim_name)) {
            auto src_idx = src_layout.get_index_by_name(dim_name);
            res.m_names[dim_name] = src_idx + dims_cnt;
            res.m_index_map[src_idx + dims_cnt] = dim_name;
            if (!rank_dynamic) {
                res_dims[src_idx + dims_cnt] = src_shape[src_idx];
            }
        } else {
            res.m_names[dim_name] = unset_idx;
            res.m_index_map[unset_idx] = dim_name;
            unset_idx++;
        }
    }
    return {std::move(res_dims), std::move(res), dims_cnt};
}

bool LayoutUtils::is_compatible(const Layout& layout, const PartialShape& shape) {
    auto layout_min_rank = layout.m_left_size + layout.m_right_size;
    int64_t layout_max_rank = layout.m_dynamic ? -1 : layout_min_rank;
    return shape.rank().compatible(Dimension(layout_min_rank, layout_max_rank));
}

namespace layout {
namespace utils {
Layout apply_permutation(const Layout& src_layout, const std::vector<uint64_t>& dims) {
    return LayoutUtils::apply_permutation(src_layout, dims);
}

std::vector<int64_t> find_permutation(const Layout& src_layout,
                                      const PartialShape& src_shape,
                                      const Layout& dst_layout) {
    return LayoutUtils::find_permutation(src_layout, src_shape, dst_layout);
}

std::tuple<PartialShape, Layout> find_squeeze(const Layout& src_layout,
                                              const PartialShape& src_shape,
                                              const Layout& dst_layout) {
    return LayoutUtils::find_squeeze(src_layout, src_shape, dst_layout);
}

std::tuple<PartialShape, Layout, size_t> find_unsqueeze(const Layout& src_layout,
                                                        const PartialShape& src_shape,
                                                        const Layout& dst_layout) {
    return LayoutUtils::find_unsqueeze(src_layout, src_shape, dst_layout);
}

bool is_compatible(const Layout& layout, const PartialShape& shape) {
    return LayoutUtils::is_compatible(layout, shape);
}

}  // namespace utils

// Helper functions
bool has_batch(const Layout& layout) {
    return layout.has_name(BATCH);
}

std::int64_t batch_idx(const Layout& layout) {
    return layout.get_index_by_name(BATCH);
}

bool has_depth(const Layout& layout) {
    return layout.has_name(DEPTH);
}

std::int64_t depth_idx(const Layout& layout) {
    return layout.get_index_by_name(DEPTH);
}

bool has_channels(const Layout& layout) {
    return layout.has_name(CHANNELS);
}

std::int64_t channels_idx(const Layout& layout) {
    return layout.get_index_by_name(CHANNELS);
}

bool has_height(const Layout& layout) {
    return layout.has_name(HEIGHT);
}

std::int64_t height_idx(const Layout& layout) {
    return layout.get_index_by_name(HEIGHT);
}

bool has_width(const Layout& layout) {
    return layout.has_name(WIDTH);
}

std::int64_t width_idx(const Layout& layout) {
    return layout.get_index_by_name(WIDTH);
}

ov::Layout get_layout(const ov::Output<const ov::Node>& output) {
    auto it = output.get_rt_info().find(ov::LayoutAttribute::get_type_info_static());
    if (it == output.get_rt_info().end()) {
        return {};
    }
    return it->second.as<ov::LayoutAttribute>().value;
}

ov::Layout get_layout(const ov::Output<ov::Node>& output) {
    return get_layout(ov::Output<const ov::Node>(output.get_node(), output.get_index()));
}

void set_layout(ov::Output<ov::Node> output, const ov::Layout& layout) {
    OPENVINO_ASSERT(
        ov::as_type<ov::op::v0::Parameter>(output.get_node()) || ov::as_type<ov::op::v0::Result>(output.get_node()),
        "Layout can be set only for Parameter and Result operations.");
    if (layout.empty()) {
        output.get_rt_info().erase(ov::LayoutAttribute::get_type_info_static());
    } else {
        OPENVINO_ASSERT(ov::layout::utils::is_compatible(layout, output.get_partial_shape()),
                        "Can't set layout for Parameter/Result ",
                        output,
                        ": layout ",
                        layout.to_string(),
                        " is not compatible with shape ",
                        output.get_partial_shape());
        output.get_rt_info()[ov::LayoutAttribute::get_type_info_static()] = ov::LayoutAttribute(layout);
    }
}

}  // namespace layout

const std::string& AttributeAdapter<ov::Layout>::get() {
    m_dump = m_ref.to_string();
    return m_dump;
}

void AttributeAdapter<ov::Layout>::set(const std::string& value) {
    m_ref = Layout(value);
}

AttributeAdapter<Layout>::~AttributeAdapter() = default;

bool LayoutAttribute::visit_attributes(AttributeVisitor& visitor) {
    std::string layout_str = value.to_string();
    visitor.on_attribute("layout", layout_str);
    // some attribute visitor will not change the value
    // for example, rt info serializer
    // in this case, parallelization can be supported in hash pass
    if (layout_str != value.to_string())
        value = Layout(layout_str);
    return true;
}

std::string LayoutAttribute::to_string() const {
    return value.to_string();
}

}  // namespace ov
