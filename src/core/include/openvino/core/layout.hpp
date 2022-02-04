// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {

class OPENVINO_API Layout {
public:
    /// \brief Constructs a dynamic Layout with no layout information.
    Layout();

    /// \brief Constructs layout representing scalar
    static Layout scalar();

    /// \brief Constructs a Layout with static or dynamic layout information based
    /// on string representation.
    ///
    /// \param layoutStr The string used to construct Layout from.
    /// The string representation can be in the following form:
    /// - can define order and meaning for dimensions "NCHW"
    /// - partial layout specialization:
    ///   - "NC?" defines 3 dimensional layout, first two NC, 3rd one is not defined
    ///   - "N...C" defines layout with dynamic rank where 1st dimension is N, last one is C
    ///   - "NC..." defines layout with dynamic rank where first two are NC, others are not
    ///   defined
    /// - only order of dimensions "adbc" (0312)
    /// - Advanced syntax can be used for multi-character names like "[N,C,H,W,...,CustomName]"
    Layout(const char* layoutStr) : Layout(std::string(layoutStr)) {}

    explicit Layout(const std::string& layoutStr);

    /// \brief Comparison operator (equal)
    bool operator==(const Layout& rhs) const;

    /// \brief Comparison operator (not equal)
    bool operator!=(const Layout& rhs) const;

    /// \brief Checks if dimension with specified name is in layout
    /// \return `true` if layout has information about dimension index with a given name
    bool has_name(const std::string& dimensionName) const;

    /// \brief Gets index of dimension with a specified name
    ///
    /// \throws ov::AssertFailure if dimension name is not found in a layout
    ///
    /// \return Index of given dimension name
    std::int64_t get_index_by_name(const std::string& dimensionName) const;

    /// \brief String representation of Layout
    std::string to_string() const;

    bool empty() const {
        return *this == Layout();
    }

private:
    /// stores dimension names map to index in a layout
    std::unordered_map<std::string, std::int64_t> m_names;
    std::unordered_map<std::int64_t, std::string> m_index_map;

    /// special case for scalar
    bool m_scalar = false;

    bool m_dynamic = false;
    int64_t m_left_size = 0;
    int64_t m_right_size = 0;

    friend class LayoutUtils;
};

namespace layout {

/// \brief Checks if layout has 'batch' dimension
OPENVINO_API bool has_batch(const Layout& layout);

/// \brief Returns 'batch' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t batch_idx(const Layout& layout);

/// \brief Checks if layout has 'channels' dimension
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API bool has_channels(const Layout& layout);

/// \brief Returns 'channels' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t channels_idx(const Layout& layout);

/// \brief Checks if layout has 'depth' dimension
OPENVINO_API bool has_depth(const Layout& layout);

/// \brief Returns 'depth' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t depth_idx(const Layout& layout);

/// \brief Checks if layout has 'height' dimension
OPENVINO_API bool has_height(const Layout& layout);

/// \brief Returns 'height' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t height_idx(const Layout& layout);

/// \brief Checks if layout has 'width' dimension
OPENVINO_API bool has_width(const Layout& layout);

/// \brief Returns 'width' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t width_idx(const Layout& layout);

/// \brief Sets Layout of port
///
/// \throws ov::Exception if port is not connected with Result or Parameter
OPENVINO_API void set_layout(ov::Output<ov::Node> output, const ov::Layout& layout);

/// \brief Gets Layout of port
///
/// \return layout from port and empty layout in other case
OPENVINO_API ov::Layout get_layout(const ov::Output<ov::Node>& output);

/// \brief Gets Layout of port
///
/// \return layout from port and empty layout in other case
OPENVINO_API ov::Layout get_layout(const ov::Output<const ov::Node>& output);

}  // namespace layout

template <>
class OPENVINO_API AttributeAdapter<Layout> : public ValueAccessor<std::string> {
public:
    OPENVINO_RTTI("AttributeAdapter<Layout>");
    explicit AttributeAdapter(Layout& value) : m_ref(value) {}

    const std::string& get() override;
    void set(const std::string& value) override;
    explicit operator Layout&() {
        return m_ref;
    }

protected:
    Layout& m_ref;
    std::string m_dump;
};

class OPENVINO_API LayoutAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("layout", "0");

    LayoutAttribute() = default;

    explicit LayoutAttribute(const Layout& value) : value(value) {}

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::string to_string() const override;

    Layout value;
};

}  // namespace ov
