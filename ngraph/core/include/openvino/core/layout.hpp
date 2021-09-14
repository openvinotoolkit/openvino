// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

#include "ngraph/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/variant.hpp"

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
    ///   - "N...C" defines layout with dynamic rank where first two are NC, others are not
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

private:
    /// stores dimension names map to index in a layout
    std::unordered_map<std::string, std::int64_t> m_names;
    std::unordered_map<std::int64_t, std::string> m_index_map;

    /// special case for scalar
    bool m_scalar = false;

    bool m_dynamic = false;
    int64_t m_left_size = 0;
    int64_t m_right_size = 0;
};

namespace layout {

/// \brief Checks if layout has 'batch' dimension
OPENVINO_API bool has_batch(const Layout& layout);

/// \brief Returns 'batch' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t batch(const Layout& layout);

/// \brief Checks if layout has 'channels' dimension
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API bool has_channels(const Layout& layout);

/// \brief Returns 'channels' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t channels(const Layout& layout);

/// \brief Checks if layout has 'depth' dimension
OPENVINO_API bool has_depth(const Layout& layout);

/// \brief Returns 'depth' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t depth(const Layout& layout);

/// \brief Checks if layout has 'height' dimension
OPENVINO_API bool has_height(const Layout& layout);

/// \brief Returns 'height' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t height(const Layout& layout);

/// \brief Checks if layout has 'width' dimension
OPENVINO_API bool has_width(const Layout& layout);

/// \brief Returns 'width' dimension index.
///
/// \throws ov::AssertFailure if dimension doesn't exist.
///
OPENVINO_API std::int64_t width(const Layout& layout);

}  // namespace layout

template <>
class OPENVINO_API AttributeAdapter<Layout> : public ValueAccessor<std::string> {
public:
    explicit AttributeAdapter(Layout& value) : m_ref(value) {}

    const std::string& get() override;
    void set(const std::string& value) override;
    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<Layout>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
    explicit operator Layout&() {
        return m_ref;
    }

protected:
    Layout& m_ref;
    std::string m_dump;
};

template <>
class OPENVINO_API VariantWrapper<Layout> : public VariantImpl<Layout> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::Layout", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    explicit VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

}  // namespace ov
