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

class OPENVINO_API LayoutRank {
public:
    using value_type = int64_t;

    LayoutRank() = default;

    bool operator==(const LayoutRank& rhs) const {
        return m_dynamic == rhs.m_dynamic && m_right == rhs.m_right && m_left == rhs.m_left && m_size == rhs.m_size;
    }

    bool operator!=(const LayoutRank& rhs) const {
        return !(*this == rhs);
    }

    static LayoutRank create_static(value_type size) {
        return LayoutRank(size);
    }

    static LayoutRank create_dynamic(value_type left = 0, value_type right = 0) {
        return {left, right};
    }

    bool is_dynamic() const {
        return m_dynamic;
    }

    value_type size() const {
        return m_size;
    }

    value_type size_left() const {
        return m_left;
    }

    value_type size_right() const {
        return m_right;
    }

    static const value_type no_size = static_cast<value_type>(-1);

private:
    /// \brief
    LayoutRank(value_type left, value_type right) : m_dynamic(true), m_left(left), m_right(right) {}

    explicit LayoutRank(value_type size) : m_dynamic(false), m_left(no_size), m_right(no_size), m_size(size) {}

    bool m_dynamic = true;
    value_type m_left = 0;
    value_type m_right = 0;
    value_type m_size = no_size;
};

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
    Layout(const std::string& layoutStr);

    /// \brief Comparison operator (equal)
    bool operator==(const Layout& rhs) const;

    /// \brief Comparison operator (not equal)
    bool operator!=(const Layout& rhs) const;

    /// \brief Checks if dimension with specified name is in layout
    /// \return `true` if layout has information about dimension index with a given name
    bool has_name(const std::string& dimensionName) const;

    /// \brief Gets index of dimension with a specified name
    /// \return Index of given dimension name
    std::int64_t get_index_by_name(const std::string& dimensionName) const;

    /// \brief Gets name of dimension at specified index
    ///
    /// \note: For layouts with dynamic rank, like 'NC...HW' negative index specifies position of dimension on the
    /// right.
    ///
    /// \code{.cpp} auto name = Layout("NC...HW).get_name_by_index(-2); //returns "H" \endcode
    ///
    /// \throw ov::AssertFailure if dimension is neither defined, not specified as '?'
    ///
    /// \return Index of given dimension name. Empty string if dimension name is not defined, e.g. "?"
    std::string get_name_by_index(std::int64_t index) const;

    /// \brief Sets index of dimension by a specified name
    void set_name_for_index(const std::string& dimensionName, std::int64_t index);

    /// \brief Checks whether layout is scalar
    /// \return `true` if layout is scalar
    bool is_scalar() const;

    /// \brief Returns rank/size of layout. E.g. for Layout("NCHW").rank() returns 4
    LayoutRank rank() const;

    /// \brief String representation of Layout
    std::string to_string() const;

private:
    /// stores dimension names map to index in a layout
    std::unordered_map<std::string, std::int64_t> m_names;

    /// special case for scalar
    bool m_scalar = false;

    /// layout rank
    LayoutRank m_rank;
};

namespace layouts {

/// \brief Predefined layout dimension names
enum class PredefinedDim {
    UNDEFINED,  /// \brief undefined standard name
    BATCH,      /// \brief Batch dimension, related to name 'N'
    CHANNELS,   /// \brief Channels dimension, related to name 'N'
    DEPTH,      /// \brief Depth dimension, related to name 'N'
    WIDTH,      /// \brief Width dimension, related to name 'N'
    HEIGHT      /// \brief Height dimension, related to name 'N'
};

/// \brief Returns one-character string representing standard dimension name
OPENVINO_API std::string predefined_name(PredefinedDim dim);

/// \brief Converts dimension name to predefined enum. Returns 'UNDEFINED' if no match is found
OPENVINO_API PredefinedDim to_predefined_dim(const std::string& predef_name);

/// \brief Checks if layout has 'batch' dimension
OPENVINO_API bool has_batch(const Layout& layout);

/// \brief Returns 'batch' dimension index. Throws if dimension doesn't exist
OPENVINO_API std::int64_t batch(const Layout& layout);

/// \brief Sets 'batch' dimension index to layout
OPENVINO_API void set_batch(Layout& layout, std::int64_t index);

/// \brief Checks if layout has 'channels' dimension
OPENVINO_API bool has_channels(const Layout& layout);

/// \brief Returns 'channels' dimension index. Throws if dimension doesn't exist
OPENVINO_API std::int64_t channels(const Layout& layout);

/// \brief Sets 'channels' dimension index to layout
OPENVINO_API void set_channels(Layout& layout, std::int64_t index);

/// \brief Checks if layout has 'depth' dimension
OPENVINO_API bool has_depth(const Layout& layout);

/// \brief Returns 'depth' dimension index. Throws if dimension doesn't exist
OPENVINO_API std::int64_t depth(const Layout& layout);

/// \brief Sets 'depth' dimension index to layout
OPENVINO_API void set_depth(Layout& layout, std::int64_t index);

/// \brief Checks if layout has 'height' dimension
OPENVINO_API bool has_height(const Layout& layout);

/// \brief Returns 'height' dimension index. Throws if dimension doesn't exist
OPENVINO_API std::int64_t height(const Layout& layout);

/// \brief Sets 'height' dimension index to layout
OPENVINO_API void set_height(Layout& layout, std::int64_t index);

/// \brief Checks if layout has 'width' dimension
OPENVINO_API bool has_width(const Layout& layout);

/// \brief Returns 'width' dimension index. Throws if dimension doesn't exist
OPENVINO_API std::int64_t width(const Layout& layout);

/// \brief Sets 'width' dimension index to layout
OPENVINO_API void set_width(Layout& layout, std::int64_t index);
}  // namespace layouts

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

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

}  // namespace ov
