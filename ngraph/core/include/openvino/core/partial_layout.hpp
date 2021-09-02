// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ov {
class OPENVINO_API PartialLayout {
public:
    /// \brief Constructs a static PartialLayout with no layout information.
    PartialLayout() = default;

    /// \brief Constructs a PartialLayout with static or dynamic layout information based
    /// on string representation.
    /// \param layoutStr The string used to construct PartialLayout from.
    /// The string representation can be in the following form:
    /// - can define order and meaning for dimensions "NCHW"
    /// - partial layout specialization:
    ///   - "NC?" defines 3 dimentional layout, first two NC, 3rd one is not defined
    ///   - "N..C" defines layout with dynamic rank where 1st dimension is N, last one is C
    ///   - "N..C" defines layout with dynamic rank where first two are NC, others are not
    ///   defined
    /// - only order of dimensions "adbc" (0312)
    PartialLayout(const char* layoutStr) : PartialLayout(std::string(layoutStr)) {}
    PartialLayout(const std::string& layoutStr);

    /// \brief Checks if dimension with specified name is in layout
    /// \return `true` if layout has information about dimension index with a given name
    bool has_dim(const std::string& dimensionName) const;

    /// \brief Gets index of dimension with a specified name
    /// \return Index of given dimension name
    std::int64_t get_index_by_name(const std::string& dimensionName) const;

    /// \brief Sets index of dimension by a specified name
    void set_dim_by_name(const std::string& dimensionName, std::int64_t index);

    /// \brief Checks whether layout is SCALAR
    /// \return `true` if layout is SCALAR
    bool is_scalar() const;

    bool is_empty() const {
        return _dimensionNames.empty();
    }
    std::size_t size() const {
        return _dimensionNames.size();
    }

private:
    /// stores dimension names
    std::unordered_map<std::string, std::int64_t> _dimensionNames;
};

namespace layouts {
OPENVINO_API bool has_batch(const PartialLayout& layout);
OPENVINO_API std::int64_t batch(const PartialLayout& layout);
OPENVINO_API void set_batch(PartialLayout& layout, std::int64_t index);

OPENVINO_API bool has_channels(const PartialLayout& layout);
OPENVINO_API std::int64_t channels(const PartialLayout& layout);
OPENVINO_API void set_channels(PartialLayout& layout, std::int64_t index);

OPENVINO_API bool has_depth(const PartialLayout& layout);
OPENVINO_API std::int64_t depth(const PartialLayout& layout);
OPENVINO_API void set_depth(PartialLayout& layout, std::int64_t index);

OPENVINO_API bool has_height(const PartialLayout& layout);
OPENVINO_API std::int64_t height(const PartialLayout& layout);
OPENVINO_API void set_height(PartialLayout& layout, std::int64_t index);

OPENVINO_API bool has_width(const PartialLayout& layout);
OPENVINO_API std::int64_t width(const PartialLayout& layout);
OPENVINO_API void set_width(PartialLayout& layout, std::int64_t index);
}  // namespace layouts

template <>
class OPENVINO_API AttributeAdapter<PartialLayout> : public ValueAccessor<std::string> {
public:
    AttributeAdapter(PartialLayout& value) : m_ref(value) {}

    const std::string& get() override;
    void set(const std::string& value) override;
    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<PartialLayout>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
    operator PartialLayout&() {
        return m_ref;
    }

protected:
    PartialLayout& m_ref;
};
}  // namespace ov
