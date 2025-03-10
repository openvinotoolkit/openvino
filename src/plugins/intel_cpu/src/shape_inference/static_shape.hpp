// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "cpu_types.h"
#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/shape.hpp"
#include "shape_infer_type_utils.hpp"
#include "static_dimension.hpp"

namespace ov {
namespace op {
struct AutoBroadcastSpec;
}  // namespace op

namespace intel_cpu {
/**
 * @brief Main template for conditional static shape adapter which holds reference or container with CPU dimensions.
 */
template <class TDims>
class StaticShapeAdapter {};

using StaticShapeRef = StaticShapeAdapter<const VectorDims>;
using StaticShape = StaticShapeAdapter<VectorDims>;

template <class T>
constexpr bool is_static_shape_adapter() {
    using U = std::decay_t<T>;
    return std::is_same_v<U, StaticShapeRef> || std::is_same_v<U, StaticShape>;
}

/**
 * @brief The static shape adapter by copy value to VectorDims.
 *
 * This adapter is read/write stored VectorDims.
 */
template <>
class StaticShapeAdapter<VectorDims> {
    using TDims = VectorDims;
    using dim_type = typename TDims::value_type;

public:
    using ShapeContainer = StaticShape;  // used for ov::result_shape_t shape infer trait

    using value_type = StaticDimension;
    using iterator = typename TDims::iterator;
    using const_iterator = typename TDims::const_iterator;

    static_assert(std::is_same_v<dim_type, typename StaticDimension::value_type>,
                  "Static dimension must be of the same type as the CPU dimension.");
    static_assert(std::is_standard_layout_v<StaticDimension>,
                  "StaticShape must be standard layout to cast on CPU dimension type.");
    static_assert(sizeof(dim_type) == sizeof(StaticDimension),
                  "StaticDimension must have the same number of bytes as the CPU dimension type.");

    StaticShapeAdapter();
    StaticShapeAdapter(const TDims& dims);
    StaticShapeAdapter(TDims&& dims) noexcept;
    StaticShapeAdapter(std::initializer_list<value_type> dims) noexcept : m_dims{dims.begin(), dims.end()} {};
    StaticShapeAdapter(std::vector<value_type> dims) noexcept : m_dims(dims.begin(), dims.end()) {}

    StaticShapeAdapter(const StaticShape& other);
    StaticShapeAdapter(const ov::PartialShape&);

    const TDims& operator*() const& noexcept {
        return m_dims;
    }

    TDims& operator*() & noexcept {
        return m_dims;
    }

    TDims&& operator*() && noexcept {
        return std::move(m_dims);
    }

    value_type& operator[](size_t i) {
        return reinterpret_cast<value_type&>(m_dims[i]);
    }

    const value_type& operator[](size_t i) const {
        return reinterpret_cast<const value_type&>(m_dims[i]);
    }

    //-- Shape functions
    static constexpr bool is_static() {
        return true;
    }

    static constexpr bool is_dynamic() {
        return !is_static();
    }

    template <class T>
    constexpr std::enable_if_t<is_static_shape_adapter<T>(), bool> compatible(const T& other) const {
        // for static shape compatible == both shape equals
        return *this == other;
    }

    template <class T>
    constexpr std::enable_if_t<is_static_shape_adapter<T>(), bool> same_scheme(const T& other) const {
        // for static shape same_scheme == compatible;
        return compatible(other);
    }

    [[nodiscard]] ov::Rank rank() const;
    bool merge_rank(const ov::Rank& r);
    [[nodiscard]] ov::Shape to_shape() const;
    [[nodiscard]] ov::Shape get_max_shape() const;
    [[nodiscard]] ov::Shape get_min_shape() const;
    [[nodiscard]] ov::Shape get_shape() const;
    [[nodiscard]] ov::PartialShape to_partial_shape() const;

    static bool merge_into(StaticShapeAdapter& dst, const StaticShapeAdapter& src);
    static bool broadcast_merge_into(StaticShapeAdapter& dst,
                                     const StaticShapeAdapter& src,
                                     const ov::op::AutoBroadcastSpec& autob);

    //-- Container functions
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return m_dims.cbegin();
    }

    [[nodiscard]] const_iterator begin() const noexcept {
        return cbegin();
    }

    iterator begin() noexcept {
        return m_dims.begin();
    }

    [[nodiscard]] const_iterator cend() const noexcept {
        return m_dims.cend();
    }

    [[nodiscard]] const_iterator end() const noexcept {
        return cend();
    }

    iterator end() noexcept {
        return m_dims.end();
    }

    [[nodiscard]] size_t size() const {
        return m_dims.size();
    }

    [[nodiscard]] bool empty() const {
        return m_dims.empty();
    }

    void resize(size_t n) {
        m_dims.resize(n);
    }

    void reserve(size_t n) {
        m_dims.reserve(n);
    }

    iterator insert(iterator position, const value_type& value) {
        return m_dims.insert(position, value.get_length());
    }

    void insert(iterator position, size_t n, const value_type& val) {
        m_dims.insert(position, n, val.get_length());
    }

    template <class InputIterator>
    void insert(iterator position, InputIterator first, InputIterator last) {
        m_dims.insert(position, first, last);
    }

    void push_back(const dim_type& value) {
        m_dims.push_back(value);
    }

    void push_back(const value_type& value) {
        m_dims.push_back(value.get_length());
    }

    template <class... Args>
    void emplace_back(Args&&... args) {
        m_dims.emplace_back(value_type(std::forward<Args>(args)...));
    }

private:
    TDims m_dims;
};

/**
 * @brief The static shape adapter by reference to VectorDims.
 *
 * This adapter is read-only access for VectorDims.
 */
template <>
class StaticShapeAdapter<const VectorDims> {
    using TDims = VectorDims;
    using dim_type = typename VectorDims::value_type;

public:
    using ShapeContainer = StaticShape;  // used for ov::result_shape_t shape infer trait

    using value_type = StaticDimension;
    using iterator = typename TDims::const_iterator;
    using const_iterator = typename TDims::const_iterator;

    static_assert(std::is_same_v<dim_type, typename StaticDimension::value_type>,
                  "Static dimension must be of the same type as the CPU dimension.");
    static_assert(std::is_standard_layout_v<StaticDimension>,
                  "StaticShape must be standard layout to cast on CPU dimension type.");
    static_assert(sizeof(dim_type) == sizeof(StaticDimension),
                  "StaticDimension must have the same number of bytes as the CPU dimension type.");

    constexpr StaticShapeAdapter() = default;
    constexpr StaticShapeAdapter(const TDims& dims) : m_dims{&dims} {}
    constexpr StaticShapeAdapter(const StaticShapeAdapter<const TDims>& other) = default;

    StaticShapeAdapter(const StaticShape& shape);
    StaticShapeAdapter(const ov::PartialShape&);

    operator StaticShape() const {
        return m_dims ? StaticShape(*m_dims) : StaticShape();
    }

    const TDims& operator*() const& noexcept {
        return *m_dims;
    }

    const value_type& operator[](size_t i) const {
        return reinterpret_cast<const value_type&>((*m_dims)[i]);
    }

    //-- Shape functions
    static constexpr bool is_static() {
        return true;
    }

    static constexpr bool is_dynamic() {
        return !is_static();
    }

    template <class T>
    constexpr std::enable_if_t<is_static_shape_adapter<T>(), bool> compatible(const T& other) const {
        // for static shape compatible == both shape equals
        return *this == other;
    }

    template <class T>
    constexpr std::enable_if_t<is_static_shape_adapter<T>(), bool> same_scheme(const T& other) const {
        // for static shape same_scheme == compatible;
        return compatible(other);
    }

    [[nodiscard]] ov::Rank rank() const;
    bool merge_rank(const ov::Rank& r);
    [[nodiscard]] ov::Shape to_shape() const;
    [[nodiscard]] ov::Shape get_max_shape() const;
    [[nodiscard]] ov::Shape get_min_shape() const;
    [[nodiscard]] ov::Shape get_shape() const;
    [[nodiscard]] ov::PartialShape to_partial_shape() const;

    //-- Container functions
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return m_dims ? m_dims->cbegin() : const_iterator{};
    }

    [[nodiscard]] const_iterator begin() const noexcept {
        return cbegin();
    }

    [[nodiscard]] const_iterator cend() const noexcept {
        return m_dims ? m_dims->cend() : const_iterator{};
    }

    [[nodiscard]] const_iterator end() const noexcept {
        return cend();
    }

    [[nodiscard]] size_t size() const noexcept {
        return m_dims ? m_dims->size() : 0;
    }

    [[nodiscard]] bool empty() const {
        return m_dims ? m_dims->empty() : true;
    }

private:
    const TDims* m_dims = nullptr;
};

template <class T>
std::enable_if_t<is_static_shape_adapter<T>(), std::ostream&> operator<<(std::ostream& out, const T& shape) {
    out << '{';
    if (!shape.empty()) {
        std::copy(shape.cbegin(), shape.cend() - 1, std::ostream_iterator<StaticDimension>(out, ","));
        out << shape[shape.size() - 1];
    }
    out << '}';
    return out;
}

template <class T, class U>
constexpr std::enable_if_t<is_static_shape_adapter<T>() && is_static_shape_adapter<U>(), bool> operator==(
    const T& lhs,
    const U& rhs) {
    // The CPU dimension type and StaticDimension::value_type is same,
    // use CPU dimension type to compare in order to reduce number of conversions to StaticDimension.
    return (lhs.size() == rhs.size()) && (lhs.empty() || std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
}
}  // namespace intel_cpu

/**
 * @brief Specialization to throw the `NodeValidationFailure` for shape inference using `StaticShape`
 *
 * @param check_loc_info Exception location details to print.
 * @param ctx            NodeValidationFailure context which got pointer to node and input shapes used for shape
 * inference.
 * @param explanation    Exception explanation string.
 */
template <>
void NodeValidationFailure::create(const char* file,
                                   int line,
                                   const char* check_string,
                                   std::pair<const Node*, const std::vector<intel_cpu::StaticShape>*>&& ctx,
                                   const std::string& explanation);

/**
 * @brief Specialization to throw the `NodeValidationFailure` for shape inference using `StaticShapeRef`
 *
 * @param check_loc_info Exception location details to print.
 * @param ctx            NodeValidationFailure context which got pointer to node and input shapes used for shape
 * inference.
 * @param explanation    Exception explanation string.
 */
template <>
void NodeValidationFailure::create(const char* file,
                                   int line,
                                   const char* check_string,
                                   std::pair<const Node*, const std::vector<intel_cpu::StaticShapeRef>*>&& ctx,
                                   const std::string& explanation);
}  // namespace ov
