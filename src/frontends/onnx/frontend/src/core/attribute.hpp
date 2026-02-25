// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include "core/sparse_tensor.hpp"
#include "core/tensor.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace frontend {
namespace onnx {
// forward declarations
class Graph;
class Subgraph;
class Model;

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::AttributeProto_AttributeType;
using ::ONNX_NAMESPACE::AttributeProto_AttributeType_Name;

namespace detail {
namespace attribute {
template <typename T>
inline T get_value(const AttributeProto& attribute) {
    OPENVINO_THROW("Unsupported attribute type");
}

#define ONNX_INVALID_ATTR(attr, expected) \
    OPENVINO_THROW("Invalid attribute type ", AttributeProto_AttributeType_Name(attr), " expected: ", expected)

template <>
inline float get_value(const AttributeProto& attribute) {
    switch (attribute.type()) {
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INT:
        return static_cast<float>(attribute.i());
    case AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT:
        return attribute.f();
    default:
        ONNX_INVALID_ATTR(attribute.type(), "INT, FLOAT");
    }
}

template <>
inline std::vector<float> get_value(const AttributeProto& attribute) {
    switch (attribute.type()) {
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INT:
        return {static_cast<float>(attribute.i())};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INTS:
        return {std::begin(attribute.floats()), std::end(attribute.floats())};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT:
        return {attribute.f()};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS:
        return {std::begin(attribute.floats()), std::end(attribute.floats())};
    default:
        ONNX_INVALID_ATTR(attribute.type(), "INT, INTS, FLOAT, FLOATS");
    }
}

template <>
inline double get_value(const AttributeProto& attribute) {
    switch (attribute.type()) {
    case AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT:
        return static_cast<double>(attribute.f());
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INT:
        return static_cast<double>(attribute.i());
    default:
        ONNX_INVALID_ATTR(attribute.type(), "INT, FLOAT");
    }
}

template <>
inline std::vector<double> get_value(const AttributeProto& attribute) {
#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4244)
#endif
    switch (attribute.type()) {
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INT:
        return {static_cast<double>(attribute.i())};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INTS:
        return {std::begin(attribute.ints()), std::end(attribute.ints())};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT:
        return {static_cast<double>(attribute.f())};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS:
        return {std::begin(attribute.floats()), std::end(attribute.floats())};
    default:
        ONNX_INVALID_ATTR(attribute.type(), "INT, INTS, FLOAT, FLOATS");
    }
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}

template <>
inline std::size_t get_value(const AttributeProto& attribute) {
    if (attribute.type() != AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
        ONNX_INVALID_ATTR(attribute.type(), "INT");
    }
    return static_cast<std::size_t>(attribute.i());
}

template <>
inline std::vector<std::size_t> get_value(const AttributeProto& attribute) {
    switch (attribute.type()) {
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INT:
        return {static_cast<std::size_t>(attribute.i())};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INTS:
        return {std::begin(attribute.ints()), std::end(attribute.ints())};
    default:
        ONNX_INVALID_ATTR(attribute.type(), "INT, INTS");
    }
}

template <>
inline int64_t get_value(const AttributeProto& attribute) {
    if (attribute.type() != AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
        ONNX_INVALID_ATTR(attribute.type(), "INT");
    }
    return attribute.i();
}

template <>
inline std::vector<int64_t> get_value(const AttributeProto& attribute) {
    switch (attribute.type()) {
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INT:
        return {attribute.i()};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_INTS:
        return {std::begin(attribute.ints()), std::end(attribute.ints())};
    default:
        ONNX_INVALID_ATTR(attribute.type(), "INT, INTS");
    }
}

template <>
inline std::string get_value(const AttributeProto& attribute) {
    if (attribute.type() != AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        ONNX_INVALID_ATTR(attribute.type(), "STRING");
    }
    return attribute.s();
}

template <>
inline std::vector<std::string> get_value(const AttributeProto& attribute) {
    switch (attribute.type()) {
    case AttributeProto_AttributeType::AttributeProto_AttributeType_STRING:
        return {attribute.s()};
    case AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS:
        return {std::begin(attribute.strings()), std::end(attribute.strings())};
    default:
        ONNX_INVALID_ATTR(attribute.type(), "STRING, STRINGS");
    }
}

}  // namespace attribute
}  // namespace detail

class Attribute {
public:
    enum class Type {
        undefined = AttributeProto_AttributeType::AttributeProto_AttributeType_UNDEFINED,
        float_point = AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT,
        integer = AttributeProto_AttributeType::AttributeProto_AttributeType_INT,
        string = AttributeProto_AttributeType::AttributeProto_AttributeType_STRING,
        tensor = AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR,
        graph = AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH,
        sparse_tensor = AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSOR,
        float_point_array = AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS,
        integer_array = AttributeProto_AttributeType::AttributeProto_AttributeType_INTS,
        string_array = AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS,
        tensor_array = AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS,
        sparse_tensor_array = AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSORS,
        graph_array = AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPHS
    };

    Attribute() = delete;
    Attribute(const AttributeProto& attribute_proto,
              const std::string& model_dir,
              detail::MappedMemoryHandles mmap_cache)
        : m_attribute_proto{&attribute_proto},
          m_model_dir{model_dir},
          m_mmap_cache{mmap_cache} {}

    Attribute(Attribute&&) noexcept = default;
    Attribute(const Attribute&) = default;

    Attribute& operator=(Attribute&&) noexcept = delete;
    Attribute& operator=(const Attribute&) = delete;

    const std::string& get_name() const {
        return m_attribute_proto->name();
    }
    Type get_type() const {
        return static_cast<Type>(m_attribute_proto->type());
    }
    bool is_tensor() const {
        return get_type() == Type::tensor;
    }
    bool is_tensor_array() const {
        return get_type() == Type::tensor_array;
    }
    bool is_sparse_tensor() const {
        return get_type() == Type::sparse_tensor;
    }
    bool is_sparse_tensor_array() const {
        return get_type() == Type::sparse_tensor_array;
    }
    bool is_float() const {
        return get_type() == Type::float_point;
    }
    bool is_float_array() const {
        return get_type() == Type::float_point_array;
    }
    bool is_integer() const {
        return get_type() == Type::integer;
    }
    bool is_integer_array() const {
        return get_type() == Type::integer_array;
    }
    bool is_string() const {
        return get_type() == Type::string;
    }
    bool is_string_array() const {
        return get_type() == Type::string_array;
    }
    bool is_graph() const {
        return get_type() == Type::graph;
    }
    bool is_graph_array() const {
        return get_type() == Type::graph_array;
    }
    Tensor get_tensor() const {
        return Tensor{m_attribute_proto->t(), m_model_dir, m_mmap_cache};
    }
    SparseTensor get_sparse_tensor() const {
        return SparseTensor{m_attribute_proto->sparse_tensor(), m_model_dir, m_mmap_cache};
    }
    float get_float() const {
        return m_attribute_proto->f();
    }
    int64_t get_integer() const {
        return m_attribute_proto->i();
    }
    const std::string& get_string() const {
        return m_attribute_proto->s();
    }
    Subgraph get_subgraph(Graph* parent_graph) const;

    std::vector<Tensor> get_tensor_array() const {
        std::vector<Tensor> ret;
        const auto& tensors = m_attribute_proto->tensors();
        ret.reserve(tensors.size());
        for (const auto& tensor : tensors)
            ret.emplace_back(tensor, m_model_dir, m_mmap_cache);
        return ret;
    }

    std::vector<SparseTensor> get_sparse_tensor_array() const {
        std::vector<SparseTensor> ret;
        const auto& sparse_tensors = m_attribute_proto->sparse_tensors();
        ret.reserve(sparse_tensors.size());
        for (const auto& tensor : sparse_tensors)
            ret.emplace_back(tensor, m_model_dir, m_mmap_cache);
        return ret;
    }

    std::vector<float> get_float_array() const {
        return {std::begin(m_attribute_proto->floats()), std::end(m_attribute_proto->floats())};
    }

    std::vector<int64_t> get_integer_array() const {
        return {std::begin(m_attribute_proto->ints()), std::end(m_attribute_proto->ints())};
    }

    std::vector<std::string> get_string_array() const {
        return {std::begin(m_attribute_proto->strings()), std::end(m_attribute_proto->strings())};
    }

    /* explicit */ operator AttributeProto_AttributeType() const {
        return m_attribute_proto->type();
    }

    template <typename T,
              typename std::enable_if<!std::is_same<T, Tensor>::value && !std::is_same<T, std::vector<Tensor>>::value &&
                                          !std::is_same<T, SparseTensor>::value &&
                                          !std::is_same<T, std::vector<SparseTensor>>::value,
                                      bool>::type = true>
    T get_value() const {
        return detail::attribute::get_value<T>(*m_attribute_proto);
    }

    template <typename T, typename std::enable_if<std::is_same<T, Tensor>::value, bool>::type = true>
    T get_value() const {
        if (is_tensor()) {
            return Tensor{m_attribute_proto->t(), m_model_dir, m_mmap_cache};
        }
        ONNX_INVALID_ATTR(m_attribute_proto->type(), "TENSOR");
    }

    template <typename T, typename std::enable_if<std::is_same<T, std::vector<Tensor>>::value, bool>::type = true>
    T get_value() const {
        if (is_tensor()) {
            return {Tensor{m_attribute_proto->t(), m_model_dir, m_mmap_cache}};
        } else if (is_tensor_array()) {
            return get_tensor_array();
        }
        ONNX_INVALID_ATTR(m_attribute_proto->type(), "TENSOR, TENSORS");
    }

    template <typename T, typename std::enable_if<std::is_same<T, SparseTensor>::value, bool>::type = true>
    T get_value() const {
        if (is_sparse_tensor()) {
            return SparseTensor{m_attribute_proto->sparse_tensor(), m_model_dir, m_mmap_cache};
        }
        ONNX_INVALID_ATTR(m_attribute_proto->type(), "SPARSE_TENSOR");
    }

    template <typename T, typename std::enable_if<std::is_same<T, std::vector<SparseTensor>>::value, bool>::type = true>
    T get_value() const {
        if (is_sparse_tensor()) {
            return {SparseTensor{m_attribute_proto->sparse_tensor(), m_model_dir, m_mmap_cache}};
        } else if (is_sparse_tensor_array()) {
            return get_sparse_tensor_array();
        }
        ONNX_INVALID_ATTR(m_attribute_proto->type(), "SPARSE_TENSOR, SPARSE_TENSORS");
    }

    ov::Any get_any() const;

private:
    const AttributeProto* m_attribute_proto;
    std::string m_model_dir;
    detail::MappedMemoryHandles m_mmap_cache;
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
