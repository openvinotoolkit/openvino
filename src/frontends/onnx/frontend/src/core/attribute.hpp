// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include "core/sparse_tensor.hpp"
#include "core/tensor.hpp"
#include "ngraph/except.hpp"

namespace ngraph {
namespace onnx_import {
// forward declarations
class Graph;
class Subgraph;
class Model;

// Detecting automatically the underlying type used to store the information
// for data type of values an attribute is holding. A bug was discovered in
// protobuf which forced ONNX team to switch from `enum AttributeProto_AttributeType`
// to `int32` in order to workaround the bug. This line allows using both versions
// of ONNX generated wrappers.
using AttributeProto_AttributeType = decltype(ONNX_NAMESPACE::AttributeProto{}.type());

namespace error {
namespace attribute {
namespace detail {
struct Attribute : ngraph_error {
    Attribute(const std::string& msg, AttributeProto_AttributeType type) : ngraph_error{msg} {}
};

}  // namespace detail

struct InvalidData : detail::Attribute {
    explicit InvalidData(AttributeProto_AttributeType type) : Attribute{"invalid attribute type", type} {}
};

struct UnsupportedType : detail::Attribute {
    explicit UnsupportedType(AttributeProto_AttributeType type) : Attribute{"unsupported attribute type", type} {}
};

}  // namespace attribute

}  // namespace error

namespace detail {
namespace attribute {
template <typename T>
inline T get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    throw ngraph::onnx_import::error::attribute::UnsupportedType{attribute.type()};
}

template <>
inline float get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
        return static_cast<float>(attribute.i());
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
        return attribute.f();
    default:
        throw error::attribute::InvalidData{attribute.type()};
    }
}

template <>
inline std::vector<float> get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
        return {static_cast<float>(attribute.i())};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
        return {std::begin(attribute.floats()), std::end(attribute.floats())};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
        return {attribute.f()};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS:
        return {std::begin(attribute.floats()), std::end(attribute.floats())};
    default:
        throw error::attribute::InvalidData{attribute.type()};
    }
}

template <>
inline double get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
        return static_cast<double>(attribute.f());
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
        return static_cast<double>(attribute.i());
    default:
        throw error::attribute::InvalidData{attribute.type()};
    }
}

template <>
inline std::vector<double> get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4244)
#endif
    switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
        return {static_cast<double>(attribute.i())};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
        return {std::begin(attribute.ints()), std::end(attribute.ints())};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
        return {static_cast<double>(attribute.f())};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS:
        return {std::begin(attribute.floats()), std::end(attribute.floats())};
    default:
        throw error::attribute::InvalidData{attribute.type()};
    }
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}

template <>
inline std::size_t get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    if (attribute.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INT) {
        throw error::attribute::InvalidData{attribute.type()};
    }
    return static_cast<std::size_t>(attribute.i());
}

template <>
inline std::vector<std::size_t> get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
        return {static_cast<std::size_t>(attribute.i())};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
        return {std::begin(attribute.ints()), std::end(attribute.ints())};
    default:
        throw error::attribute::InvalidData{attribute.type()};
    }
}

template <>
inline int64_t get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    if (attribute.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INT) {
        throw error::attribute::InvalidData{attribute.type()};
    }
    return attribute.i();
}

template <>
inline std::vector<int64_t> get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
        return {attribute.i()};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
        return {std::begin(attribute.ints()), std::end(attribute.ints())};
    default:
        throw error::attribute::InvalidData{attribute.type()};
    }
}

template <>
inline std::string get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    if (attribute.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_STRING) {
        throw error::attribute::InvalidData{attribute.type()};
    }
    return attribute.s();
}

template <>
inline std::vector<std::string> get_value(const ONNX_NAMESPACE::AttributeProto& attribute) {
    switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
        return {attribute.s()};
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS:
        return {std::begin(attribute.strings()), std::end(attribute.strings())};
    default:
        throw error::attribute::InvalidData{attribute.type()};
    }
}

}  // namespace attribute

}  // namespace detail

class Attribute {
public:
    enum class Type {
        undefined = ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED,
        float_point = ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT,
        integer = ONNX_NAMESPACE::AttributeProto_AttributeType_INT,
        string = ONNX_NAMESPACE::AttributeProto_AttributeType_STRING,
        tensor = ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR,
        graph = ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH,
        sparse_tensor = ONNX_NAMESPACE::AttributeProto_AttributeType_SPARSE_TENSOR,
        float_point_array = ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS,
        integer_array = ONNX_NAMESPACE::AttributeProto_AttributeType_INTS,
        string_array = ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS,
        tensor_array = ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS,
        sparse_tensor_array = ONNX_NAMESPACE::AttributeProto_AttributeType_SPARSE_TENSORS,
        graph_array = ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS
    };

    Attribute() = delete;
    explicit Attribute(const ONNX_NAMESPACE::AttributeProto& attribute_proto, const std::string& model_dir)
        : m_attribute_proto{&attribute_proto},
          m_model_dir{model_dir} {}

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
        return Tensor{m_attribute_proto->t(), m_model_dir};
    }
    SparseTensor get_sparse_tensor() const {
        return SparseTensor{m_attribute_proto->sparse_tensor(), m_model_dir};
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
    Subgraph get_subgraph(const Graph* parent_graph) const;

    std::vector<Tensor> get_tensor_array() const {
        std::vector<Tensor> ret;
        const auto& tensors = m_attribute_proto->tensors();
        ret.reserve(tensors.size());
        for (const auto& tensor : tensors)
            ret.emplace_back(tensor, m_model_dir);
        return ret;
    }

    std::vector<SparseTensor> get_sparse_tensor_array() const {
        std::vector<SparseTensor> ret;
        const auto& sparse_tensors = m_attribute_proto->sparse_tensors();
        ret.reserve(sparse_tensors.size());
        for (const auto& tensor : sparse_tensors)
            ret.emplace_back(tensor, m_model_dir);
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

    /* explicit */ operator ONNX_NAMESPACE::AttributeProto_AttributeType() const {
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
            return Tensor{m_attribute_proto->t(), m_model_dir};
        }
        throw error::attribute::InvalidData{m_attribute_proto->type()};
    }

    template <typename T, typename std::enable_if<std::is_same<T, std::vector<Tensor>>::value, bool>::type = true>
    T get_value() const {
        if (is_tensor()) {
            return {Tensor{m_attribute_proto->t(), m_model_dir}};
        } else if (is_tensor_array()) {
            return get_tensor_array();
        }
        throw error::attribute::InvalidData{m_attribute_proto->type()};
    }

    template <typename T, typename std::enable_if<std::is_same<T, SparseTensor>::value, bool>::type = true>
    T get_value() const {
        if (is_sparse_tensor()) {
            return SparseTensor{m_attribute_proto->sparse_tensor(), m_model_dir};
        }
        throw error::attribute::InvalidData{m_attribute_proto->type()};
    }

    template <typename T, typename std::enable_if<std::is_same<T, std::vector<SparseTensor>>::value, bool>::type = true>
    T get_value() const {
        if (is_sparse_tensor()) {
            return {SparseTensor{m_attribute_proto->sparse_tensor(), m_model_dir}};
        } else if (is_sparse_tensor_array()) {
            return get_sparse_tensor_array();
        }
        throw error::attribute::InvalidData{m_attribute_proto->type()};
    }

    ov::Any get_any() const;

private:
    const ONNX_NAMESPACE::AttributeProto* m_attribute_proto;
    std::string m_model_dir;
};

}  // namespace onnx_import

}  // namespace ngraph
