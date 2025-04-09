// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <variant>

#include "openvino/op/constant.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/mmap_object.hpp"
#include "serialization.hpp"

namespace ov {
namespace npuw {
namespace weights {
// Forward declaration
class LazyTensor;
struct LazyTensorImpl;

namespace op {
struct Const;
struct Concat;
struct Unpack;
struct Permute;
struct Convert;
}  // namespace op

class LazyTensor {
public:
    class Hash {
    public:
        std::size_t operator()(const LazyTensor& lt) const;
    };

    using Transform = std::variant<ov::npuw::weights::op::Const,
                                   ov::npuw::weights::op::Concat,
                                   ov::npuw::weights::op::Unpack,
                                   ov::npuw::weights::op::Permute,
                                   ov::npuw::weights::op::Convert>;

    LazyTensor() = default;
    LazyTensor(const std::shared_ptr<ov::op::v0::Constant>& const_ptr);
    LazyTensor(const std::vector<LazyTensor>& to_concat, const std::size_t axis);  // construct from concat
    LazyTensor(const LazyTensor& cw,
               const LazyTensor& cz,
               const LazyTensor& cs,
               const ov::element::Type& type,
               const ov::Shape& shape);  // construct from unpack

    LazyTensor permute(const std::vector<std::size_t>& axes);
    LazyTensor convert(const ov::element::Type& type);

    bool operator==(const LazyTensor& other) const;
    bool operator!=(const LazyTensor& other) const;

    ov::Tensor eval() const;
    std::size_t get_hash() const;
    std::vector<Transform> get_transformations() const;
    void detach();

    void serialize(std::ostream& stream) const;
    static LazyTensor deserialize(std::istream& stream);
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);
    operator bool() const;

private:
    std::shared_ptr<LazyTensorImpl> m_impl = nullptr;
};

namespace op {
struct Const {
    std::shared_ptr<ov::op::v0::Constant> m_node = nullptr;
    ov::element::Type m_cached_type;
    ov::Shape m_cached_shape;
    const void* m_cached_ptr = nullptr;
    std::size_t m_offset = 0;
    std::size_t m_byte_size = 0;
    ov::Tensor m_read_from_bin;

    Const() = default;

    explicit Const(std::shared_ptr<ov::op::v0::Constant> n);
    std::size_t hash() const;
    bool operator==(const Const& other) const;
    ov::Tensor eval() const;
    void read_weight(const ov::npuw::s11n::Weights& weights);
    void detach();
    void serialize(std::ostream& stream) const;
    static Const deserialize(std::istream& stream);
};
struct Concat {
    std::vector<LazyTensor> tensors;
    std::size_t axis;

    Concat() = default;

    std::size_t hash() const;
    bool operator==(const Concat& other) const;
    ov::Tensor eval() const;
    void read_weight(const ov::npuw::s11n::Weights& weights);
    void detach();
    void serialize(std::ostream& stream) const;
    static Concat deserialize(std::istream& stream);
};
struct Unpack {
    LazyTensor w, z, s;
    ov::element::Type type;
    ov::Shape shape;

    Unpack() = default;

    std::size_t hash() const;
    bool operator==(const Unpack& other) const;
    ov::Tensor eval() const;
    void read_weight(const ov::npuw::s11n::Weights& weights);
    void detach();
    void serialize(std::ostream& stream) const;
    static Unpack deserialize(std::istream& stream);
};
struct Permute {
    LazyTensor tensor;
    std::vector<std::size_t> axes;

    Permute() = default;

    std::size_t hash() const;
    bool operator==(const Permute& other) const;
    ov::Tensor eval() const;
    void read_weight(const ov::npuw::s11n::Weights& weights);
    void detach();
    void serialize(std::ostream& stream) const;
    static Permute deserialize(std::istream& stream);
};
struct Convert {
    LazyTensor tensor;
    ov::element::Type type;

    Convert() = default;

    std::size_t hash() const;
    bool operator==(const Convert& other) const;
    ov::Tensor eval() const;
    void read_weight(const ov::npuw::s11n::Weights& weights);
    void detach();
    void serialize(std::ostream& stream) const;
    static Convert deserialize(std::istream& stream);
};
}  // namespace op

}  // namespace weights
}  // namespace npuw
}  // namespace ov
