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
class Const;
class Concat;
class Unpack;
class Permute;
class Convert;
class Gather;
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
                                   ov::npuw::weights::op::Convert,
                                   ov::npuw::weights::op::Gather>;

    LazyTensor() = default;
    LazyTensor(const std::shared_ptr<ov::op::v0::Constant>& const_ptr);
    LazyTensor(const std::vector<LazyTensor>& to_concat, const std::size_t axis);  // construct from concat
    LazyTensor(const LazyTensor& cw,
               const LazyTensor& cz,
               const LazyTensor& cs,
               const ov::element::Type& type,
               const ov::Shape& shape);  // construct from unpack
    LazyTensor(const LazyTensor& cw,
               const ov::Tensor& t,
               const ov::element::Type& type,
               const ov::Shape& shape);  // construct from nf4_gather

    LazyTensor permute(const std::vector<std::size_t>& axes);
    LazyTensor convert(const ov::element::Type& type);

    bool operator==(const LazyTensor& other) const;
    bool operator!=(const LazyTensor& other) const;

    ov::Tensor eval() const;
    std::size_t get_hash() const;
    std::vector<Transform> get_transformations() const;
    void detach();

    struct Meta {
        ov::Shape shape;
        ov::element::Type type;
    };
    Meta eval_meta() const;

    void serialize(std::ostream& stream) const;
    static LazyTensor deserialize(std::istream& stream);
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);
    operator bool() const;

private:
    std::shared_ptr<LazyTensorImpl> m_impl = nullptr;
};

namespace op {
class Const {
    friend struct ov::npuw::weights::LazyTensorImpl;

public:
    Const() = default;

    explicit Const(std::shared_ptr<ov::op::v0::Constant> n);
    std::size_t hash() const;
    bool operator==(const Const& other) const;
    ov::Tensor eval() const;
    LazyTensor::Meta eval_meta() const;
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);
    void detach();
    void serialize(std::ostream& stream) const;
    static Const deserialize(std::istream& stream);

private:
    std::shared_ptr<ov::op::v0::Constant> m_node = nullptr;
    ov::element::Type m_cached_type;
    ov::Shape m_cached_shape;
    const void* m_cached_ptr = nullptr;
    std::size_t m_offset = 0;
    std::size_t m_byte_size = 0;
    ov::Tensor m_read_from_bin;
};

class Concat {
    friend struct ov::npuw::weights::LazyTensorImpl;

public:
    Concat() = default;
    Concat(const std::vector<LazyTensor>& _tensors, std::size_t _axis) : tensors(_tensors), axis(_axis) {}

    std::size_t hash() const;
    bool operator==(const Concat& other) const;
    ov::Tensor eval() const;
    LazyTensor::Meta eval_meta() const;
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);
    void detach();
    void serialize(std::ostream& stream) const;
    static Concat deserialize(std::istream& stream);

private:
    std::vector<LazyTensor> tensors;
    std::size_t axis = 0;
};

class Unpack {
    friend struct ov::npuw::weights::LazyTensorImpl;

public:
    Unpack() = default;
    Unpack(const LazyTensor& _w, const LazyTensor& _z, const LazyTensor& _s, ov::element::Type _type, ov::Shape _shape)
        : w(_w),
          z(_z),
          s(_s),
          type(_type),
          shape(_shape) {}

    std::size_t hash() const;
    bool operator==(const Unpack& other) const;
    ov::Tensor eval() const;
    LazyTensor::Meta eval_meta() const;
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);
    void detach();
    void serialize(std::ostream& stream) const;
    static Unpack deserialize(std::istream& stream);

private:
    LazyTensor w, z, s;
    ov::element::Type type;
    ov::Shape shape;
};

class Permute {
    friend struct ov::npuw::weights::LazyTensorImpl;

public:
    Permute() = default;
    Permute(const LazyTensor& _tensor, const std::vector<std::size_t>& _axes) : tensor(_tensor), axes(_axes) {}

    std::size_t hash() const;
    bool operator==(const Permute& other) const;
    ov::Tensor eval() const;
    LazyTensor::Meta eval_meta() const;
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);
    void detach();
    void serialize(std::ostream& stream) const;
    static Permute deserialize(std::istream& stream);

private:
    LazyTensor tensor;
    std::vector<std::size_t> axes;
};

class Convert {
    friend struct ov::npuw::weights::LazyTensorImpl;

public:
    Convert() = default;
    Convert(const LazyTensor& _tensor, ov::element::Type _type) : tensor(_tensor), type(_type) {}

    std::size_t hash() const;
    bool operator==(const Convert& other) const;
    ov::Tensor eval() const;
    LazyTensor::Meta eval_meta() const;
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);
    void detach();
    void serialize(std::ostream& stream) const;
    static Convert deserialize(std::istream& stream);

private:
    LazyTensor tensor;
    ov::element::Type type;
};

class Gather {
    friend struct ov::npuw::weights::LazyTensorImpl;

public:
    Gather() = default;
    Gather(const LazyTensor& _w, const ov::Tensor& _t, const ov::element::Type& _type, const ov::Shape& _shape)
        : w(_w),
          t(_t),
          type(_type),
          shape(_shape) {}

    std::size_t hash() const;
    bool operator==(const Gather& other) const;
    ov::Tensor eval() const;
    LazyTensor::Meta eval_meta() const;
    void read_weight(const ov::npuw::s11n::WeightsContext& ctx);
    void detach();
    void serialize(std::ostream& stream) const;
    static Gather deserialize(std::istream& stream);

private:
    LazyTensor w;
    ov::Tensor t;
    ov::element::Type type;
    ov::Shape shape;
};
}  // namespace op

}  // namespace weights
}  // namespace npuw
}  // namespace ov
