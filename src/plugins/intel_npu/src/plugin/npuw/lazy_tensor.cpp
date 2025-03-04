// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

#include <tuple>
#include <type_traits>
#include <variant>

#include "logging.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

using ov::npuw::weights::LazyTensor;

namespace ov {
namespace npuw {
namespace weights {
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

    explicit Const(std::shared_ptr<ov::op::v0::Constant> n) : m_node(n) {
        m_cached_type = m_node->get_element_type();
        m_cached_shape = m_node->get_shape();
        m_cached_ptr = m_node->get_data_ptr();
        m_byte_size = m_node->get_byte_size();

        auto rt_info = m_node->get_rt_info();
        auto weightless_cache_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (weightless_cache_attr != rt_info.end()) {
            m_offset = weightless_cache_attr->second.as<ov::WeightlessCacheAttribute>().bin_offset;
        }
    }
    std::size_t hash() const {
        std::size_t seed = std::hash<const void*>()(m_cached_ptr) + 0x9e3779b9;
        seed ^= m_cached_type.hash() + 0x9e3779b9;
        for (const auto& dim : m_cached_shape) {
            seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
        }
        return seed;
    }
    bool operator==(const Const& other) const {
        return (m_cached_type == other.m_cached_type && m_cached_shape == other.m_cached_shape &&
                m_cached_ptr == other.m_cached_ptr);
    }
    ov::Tensor eval() const {
        if (m_node) {
            return ov::npuw::util::tensor_from_const(m_node);
        }

        NPUW_ASSERT(m_read_from_bin && "Underlying data should have been read first!");
        return m_read_from_bin;
    }
    void read_weight(const ov::npuw::s11n::Weights& weights) {
        NPUW_ASSERT(!m_node &&
                    "LazyTensor can only read weight when it's being deserialized and not created from a Constant!");
        m_read_from_bin = ov::Tensor(m_cached_type, m_cached_shape);
        std::memcpy(m_read_from_bin.data(), weights->get_ptr(m_offset), m_byte_size);
    }
    void detach() {
        m_node.reset();
        m_read_from_bin = ov::Tensor();
    }
    void serialize(std::ostream& stream) const {
        using namespace ov::npuw::s11n;
        write(stream, m_cached_type.to_string());
        write(stream, m_cached_shape);
        write(stream, m_offset);
        write(stream, m_byte_size);
    }
    static Const deserialize(std::istream& stream) {
        using namespace ov::npuw::s11n;
        Const c;
        std::string type_str;
        read(stream, type_str);
        c.m_cached_type = ov::element::Type(type_str);
        read(stream, c.m_cached_shape);
        read(stream, c.m_offset);
        read(stream, c.m_byte_size);
        return c;
    }
};
struct Concat {
    std::vector<LazyTensor> tensors;
    std::size_t axis;

    Concat() = default;

    std::size_t hash() const {
        std::size_t seed = std::hash<std::size_t>()(axis) + 0x9e3779b9;
        for (auto& lt : tensors) {
            seed ^= lt.get_hash() + 0x9e3779b9;
        }
        return seed;
    }
    bool operator==(const Concat& other) const {
        return (axis == other.axis && tensors == other.tensors);
    }
    ov::Tensor eval() const {
        std::vector<ov::Tensor> to_concat;
        for (const auto& lt : tensors) {
            to_concat.push_back(lt.eval());
        }
        return ov::npuw::util::concat(to_concat, axis);
    }
    void read_weight(const ov::npuw::s11n::Weights& weights) {
        for (auto& lt : tensors) {
            lt.read_weight(weights);
        }
    }
    void detach() {
        for (auto&& lt : tensors) {
            lt.detach();
        }
    }
    void serialize(std::ostream& stream) const {
        using namespace ov::npuw::s11n;
        write(stream, axis);
        write(stream, tensors);
    }
    static Concat deserialize(std::istream& stream) {
        using namespace ov::npuw::s11n;
        Concat c;
        read(stream, c.axis);
        read(stream, c.tensors);
        return c;
    }
};
struct Unpack {
    LazyTensor w, z, s;
    ov::element::Type type;
    ov::Shape shape;

    Unpack() = default;

    std::size_t hash() const {
        std::size_t seed = w.get_hash() + 0x9e3779b9;
        seed ^= z.get_hash() + 0x9e3779b9;
        seed ^= s.get_hash() + 0x9e3779b9;
        seed ^= type.hash() + 0x9e3779b9;
        for (const auto& dim : shape) {
            seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
        }
        return seed;
    }
    bool operator==(const Unpack& other) const {
        return (type == other.type && shape == other.shape && w == other.w && z == other.z && s == other.s);
    }
    ov::Tensor eval() const {
        const auto& gti = ov::get_tensor_impl;
        const auto& tw = w.eval();
        const auto& tz = z.eval();
        const auto& ts = s.eval();
        NPUW_ASSERT(tw);
        ov::Tensor dst(type, shape);
        if (tw && tz && ts) {
            ov::npuw::util::unpack(gti(tw), gti(tz), gti(ts), gti(dst));
        } else if (tw && ts) {
            ov::npuw::util::unpack(gti(tw), gti(ts), gti(dst));
        } else {
            NPUW_ASSERT(false && "Unsupported combination");
        }
        return dst;
    }
    void read_weight(const ov::npuw::s11n::Weights& weights) {
        w.read_weight(weights);
        if (z) {  // could be empty
            z.read_weight(weights);
        }
        s.read_weight(weights);
    }
    void detach() {
        w.detach();
        z.detach();
        s.detach();
    }
    void serialize(std::ostream& stream) const {
        using namespace ov::npuw::s11n;
        write(stream, type.to_string());
        write(stream, shape);
        write(stream, w);
        write(stream, z);
        write(stream, s);
    }
    static Unpack deserialize(std::istream& stream) {
        using namespace ov::npuw::s11n;
        Unpack u;
        std::string type_str;
        read(stream, type_str);
        u.type = ov::element::Type(type_str);
        read(stream, u.shape);
        read(stream, u.w);
        read(stream, u.z);
        read(stream, u.s);
        return u;
    }
};
struct Permute {
    LazyTensor tensor;
    std::vector<std::size_t> axes;

    Permute() = default;

    std::size_t hash() const {
        std::size_t seed = tensor.get_hash() + 0x9e3779b9;
        for (const auto& axis : axes) {
            seed ^= std::hash<std::size_t>()(axis) + 0x9e3779b9;
        }
        return seed;
    }
    bool operator==(const Permute& other) const {
        return (axes == other.axes && tensor == other.tensor);
    }
    ov::Tensor eval() const {
        return ov::npuw::util::permute(tensor.eval(), axes);
    }
    void read_weight(const ov::npuw::s11n::Weights& weights) {
        tensor.read_weight(weights);
    }
    void detach() {
        tensor.detach();
    }
    void serialize(std::ostream& stream) const {
        using namespace ov::npuw::s11n;
        write(stream, axes);
        write(stream, tensor);
    }
    static Permute deserialize(std::istream& stream) {
        using namespace ov::npuw::s11n;
        Permute p;
        read(stream, p.axes);
        read(stream, p.tensor);
        return p;
    }
};
struct Convert {
    LazyTensor tensor;
    ov::element::Type type;

    Convert() = default;

    std::size_t hash() const {
        std::size_t seed = type.hash() + 0x9e3779b9;
        seed ^= tensor.get_hash() + 0x9e3779b9;
        return seed;
    }
    bool operator==(const Convert& other) const {
        return (type == other.type && tensor == other.tensor);
    }
    ov::Tensor eval() const {
        NPUW_ASSERT(ov::element::f16 == type);
        return ov::npuw::util::to_f16(tensor.eval());
    }
    void read_weight(const ov::npuw::s11n::Weights& weights) {
        tensor.read_weight(weights);
    }
    void detach() {
        tensor.detach();
    }
    void serialize(std::ostream& stream) const {
        using namespace ov::npuw::s11n;
        write(stream, type.to_string());
        write(stream, tensor);
    }
    static Convert deserialize(std::istream& stream) {
        using namespace ov::npuw::s11n;
        Convert c;
        std::string type_str;
        read(stream, type_str);
        c.type = ov::element::Type(type_str);
        read(stream, c.tensor);
        return c;
    }
};
}  // namespace op

using Transform = std::variant<op::Const, op::Concat, op::Unpack, op::Permute, op::Convert>;

enum class TransformType : int { CONST = 0, CONCAT, UNPACK, PERMUTE, CONVERT };

struct LazyTensorImpl {
public:
    LazyTensorImpl() = default;
    explicit LazyTensorImpl(Transform&& t);
    bool operator==(const LazyTensorImpl& other) const;

    ov::Tensor eval() const;
    std::size_t get_hash() const;

    void detach();

    void serialize(std::ostream& stream) const;
    static std::shared_ptr<LazyTensorImpl> deserialize(std::istream& stream);
    void read_weight(const ov::npuw::s11n::Weights& weights);

    Transform m_transform;
    std::size_t m_hash = 0;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov

using namespace ov::npuw::weights::op;
using ov::npuw::weights::LazyTensorImpl;
using ov::npuw::weights::Transform;

// std::visit helper
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

LazyTensorImpl::LazyTensorImpl(Transform&& t)
    : m_transform(std::move(t)),
      m_hash(std::visit(overloaded{[](const auto& op) {
                            return op.hash();
                        }},
                        m_transform)) {}

bool LazyTensorImpl::operator==(const LazyTensorImpl& other) const {
    return m_hash == other.m_hash && m_transform == other.m_transform;
}

ov::Tensor LazyTensorImpl::eval() const {
    /* FIXME:
    Consider case:
        model1: concat->permute->f16
        model2: permute->f16
    Due to different history of transformation new tensors will be allocated for model2.
    However, we could avoid it by introducing a proper slicing on top of known axes and
    some kind of indicator that the only difference is concat and we should look for an existing ov::Tensor.
    Perhaps it should be done after model compilation and not handled here.
    */
    return std::visit(overloaded{[](const auto& op) {
                          return op.eval();
                      }},
                      m_transform);
}

void LazyTensorImpl::read_weight(const ov::npuw::s11n::Weights& weights) {
    std::visit(overloaded{[&weights](auto& op) {
                   return op.read_weight(weights);
               }},
               m_transform);
}

std::size_t LazyTensorImpl::get_hash() const {
    return m_hash;
}

void LazyTensorImpl::detach() {
    std::visit(overloaded{[](auto& op) {
                   op.detach();
               }},
               m_transform);
}

void LazyTensorImpl::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, m_hash);
    // FIXME: create proper op identificators instead of int
    std::visit(overloaded{[&stream](const op::Concat& op) {
                              write(stream, static_cast<int>(TransformType::CONCAT));
                              op.serialize(stream);
                          },
                          [&stream](const op::Const& op) {
                              write(stream, static_cast<int>(TransformType::CONST));
                              op.serialize(stream);
                          },
                          [&stream](const op::Convert& op) {
                              write(stream, static_cast<int>(TransformType::CONVERT));
                              op.serialize(stream);
                          },
                          [&stream](const op::Permute& op) {
                              write(stream, static_cast<int>(TransformType::PERMUTE));
                              op.serialize(stream);
                          },
                          [&stream](const op::Unpack& op) {
                              write(stream, static_cast<int>(TransformType::UNPACK));
                              op.serialize(stream);
                          }},
               m_transform);
}

std::shared_ptr<LazyTensorImpl> LazyTensorImpl::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    auto lt_impl = std::make_shared<LazyTensorImpl>();
    read(stream, lt_impl->m_hash);
    int op_type;
    read(stream, op_type);
    switch (TransformType(op_type)) {
    case TransformType::CONCAT:
        lt_impl->m_transform = op::Concat::deserialize(stream);
        break;
    case TransformType::CONST:
        lt_impl->m_transform = op::Const::deserialize(stream);
        break;
    case TransformType::CONVERT:
        lt_impl->m_transform = op::Convert::deserialize(stream);
        break;
    case TransformType::PERMUTE:
        lt_impl->m_transform = op::Permute::deserialize(stream);
        break;
    case TransformType::UNPACK:
        lt_impl->m_transform = op::Unpack::deserialize(stream);
        break;
    default:
        NPUW_ASSERT(false && "Unsupported type");
        break;
    }
    return lt_impl;
}

LazyTensor::LazyTensor(const std::shared_ptr<ov::op::v0::Constant>& const_ptr)
    : m_impl(std::make_shared<LazyTensorImpl>(op::Const(const_ptr))) {}
LazyTensor::LazyTensor(const std::vector<LazyTensor>& to_concat, const std::size_t axis)
    : m_impl(std::make_shared<LazyTensorImpl>(op::Concat{to_concat, axis})) {}
LazyTensor::LazyTensor(const LazyTensor& cw,
                       const LazyTensor& cz,
                       const LazyTensor& cs,
                       const ov::element::Type& type,
                       const ov::Shape& shape)
    : m_impl(std::make_shared<LazyTensorImpl>(op::Unpack{cw, cz, cs, type, shape})) {}

LazyTensor LazyTensor::permute(const std::vector<std::size_t>& axes) {
    LazyTensor new_lt;
    new_lt.m_impl = std::make_shared<LazyTensorImpl>(op::Permute{*this, axes});
    return new_lt;
}

LazyTensor LazyTensor::convert(const ov::element::Type& type) {
    LazyTensor new_lt;
    new_lt.m_impl = std::make_shared<LazyTensorImpl>(op::Convert{*this, type});
    return new_lt;
}

bool LazyTensor::operator==(const LazyTensor& other) const {
    if (!m_impl && !other.m_impl) {
        return true;
    }
    if ((!m_impl && other.m_impl) || (m_impl && !other.m_impl)) {
        return false;
    }
    return *m_impl.get() == *other.m_impl.get();
}

bool LazyTensor::operator!=(const LazyTensor& other) const {
    return !(*this == other);
}

ov::Tensor LazyTensor::eval() const {
    if (!m_impl) {
        return ov::Tensor();
    }
    return m_impl->eval();
}

void LazyTensor::read_weight(const ov::npuw::s11n::Weights& weights) {
    NPUW_ASSERT(m_impl && "Trying to read weights into uninitialized tensor!");
    m_impl->read_weight(weights);
}

LazyTensor::operator bool() const {
    return m_impl != nullptr;
}

std::size_t LazyTensor::get_hash() const {
    if (!m_impl) {
        return 0;
    }
    return m_impl->get_hash();
}

void LazyTensor::detach() {
    if (m_impl) {
        m_impl->detach();
    }
}

void LazyTensor::serialize(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    if (!m_impl) {
        write(stream, false);
        return;
    }
    write(stream, true);
    m_impl->serialize(stream);
}

LazyTensor LazyTensor::deserialize(std::istream& stream) {
    using namespace ov::npuw::s11n;
    bool is_initialized;
    read(stream, is_initialized);
    LazyTensor lt;
    if (!is_initialized) {
        return lt;
    }
    lt.m_impl = LazyTensorImpl::deserialize(stream);
    return lt;
}

std::size_t LazyTensor::Hash::operator()(const LazyTensor& lt) const {
    return lt.get_hash();
}
