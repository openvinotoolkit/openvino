// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

#include <tuple>
#include <type_traits>
#include <variant>

#include "logging.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

using ov::npuw::weights::LazyTensor;

namespace ov {
namespace npuw {
namespace weights {
namespace op {
struct Const {
    std::shared_ptr<ov::op::v0::Constant> m_node;
    ov::element::Type m_cached_type;
    ov::Shape m_cached_shape;
    const void* m_cached_ptr = nullptr;

    explicit Const(std::shared_ptr<ov::op::v0::Constant> n) : m_node(n) {
        m_cached_type = m_node->get_element_type();
        m_cached_shape = m_node->get_shape();
        m_cached_ptr = m_node->get_data_ptr();
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
        NPUW_ASSERT(m_node && "Const::eval() can only happen before detach");
        return ov::npuw::util::tensor_from_const(m_node);
    }
    void detach() {
        m_node.reset();
    }
};
struct Concat {
    std::vector<LazyTensor> tensors;
    std::size_t axis;

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
    void detach() {
        for (auto&& lt : tensors) {
            lt.detach();
        }
    }
};

struct Unpack {
    LazyTensor w, z, s;
    ov::element::Type type;
    ov::Shape shape;

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
    void detach() {
        w.detach();
        z.detach();
        s.detach();
    }
};
struct Permute {
    LazyTensor tensor;
    std::vector<std::size_t> axes;

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
    void detach() {
        tensor.detach();
    }
};
struct Convert {
    LazyTensor tensor;
    ov::element::Type type;

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
    void detach() {
        tensor.detach();
    }
};
}  // namespace op

using Transform = std::variant<op::Const, op::Concat, op::Unpack, op::Permute, op::Convert>;

struct LazyTensorImpl {
public:
    explicit LazyTensorImpl(Transform&& t);
    bool operator==(const LazyTensorImpl& other) const;

    ov::Tensor eval() const;
    std::size_t get_hash() const;

    void detach();

    Transform m_transform;
    const std::size_t m_hash = 0;
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

std::size_t LazyTensorImpl::get_hash() const {
    return m_hash;
}

void LazyTensorImpl::detach() {
    std::visit(overloaded{[](auto& op) {
                   op.detach();
               }},
               m_transform);
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

std::size_t LazyTensor::Hash::operator()(const LazyTensor& lt) const {
    return lt.get_hash();
}
