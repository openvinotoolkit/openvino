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
struct Tensor {
    ov::Tensor tensor;
};
struct Const {
    std::shared_ptr<ov::op::v0::Constant> node;
};
struct Concat {
    std::vector<LazyTensor> tensors;
    std::size_t axis;
};
struct Unpack {
    LazyTensor w, z, s;
    ov::element::Type type;
    ov::Shape shape;
};
struct Permute {
    LazyTensor tensor;
    std::vector<std::size_t> axes;
};
struct Convert {
    LazyTensor tensor;
    ov::element::Type type;
};
}  // namespace op

using Transform = std::variant<op::Tensor, op::Const, op::Concat, op::Unpack, op::Permute, op::Convert>;

struct LazyTensorImpl {
public:
    LazyTensorImpl() = default;
    explicit LazyTensorImpl(Transform&& t);

    ov::Tensor eval() const;

    bool operator==(const LazyTensorImpl& other) const;
    std::size_t get_hash() const;
    const void* get_data() const;
    const ov::Shape& get_shape() const;
    const ov::element::Type& get_type() const;

    Transform m_transform;
    std::size_t m_hash = 0;

    const void* m_orig_data = nullptr;
    ov::Shape m_orig_shape;
    ov::element::Type m_orig_type;
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

std::size_t LazyTensorImpl::get_hash() const {
    // Already calculated
    if (m_hash != 0) {
        return m_hash;
    }

    // Get initial hash
    std::size_t seed = std::hash<const void*>()(m_orig_data) + 0x9e3779b9;
    for (const auto& dim : m_orig_shape) {
        seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
    }
    seed ^= m_orig_type.hash() + 0x9e3779b9;

    // Combine with this hash
    std::visit(overloaded{[](const auto& op) { /* do nothing */ },
                          [&seed](const op::Permute& op) {
                              for (const auto& axis : op.axes) {
                                  seed ^= std::hash<std::size_t>()(axis) + 0x9e3779b9;
                              }
                              seed ^= op.tensor.get_hash() + 0x9e3779b9;
                          },
                          [&seed](const op::Convert& op) {
                              seed ^= op.type.hash() + 0x9e3779b9;
                              seed ^= op.tensor.get_hash() + 0x9e3779b9;
                          },
                          [&seed](const op::Concat& op) {
                              seed ^= std::hash<std::size_t>()(op.axis) + 0x9e3779b9;
                              for (auto& lt : op.tensors) {
                                  seed ^= lt.get_hash() + 0x9e3779b9;
                              }
                          },
                          [&seed](const op::Unpack& op) {
                              seed ^= op.w.get_hash() + 0x9e3779b9;
                              seed ^= op.z.get_hash() + 0x9e3779b9;
                              seed ^= op.s.get_hash() + 0x9e3779b9;
                              seed ^= op.type.hash() + 0x9e3779b9;
                              for (const auto& dim : op.shape) {
                                  seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
                              }
                          }},
               m_transform);

    return seed;
}

const void* LazyTensorImpl::get_data() const {
    return m_orig_data;
}

const ov::Shape& LazyTensorImpl::get_shape() const {
    return m_orig_shape;
}

const ov::element::Type& LazyTensorImpl::get_type() const {
    return m_orig_type;
}

LazyTensorImpl::LazyTensorImpl(Transform&& t) {
    std::visit(overloaded{[](const auto& op) {
                              NPUW_ASSERT(false && "Unsupported LazyTensorImpl constructor!");
                          },
                          [this](const op::Tensor& op) {
                              const auto& tensor = op.tensor;
                              if (!tensor) {
                                  // Dummy tensor - don't initialize
                                  return;
                              }
                              m_orig_data = tensor.data();
                              m_orig_shape = tensor.get_shape();
                              m_orig_type = tensor.get_element_type();
                              m_transform = op::Tensor{tensor};
                          },
                          [this](const op::Const& op) {
                              const auto& const_ptr = op.node;
                              m_orig_data = const_ptr->get_data_ptr();
                              m_orig_shape = const_ptr->get_shape();
                              m_orig_type = const_ptr->get_element_type();
                              m_transform = op::Const{const_ptr};
                          },
                          [this](const op::Concat& op) {
                              m_transform = op;
                          },
                          [this](const op::Unpack& op) {
                              m_transform = op;
                          },
                          [this](const op::Permute& op) {
                              m_transform = op;
                              m_orig_data = op.tensor.get_data();
                              m_orig_shape = op.tensor.get_shape();
                              m_orig_type = op.tensor.get_type();
                          },
                          [this](const op::Convert& op) {
                              m_transform = op;
                              m_orig_data = op.tensor.get_data();
                              m_orig_shape = op.tensor.get_shape();
                              m_orig_type = op.tensor.get_type();
                          }},
               t);

    m_hash = get_hash();
}

bool LazyTensorImpl::operator==(const LazyTensorImpl& other) const {
    if (m_hash != other.m_hash || m_orig_data != other.m_orig_data || m_orig_shape != other.m_orig_shape ||
        m_orig_type != other.m_orig_type) {
        return false;
    }

    bool meta_diff = false;
    std::visit(overloaded{[](const auto& op) { /* do nothing */ },
                          [&](const op::Permute& op) {
                              meta_diff = (op.axes != std::get<op::Permute>(other.m_transform).axes ||
                                           op.tensor != std::get<op::Permute>(other.m_transform).tensor);
                          },
                          [&](const op::Convert& op) {
                              meta_diff = (op.type != std::get<op::Convert>(other.m_transform).type ||
                                           op.tensor != std::get<op::Convert>(other.m_transform).tensor);
                          },
                          [&](const op::Concat& op) {
                              meta_diff = (op.axis != std::get<op::Concat>(other.m_transform).axis ||
                                           op.tensors != std::get<op::Concat>(other.m_transform).tensors);
                          },
                          [&](const op::Unpack& op) {
                              meta_diff = (op.type != std::get<op::Unpack>(other.m_transform).type ||
                                           op.shape != std::get<op::Unpack>(other.m_transform).shape ||
                                           op.w != std::get<op::Unpack>(other.m_transform).w ||
                                           op.z != std::get<op::Unpack>(other.m_transform).z ||
                                           op.s != std::get<op::Unpack>(other.m_transform).s);
                          }},
               m_transform);

    if (meta_diff) {
        return false;
    }

    return true;
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

    ov::Tensor result;
    std::visit(overloaded{[](const auto& op) {
                              NPUW_ASSERT(false);
                          },
                          [&](const op::Tensor& op) {
                              result = op.tensor;
                          },
                          [&](const op::Const& op) {
                              result = ov::npuw::util::tensor_from_const(op.node);
                          },
                          [&](const op::Concat& op) {
                              std::vector<ov::Tensor> to_concat;
                              for (const auto& lt : op.tensors) {
                                  to_concat.push_back(lt.eval());
                              }
                              result = std::move(ov::npuw::util::concat(to_concat, op.axis));
                          },
                          [&](const op::Unpack& op) {
                              const auto& gti = ov::get_tensor_impl;
                              const auto& tw = op.w.eval();
                              const auto& tz = op.z.eval();
                              const auto& ts = op.s.eval();
                              NPUW_ASSERT(tw);
                              ov::Tensor dst(op.type, op.shape);
                              if (tw && tz && ts) {
                                  ov::npuw::util::unpack(gti(tw), gti(tz), gti(ts), gti(dst));
                              } else if (tw && ts) {
                                  ov::npuw::util::unpack(gti(tw), gti(ts), gti(dst));
                              } else {
                                  NPUW_ASSERT(false && "Unsupported combination");
                              }
                              result = std::move(dst);
                          },
                          [&](const op::Permute& op) {
                              result = std::move(ov::npuw::util::permute(op.tensor.eval(), op.axes));
                          },
                          [&](const op::Convert& op) {
                              NPUW_ASSERT(ov::element::f16 == op.type);
                              result = std::move(ov::npuw::util::to_f16(op.tensor.eval()));
                          }},
               m_transform);

    NPUW_ASSERT(result);
    return result;
}

LazyTensor::LazyTensor(const ov::Tensor& tensor) : m_impl(std::make_shared<LazyTensorImpl>(op::Tensor{tensor})) {}
LazyTensor::LazyTensor(const std::shared_ptr<ov::op::v0::Constant>& const_ptr)
    : m_impl(std::make_shared<LazyTensorImpl>(op::Const{const_ptr})) {}
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
    return *m_impl.get() == *other.m_impl.get();
}

bool LazyTensor::operator!=(const LazyTensor& other) const {
    return !(*m_impl.get() == *other.m_impl.get());
}

ov::Tensor LazyTensor::eval() const {
    return m_impl->eval();
}

std::size_t LazyTensor::get_hash() const {
    return m_impl->get_hash();
}

const void* LazyTensor::get_data() const {
    return m_impl->get_data();
}

const ov::Shape& LazyTensor::get_shape() const {
    return m_impl->get_shape();
}

const ov::element::Type& LazyTensor::get_type() const {
    return m_impl->get_type();
}

std::size_t LazyTensor::Hash::operator()(const LazyTensor& lt) const {
    return lt.get_hash();
}
