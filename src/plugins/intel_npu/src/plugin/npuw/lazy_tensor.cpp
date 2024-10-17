// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

using ov::npuw::weights::LazyTensor;

namespace ov {
namespace npuw {
namespace weights {

enum class TransformType : int { TENSOR, CONST, CONCAT, UNPACK, PERMUTE, CONVERT };
using ConcatMeta = std::pair<std::vector<LazyTensor>, std::size_t>;
using UnpackMeta = std::tuple<LazyTensor, LazyTensor, LazyTensor, ov::Shape, ov::element::Type>;
using ConstPtr = std::shared_ptr<ov::op::v0::Constant>;
using Transform = std::variant<ov::Tensor, ConstPtr, ConcatMeta, UnpackMeta, std::vector<std::size_t>, std::monostate>;

struct LazyTensorImpl {
public:
    LazyTensorImpl() = default;
    LazyTensorImpl(const ov::Tensor& tensor);
    LazyTensorImpl(const std::shared_ptr<ov::op::v0::Constant>& const_ptr);
    LazyTensorImpl(const std::vector<LazyTensor>& to_concat, const std::size_t& axis);
    LazyTensorImpl(const LazyTensor& cw,
                   const LazyTensor& cz,
                   const LazyTensor& cs,
                   const ov::Shape& shape,
                   const ov::element::Type& type);

    ov::Tensor eval() const;

    bool operator==(const LazyTensorImpl& other) const;
    std::size_t get_hash() const;

    std::shared_ptr<LazyTensorImpl> m_parent = nullptr;
    std::pair<TransformType, Transform> m_transform;
    std::size_t m_hash = 0;

    const void* m_orig_data = nullptr;
    ov::Shape m_orig_shape;
    ov::element::Type m_orig_type;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov

using ov::npuw::weights::ConcatMeta;
using ov::npuw::weights::ConstPtr;
using ov::npuw::weights::LazyTensorImpl;
using ov::npuw::weights::Transform;
using ov::npuw::weights::TransformType;
using ov::npuw::weights::UnpackMeta;

std::size_t LazyTensorImpl::get_hash() const {
    // Already calculated
    if (m_hash != 0) {
        return m_hash;
    }

    // Get parent's hash
    std::size_t seed = 0;
    if (m_parent) {
        seed = m_parent->get_hash();
    } else {
        seed = std::hash<const void*>()(m_orig_data) + 0x9e3779b9;
        for (const auto& dim : m_orig_shape) {
            seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
        }
        seed ^= m_orig_type.hash() + 0x9e3779b9;
    }

    // Combine with this hash
    seed ^= std::hash<int>()(static_cast<int>(m_transform.first)) + 0x9e3779b9;
    if (m_transform.first == TransformType::PERMUTE) {
        const auto& axes = std::get<std::vector<std::size_t>>(m_transform.second);
        for (const auto& axis : axes) {
            seed ^= std::hash<std::size_t>()(axis) + 0x9e3779b9;
        }
    } else if (m_transform.first == TransformType::CONCAT) {
        const auto& axis = std::get<ConcatMeta>(m_transform.second).second;
        seed ^= std::hash<std::size_t>()(axis) + 0x9e3779b9;
        for (auto& lt : std::get<ConcatMeta>(m_transform.second).first) {
            seed ^= lt.get_hash() + 0x9e3779b9;
        }
    } else if (m_transform.first == TransformType::UNPACK) {
        const auto& unpack_meta = std::get<UnpackMeta>(m_transform.second);
        seed ^= std::get<0>(unpack_meta).get_hash() + 0x9e3779b9;
        seed ^= std::get<1>(unpack_meta).get_hash() + 0x9e3779b9;
        seed ^= std::get<2>(unpack_meta).get_hash() + 0x9e3779b9;
        for (const auto& dim : std::get<3>(unpack_meta)) {
            seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
        }
        seed ^= std::get<4>(unpack_meta).hash() + 0x9e3779b9;
    }

    return seed;
}

LazyTensorImpl::LazyTensorImpl(const ov::Tensor& tensor) {
    if (!tensor) {
        // Dummy tensor - don't initialize
        return;
    }
    m_orig_data = tensor.data();
    m_orig_shape = tensor.get_shape();
    m_orig_type = tensor.get_element_type();
    m_transform = std::make_pair(TransformType::TENSOR, tensor);
    m_hash = get_hash();
}

LazyTensorImpl::LazyTensorImpl(const std::shared_ptr<ov::op::v0::Constant>& const_ptr) {
    m_orig_data = const_ptr->get_data_ptr();
    m_orig_shape = const_ptr->get_shape();
    m_orig_type = const_ptr->get_element_type();
    m_transform = std::make_pair(TransformType::CONST, const_ptr);
    m_hash = get_hash();
}

LazyTensorImpl::LazyTensorImpl(const std::vector<LazyTensor>& to_concat, const std::size_t& axis) {
    m_transform = std::make_pair(TransformType::CONCAT, std::make_pair(to_concat, axis));
    m_hash = get_hash();
}

LazyTensorImpl::LazyTensorImpl(const LazyTensor& cw,
                               const LazyTensor& cz,
                               const LazyTensor& cs,
                               const ov::Shape& shape,
                               const ov::element::Type& type) {
    m_transform = std::make_pair(TransformType::UNPACK, std::make_tuple(cw, cz, cs, shape, type));
    m_hash = get_hash();
}

bool LazyTensorImpl::operator==(const LazyTensorImpl& other) const {
    if (m_hash != other.m_hash || m_orig_data != other.m_orig_data || m_orig_shape != other.m_orig_shape ||
        m_orig_type != other.m_orig_type || m_transform.first != other.m_transform.first) {
        return false;
    }

    switch (m_transform.first) {
    case TransformType::TENSOR:
        // everything is already compared above - skip
        break;
    case TransformType::CONST:
        // everything is already compared above - skip
        break;
    case TransformType::CONVERT:
        // everything is already compared above - skip
        break;
    case TransformType::PERMUTE:
        if (std::get<std::vector<std::size_t>>(m_transform.second) !=
            std::get<std::vector<std::size_t>>(other.m_transform.second)) {
            return false;
        }
        break;
    case TransformType::CONCAT:
        if (std::get<ConcatMeta>(m_transform.second) != std::get<ConcatMeta>(other.m_transform.second)) {
            return false;
        }
        break;
    case TransformType::UNPACK:
        if (std::get<UnpackMeta>(m_transform.second) != std::get<UnpackMeta>(other.m_transform.second)) {
            return false;
        }
        break;
    default:
        NPUW_ASSERT(false);
        break;
    }

    if ((m_parent && !other.m_parent) || (!m_parent && other.m_parent)) {
        return false;
    }

    if (m_parent && other.m_parent) {
        return *m_parent.get() == *other.m_parent.get();
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

    // Process the initial tensor - either from Const or from Concat
    if (!m_parent) {
        if (m_transform.first == TransformType::TENSOR) {
            return std::get<ov::Tensor>(m_transform.second);
        } else if (m_transform.first == TransformType::CONST) {
            return ov::npuw::util::tensor_from_const(std::get<ConstPtr>(m_transform.second));
        } else if (m_transform.first == TransformType::CONCAT) {
            std::vector<ov::Tensor> to_concat;
            for (const auto& lt : std::get<ConcatMeta>(m_transform.second).first) {
                to_concat.push_back(lt.eval());
            }
            return ov::npuw::util::concat(to_concat, std::get<ConcatMeta>(m_transform.second).second);
        } else if (m_transform.first == TransformType::UNPACK) {
            const auto& unpack_meta = std::get<UnpackMeta>(m_transform.second);
            const auto& cw = std::get<0>(unpack_meta);
            const auto& cz = std::get<1>(unpack_meta);
            const auto& cs = std::get<2>(unpack_meta);
            const auto& shape = std::get<3>(unpack_meta);
            const auto& type = std::get<4>(unpack_meta);

            const auto& gti = ov::get_tensor_impl;
            const auto& tw = cw.eval();
            const auto& tz = cz.eval();
            const auto& ts = cs.eval();
            ov::Tensor dst(type, shape);
            if (tw && tz && ts) {
                ov::npuw::util::unpack(gti(tw), gti(tz), gti(ts), gti(dst));
            } else if (tw && ts) {
                ov::npuw::util::unpack(gti(tw), gti(ts), gti(dst));
            } else {
                NPUW_ASSERT(false && "Unsupported combination");
            }
            return dst;
        } else {
            NPUW_ASSERT(false);
        }
    }

    // Process transformation
    switch (m_transform.first) {
    case TransformType::PERMUTE:
        return ov::npuw::util::permute(m_parent->eval(), std::get<std::vector<std::size_t>>(m_transform.second));
    case TransformType::CONVERT:
        return ov::npuw::util::to_f16(m_parent->eval());
    default:
        NPUW_ASSERT(false);
    }

    NPUW_ASSERT(false);
    return ov::Tensor();
}

LazyTensor::LazyTensor(const ov::Tensor& tensor) : m_impl(std::make_shared<LazyTensorImpl>(tensor)) {}
LazyTensor::LazyTensor(const std::shared_ptr<ov::op::v0::Constant>& const_ptr)
    : m_impl(std::make_shared<LazyTensorImpl>(const_ptr)) {}
LazyTensor::LazyTensor(const std::vector<LazyTensor>& to_concat, const std::size_t axis)
    : m_impl(std::make_shared<LazyTensorImpl>(to_concat, axis)) {}
LazyTensor::LazyTensor(const LazyTensor& cw,
                       const LazyTensor& cz,
                       const LazyTensor& cs,
                       const ov::Shape& shape,
                       const ov::element::Type& type)
    : m_impl(std::make_shared<LazyTensorImpl>(cw, cz, cs, shape, type)) {}

LazyTensor LazyTensor::permute(const std::vector<std::size_t>& axes) {
    const auto& curr = m_impl;
    auto new_lt = std::make_shared<LazyTensorImpl>();

    new_lt->m_orig_data = curr->m_orig_data;
    new_lt->m_orig_shape = curr->m_orig_shape;
    new_lt->m_orig_type = curr->m_orig_type;

    new_lt->m_transform = std::make_pair(TransformType::PERMUTE, axes);
    new_lt->m_parent = curr;
    new_lt->m_hash = new_lt->get_hash();

    m_impl = new_lt;
    return *this;
}

LazyTensor LazyTensor::convert() {
    const auto& curr = m_impl;
    auto new_lt = std::make_shared<LazyTensorImpl>();

    new_lt->m_orig_data = curr->m_orig_data;
    new_lt->m_orig_shape = curr->m_orig_shape;
    new_lt->m_orig_type = curr->m_orig_type;

    new_lt->m_transform = std::make_pair(TransformType::CONVERT, std::monostate{});
    new_lt->m_parent = curr;
    new_lt->m_hash = new_lt->get_hash();

    m_impl = new_lt;
    return *this;
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

std::size_t LazyTensor::Hash::operator()(const LazyTensor& lt) const {
    return lt.get_hash();
}
