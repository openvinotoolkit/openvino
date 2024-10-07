// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

using ov::npuw::weights::ConcatMeta;
using ov::npuw::weights::ConstPtr;
using ov::npuw::weights::LazyTensor;
using ov::npuw::weights::OrigData;
using ov::npuw::weights::Transform;
using ov::npuw::weights::TransformType;
using ov::npuw::weights::UnpackMeta;

namespace ov {
namespace npuw {
namespace weights {

struct LazyTensorImpl {
public:
    LazyTensorImpl() = default;
    LazyTensorImpl(const TransformType& type, const Transform& transform);

    bool operator==(const LazyTensorImpl& other) const;

    ov::Tensor eval() const;

    ov::Tensor get_orig_tensor() const;

    std::size_t get_hash() const;

    bool has_transformations() const;

    std::shared_ptr<LazyTensorImpl> m_parent = nullptr;
    std::pair<TransformType, Transform> m_transform;
    std::size_t m_hash = 0;

    void* m_orig_data = nullptr;
    ov::Shape m_orig_shape;
    ov::element::Type m_orig_type;
};

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
        seed = std::hash<void*>()(m_orig_data) + 0x9e3779b9;
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
        const auto& dst = std::get<3>(unpack_meta);
        for (const auto& dim : dst.get_shape()) {
            seed ^= std::hash<std::size_t>()(dim) + 0x9e3779b9;
        }
        seed ^= dst.get_element_type().hash() + 0x9e3779b9;
    }

    return seed;
}
}  // namespace weights
}  // namespace npuw
}  // namespace ov

using ov::npuw::weights::LazyTensorImpl;

LazyTensorImpl::LazyTensorImpl(const TransformType& type, const Transform& transform) {
    if (type == TransformType::THIS && std::holds_alternative<OrigData>(transform)) {
        m_transform = std::make_pair(type, transform);
        ov::Tensor tensor;
        if (std::holds_alternative<ConstPtr>(std::get<OrigData>(transform))) {
            tensor = ov::npuw::util::tensor_from_const(std::get<ConstPtr>(std::get<OrigData>(transform)));
        } else {
            tensor = std::get<ov::Tensor>(std::get<OrigData>(transform));
            if (!tensor) {
                // Don't set anything
                return;
            }
        }
        m_orig_data = tensor.data();
        m_orig_shape = tensor.get_shape();
        m_orig_type = tensor.get_element_type();
    } else if (type == TransformType::CONCAT && std::holds_alternative<ConcatMeta>(transform)) {
        m_transform = std::make_pair(type, transform);
    } else if (type == TransformType::UNPACK && std::holds_alternative<UnpackMeta>(transform)) {
        m_transform = std::make_pair(type, transform);
    } else {
        NPUW_ASSERT(false);
    }

    m_hash = get_hash();
}

bool LazyTensorImpl::operator==(const LazyTensorImpl& other) const {
    if (m_hash != other.m_hash || m_orig_data != other.m_orig_data || m_orig_shape != other.m_orig_shape ||
        m_orig_type != other.m_orig_type || m_transform.first != other.m_transform.first) {
        return false;
    }

    ConcatMeta c1, c2;
    UnpackMeta u1, u2;
    ov::Tensor unpack_dst1, unpack_dst2;

    switch (m_transform.first) {
    case TransformType::THIS:
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
        c1 = std::get<ConcatMeta>(m_transform.second);
        c2 = std::get<ConcatMeta>(other.m_transform.second);
        if (c1.second != c2.second) {
            return false;
        }
        if (c1.first.size() != c2.first.size()) {
            return false;
        }
        for (std::size_t mi = 0; mi < c1.first.size(); ++mi) {
            if (c1.first[mi] != c2.first[mi]) {
                return false;
            }
        }
        break;
    case TransformType::UNPACK:
        u1 = std::get<UnpackMeta>(m_transform.second);
        u2 = std::get<UnpackMeta>(other.m_transform.second);
        if (std::get<0>(u1) != std::get<0>(u2) || std::get<1>(u1) != std::get<1>(u2) ||
            std::get<2>(u1) != std::get<2>(u2)) {
            return false;
        }
        unpack_dst1 = std::get<3>(u1);
        unpack_dst2 = std::get<3>(u2);
        if (unpack_dst1.get_shape() != unpack_dst2.get_shape() ||
            unpack_dst1.get_element_type() != unpack_dst2.get_element_type()) {
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
        if (m_transform.first == TransformType::THIS) {
            return get_orig_tensor();
        } else if (m_transform.first == TransformType::CONCAT) {
            std::vector<ov::Tensor> to_concat;
            for (const auto& lt : std::get<ConcatMeta>(m_transform.second).first) {
                // Sanity check
                NPUW_ASSERT(!lt.has_transformations());
                to_concat.push_back(lt.get_orig_tensor());
            }
            return ov::npuw::util::concat(to_concat, std::get<ConcatMeta>(m_transform.second).second);
        } else if (m_transform.first == TransformType::UNPACK) {
            const auto& unpack_meta = std::get<UnpackMeta>(m_transform.second);
            const auto& cw = std::get<0>(unpack_meta);
            const auto& cz = std::get<1>(unpack_meta);
            const auto& cs = std::get<2>(unpack_meta);
            auto& dst = std::get<3>(unpack_meta);

            // Note: unpacking done in-place since the original tensor is empty at this point
            NPUW_ASSERT(!dst.data());
            NPUW_ASSERT(!cw.has_transformations());
            NPUW_ASSERT(!cs.has_transformations());
            // FIXME: Ugly check concat case as well since cz might be not set
            if (cz.has_transformations()) {
                NPUW_ASSERT(false);
            }

            const auto& gti = ov::get_tensor_impl;
            const auto& tw = cw.get_orig_tensor();
            const auto& tz = cz.get_orig_tensor();
            const auto& ts = cs.get_orig_tensor();
            if (tw && tz && ts) {
                ov::npuw::util::unpack(gti(tw), gti(tz), gti(ts), gti(dst));
                return dst;
            } else if (cw.get_orig_tensor() && cs.get_orig_tensor()) {
                ov::npuw::util::unpack(gti(tw), gti(ts), gti(dst));
                return dst;
            } else {
                NPUW_ASSERT(false && "Unsupported combination");
            }
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

ov::Tensor LazyTensorImpl::get_orig_tensor() const {
    // Sanity check
    NPUW_ASSERT(!has_transformations());
    if (std::holds_alternative<ConstPtr>(std::get<OrigData>(m_transform.second))) {
        return ov::npuw::util::tensor_from_const(std::get<ConstPtr>(std::get<OrigData>(m_transform.second)));
    }
    return std::get<ov::Tensor>(std::get<OrigData>(m_transform.second));
}

bool LazyTensorImpl::has_transformations() const {
    return m_transform.first != TransformType::THIS;
}

LazyTensor::LazyTensor(const TransformType& type, const Transform& transform)
    : m_impl(std::make_shared<LazyTensorImpl>(type, transform)) {}

bool LazyTensor::operator==(const LazyTensor& other) const {
    return *m_impl.get() == *other.m_impl.get();
}

bool LazyTensor::operator!=(const LazyTensor& other) const {
    return !(*m_impl.get() == *other.m_impl.get());
}

void LazyTensor::update(const TransformType& type, const Transform& transform) {
    const auto& curr = m_impl;
    auto new_lt = std::make_shared<LazyTensorImpl>();

    new_lt->m_orig_data = curr->m_orig_data;
    new_lt->m_orig_shape = curr->m_orig_shape;
    new_lt->m_orig_type = curr->m_orig_type;

    new_lt->m_transform = std::make_pair(type, transform);
    new_lt->m_parent = curr;
    new_lt->m_hash = new_lt->get_hash();

    m_impl = new_lt;
}

ov::Tensor LazyTensor::eval() const {
    return m_impl->eval();
}

ov::Tensor LazyTensor::get_orig_tensor() const {
    return m_impl->get_orig_tensor();
}

std::size_t LazyTensor::get_hash() const {
    return m_impl->get_hash();
}

std::size_t LazyTensor::Hash::operator()(const LazyTensor& lt) const {
    return lt.get_hash();
}

bool LazyTensor::has_transformations() const {
    return m_impl->has_transformations();
}
