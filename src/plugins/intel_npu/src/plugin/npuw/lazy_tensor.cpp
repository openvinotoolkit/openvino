// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

using ov::npuw::weights::LazyTensor;
using ov::npuw::weights::Transform;
using ov::npuw::weights::TransformType;

std::size_t LazyTensor::Hash::operator()(const LazyTensor& lt) const {
    std::size_t seed = std::hash<void*>()(lt.m_orig_data) + 0x9e3779b9;
    seed ^= std::hash<std::string>()(lt.m_orig_shape.to_string()) + 0x9e3779b9;
    seed ^= std::hash<std::string>()(lt.m_orig_type.to_string()) + 0x9e3779b9;
    for (const auto& tr : lt.m_transforms) {
        seed ^= std::hash<int>()(static_cast<int>(tr.first)) + 0x9e3779b9;
        if (tr.first == TransformType::PERMUTE) {
            const auto& axes = std::get<std::vector<std::size_t>>(tr.second);
            for (const auto& axis : axes) {
                seed ^= std::hash<std::size_t>()(axis) + 0x9e3779b9;
            }
        }
    }
    return seed;
}

LazyTensor::LazyTensor(const TransformType& type, const Transform& transform) {
    // Sanity check
    NPUW_ASSERT(type == TransformType::TENSOR && std::holds_alternative<ov::Tensor>(transform));
    m_transforms.push_back({type, transform});
    const auto& tensor = std::get<ov::Tensor>(transform);
    m_orig_data = tensor.data();
    m_orig_shape = tensor.get_shape();
    m_orig_type = tensor.get_element_type();
};

bool LazyTensor::operator==(const LazyTensor& other) const {
    if (m_orig_data != other.m_orig_data || m_orig_shape != other.m_orig_shape || m_orig_type != other.m_orig_type ||
        m_transforms.size() != other.m_transforms.size()) {
        return false;
    }

    for (size_t i = 0; i < m_transforms.size(); ++i) {
        if (m_transforms[i].first != other.m_transforms[i].first) {
            return false;
        }
        // Only PERMUTE has meta which needs to be compared
        if (m_transforms[i].first == TransformType::PERMUTE) {
            if (std::get<std::vector<std::size_t>>(m_transforms[i].second) !=
                std::get<std::vector<std::size_t>>(other.m_transforms[i].second)) {
                return false;
            }
        }
    }

    return true;
}

void LazyTensor::update(const TransformType& type, const Transform& transform) {
    // Sanity check
    NPUW_ASSERT((type == TransformType::PERMUTE && std::holds_alternative<std::vector<std::size_t>>(transform)) ||
                (type == TransformType::CONVERT && std::holds_alternative<std::monostate>(transform)));
    m_transforms.push_back({type, transform});
}

ov::Tensor LazyTensor::eval() const {
    // Sanity check
    NPUW_ASSERT(std::holds_alternative<ov::Tensor>(m_transforms.front().second));

    ov::Tensor transformed = get_tensor();
    ov::Tensor tnew;
    for (auto& tr : m_transforms) {
        switch (tr.first) {
        case TransformType::TENSOR:
            continue;
        case TransformType::PERMUTE:
            tnew = ov::npuw::util::permute(transformed, std::get<std::vector<std::size_t>>(tr.second));
            tnew.copy_to(transformed);
        case TransformType::CONVERT:
            tnew = ov::npuw::util::to_f16(transformed);
            tnew.copy_to(transformed);
        default:
            NPUW_ASSERT(false);
        }
    }

    return transformed;
}

void* LazyTensor::get_orig_data() const {
    return m_orig_data;
}

ov::Tensor LazyTensor::get_tensor() const {
    // Sanity check
    NPUW_ASSERT(std::holds_alternative<ov::Tensor>(m_transforms.front().second));
    return std::get<ov::Tensor>(m_transforms.front().second);
}
