// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

using ov::npuw::weights::ConcatMeta;
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
        } else if (tr.first == TransformType::CONCAT) {
            // concat tag can be different, no need to hash it
            const auto& axis = std::get<ConcatMeta>(tr.second).second;
            seed ^= std::hash<std::size_t>()(axis) + 0x9e3779b9;
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
        // Only PERMUTE and CONCAT have meta which needs to be compared
        if (m_transforms[i].first == TransformType::PERMUTE) {
            if (std::get<std::vector<std::size_t>>(m_transforms[i].second) !=
                std::get<std::vector<std::size_t>>(other.m_transforms[i].second)) {
                return false;
            }
        } else if (m_transforms[i].first == TransformType::CONCAT) {
            // concat tag can be different, no need to compare it
            if (std::get<ConcatMeta>(m_transforms[i].second).second !=
                std::get<ConcatMeta>(other.m_transforms[i].second).second) {
                return false;
            }
        }
    }

    return true;
}

void LazyTensor::update(const TransformType& type, const Transform& transform) {
    // Sanity check
    NPUW_ASSERT((type == TransformType::PERMUTE && std::holds_alternative<std::vector<std::size_t>>(transform)) ||
                (type == TransformType::CONVERT && std::holds_alternative<std::monostate>(transform)) ||
                (type == TransformType::CONCAT && std::holds_alternative<ConcatMeta>(transform)));
    m_transforms.push_back({type, transform});
}

ov::Tensor LazyTensor::eval() const {
    /* FIXME:
    Consider case:
        model1: concat->permute->f16
        model2: permute->f16
    Due to different history of transformation new tensors will be allocated for model2.
    However, we could avoid it by introducing a proper slicing on top of known axes and
    some kind of indicator that the only difference is concat and we should look for an existing ov::Tensor.
    Perhaps it should be done after model compilation and not handled here.
    */

    // Sanity check
    NPUW_ASSERT(std::holds_alternative<ov::Tensor>(m_transforms.front().second));

    ov::Tensor transformed = get_orig_tensor();
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
        case TransformType::CONCAT:
            tnew = ov::npuw::util::concat(get_to_concat(), std::get<ConcatMeta>(tr.second).second);
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

ov::Tensor LazyTensor::get_orig_tensor() const {
    // Sanity check
    NPUW_ASSERT(std::holds_alternative<ov::Tensor>(m_transforms.front().second));
    return std::get<ov::Tensor>(m_transforms.front().second);
}

bool LazyTensor::has_concat() const {
    for (auto& tr : m_transforms) {
        if (tr.first == TransformType::CONCAT) {
            return true;
        }
    }
    return false;
}

std::vector<ov::Tensor> LazyTensor::get_to_concat() const {
    NPUW_ASSERT(has_concat());
    std::vector<ov::Tensor> to_concat;
    for (auto& tr : m_transforms) {
        if (tr.first == TransformType::CONCAT) {
            for (const auto& lt : std::get<ConcatMeta>(tr.second).first) {
                to_concat.push_back(lt.get_orig_tensor());
            }
        }
    }
    return to_concat;
}

std::vector<LazyTensor> LazyTensor::get_lt_to_concat() const {
    NPUW_ASSERT(has_concat());
    for (auto& tr : m_transforms) {
        if (tr.first == TransformType::CONCAT) {
            return std::get<ConcatMeta>(tr.second).first;
        }
    }
    NPUW_ASSERT(false);
    return {};
}
