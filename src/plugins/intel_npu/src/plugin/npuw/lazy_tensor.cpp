// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

using ov::npuw::weights::ConcatMeta;
using ov::npuw::weights::ConstPtr;
using ov::npuw::weights::LTData;
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
            const auto& axis = std::get<ConcatMeta>(tr.second).second;
            seed ^= std::hash<std::size_t>()(axis) + 0x9e3779b9;
            for (const auto& lt : std::get<ConcatMeta>(tr.second).first) {
                seed ^= LazyTensor::Hash::operator()(lt) + 0x9e3779b9;
            }
        }
    }
    return seed;
}

LazyTensor::LazyTensor(const TransformType& type, const Transform& transform) {
    if (type == TransformType::TENSOR && std::holds_alternative<LTData>(transform)) {
        m_transforms.push_back({type, transform});
        ov::Tensor tensor;
        if (std::holds_alternative<ConstPtr>(std::get<LTData>(transform))){
            tensor = ov::npuw::util::tensor_from_const(std::get<ConstPtr>(std::get<LTData>(transform)));
        } else {
            tensor = std::get<ov::Tensor>(std::get<LTData>(transform));
        }
        m_orig_data = tensor.data();
        m_orig_shape = tensor.get_shape();
        m_orig_type = tensor.get_element_type();
    } else if (type == TransformType::CONCAT && std::holds_alternative<ConcatMeta>(transform)) {
        m_transforms.push_back({type, transform});
    } else {
        NPUW_ASSERT(false);
    }
}

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
            const auto& m1 = std::get<ConcatMeta>(m_transforms[i].second);
            const auto& m2 = std::get<ConcatMeta>(other.m_transforms[i].second);
            if (m1.second != m2.second) {
                return false;
            }
            if (m1.first.size() != m2.first.size()) {
                return false;
            }
            for (std::size_t mi = 0; mi < m1.first.size(); ++mi) {
                if (!(m1.first[mi] == m2.first[mi])) {
                    return false;
                }
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
    /* FIXME:
    Consider case:
        model1: concat->permute->f16
        model2: permute->f16
    Due to different history of transformation new tensors will be allocated for model2.
    However, we could avoid it by introducing a proper slicing on top of known axes and
    some kind of indicator that the only difference is concat and we should look for an existing ov::Tensor.
    Perhaps it should be done after model compilation and not handled here.
    */

    ov::Tensor transformed;
    ov::Tensor tnew;

    NPUW_ASSERT(!m_transforms.empty());

    // Process the initial tensor - either from Const or from Concat
    if (m_transforms.front().first == TransformType::TENSOR) {
        transformed = get_orig_tensor();
    } else if (m_transforms.front().first == TransformType::CONCAT) {
        std::vector<ov::Tensor> to_concat;
        for (const auto& lt : std::get<ConcatMeta>(m_transforms.front().second).first) {
            // Sanity check
            NPUW_ASSERT(!lt.has_transformations());
            to_concat.push_back(lt.get_orig_tensor());
        }
        transformed = ov::npuw::util::concat(to_concat, std::get<ConcatMeta>(m_transforms.front().second).second);
    } else {
        NPUW_ASSERT(false);
    }

    // Process transformation on top of initial tensor
    for (std::size_t i = 1; i < m_transforms.size(); ++i) {
        const auto& tr = m_transforms[i];
        switch (tr.first) {
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

ov::Tensor LazyTensor::get_orig_tensor() const {
    // Sanity check
    NPUW_ASSERT(!has_transformations());
    if (std::holds_alternative<ConstPtr>(std::get<LTData>(m_transforms.front().second))){
        return ov::npuw::util::tensor_from_const(std::get<ConstPtr>(std::get<LTData>(m_transforms.front().second)));
    }
    return std::get<ov::Tensor>(std::get<LTData>(m_transforms.front().second));
}

bool LazyTensor::has_transformations() const {
    // The first transformation is always initial Tensor or Concat
    if (m_transforms.size() == 1 && m_transforms.front().first == TransformType::TENSOR) {
        return false;
    }
    return true;
}
