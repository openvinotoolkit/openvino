// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "util.hpp"
#include "logging.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace weights {

enum class TransformType {
    TENSOR,
    PERMUTE,
    CONVERT,
    CONCAT,  // TODO: support
    // FIXME: workaround to prevent the second model from pushing to transformations history
    END // Once the first model finishes Tensor processing, need to notify the second model it's ready to be used
};

using Transform = std::variant<ov::Tensor, std::vector<std::size_t>, std::monostate>;

class LazyTensor {
public:
    class Hash {
    public:
        std::size_t operator()(const LazyTensor& lt) const {
            // FIXME: implement
            return 0;
        }
    };

    explicit LazyTensor(const TransformType& type, const Transform& transform) {
        // Sanity check
        NPUW_ASSERT(type == TransformType::TENSOR && std::holds_alternative<ov::Tensor>(transform));
        m_transforms.push_back({type, transform});
        const auto& tensor = std::get<ov::Tensor>(transform);
        m_orig_data = tensor.data();
        m_orig_shape = tensor.get_shape();
        m_orig_type = tensor.get_element_type();
    };

    bool operator==(const LazyTensor& other) const {
        if (m_orig_data != other.m_orig_data || m_orig_shape != other.m_orig_shape || m_orig_type != other.m_orig_type || m_transforms.size() != other.m_transforms.size()) {
            return false;
        }

        for (size_t i = 0; i < m_transforms.size(); ++i) {
            if (m_transforms[i].first != other.m_transforms[i].first) {
                return false;
            }
            if (m_transforms[i].first != TransformType::TENSOR) { // Tensor can be already transformed, compare only the original meta above
                if (m_transforms[i].second != other.m_transforms[i].second) {
                    return false;
                }
            }
        }

        return false;
    }

    void update(const TransformType& type, const Transform& transform) {
        // Sanity check
        NPUW_ASSERT((type == TransformType::PERMUTE && std::holds_alternative<std::vector<std::size_t>>(transform)) ||
                    (type == TransformType::CONVERT && std::holds_alternative<std::monostate>(transform)) ||
                    (type == TransformType::END && std::holds_alternative<std::monostate>(transform)));
        m_transforms.push_back({type, transform});
    }

    ov::Tensor transform() {
        for (auto& tr: m_transforms) {
            switch (tr.first) {
                case TransformType::TENSOR:
                    continue;
                case TransformType::PERMUTE:
                    ov::npuw::util::permute(get_tensor(), std::get<std::vector<std::size_t>>(tr.second));
                case TransformType::CONVERT:
                    ov::npuw::util::to_f16(get_tensor());
                case TransformType::END:
                    return get_tensor();
                default:
                    NPUW_ASSERT(false);
            }
        }
        // Should be unreachable
        NPUW_ASSERT(false);
        return {};
    }

    ov::Tensor& get_tensor() {
        return std::get<ov::Tensor>(m_transforms.front().second);
    }

    void* get_orig_data() const {
        return m_orig_data;
    }

private:
    std::vector<std::pair<TransformType, Transform>> m_transforms;
    void* m_orig_data;
    ov::Shape m_orig_shape;
    ov::element::Type m_orig_type;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
