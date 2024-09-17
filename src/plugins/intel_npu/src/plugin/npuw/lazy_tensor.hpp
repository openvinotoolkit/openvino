// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "logging.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace weights {

enum class TransformType : int {
    TENSOR,
    PERMUTE,
    CONVERT,
    CONCAT  // TODO: support
};

using Transform = std::variant<ov::Tensor, std::vector<std::size_t>, std::monostate>;

class LazyTensor {
public:
    class Hash {
    public:
        std::size_t operator()(const LazyTensor& lt) const;
    };

    explicit LazyTensor() = default;
    explicit LazyTensor(const TransformType& type, const Transform& transform);

    bool operator==(const LazyTensor& other) const;

    void update(const TransformType& type, const Transform& transform);
    ov::Tensor eval() const;
    void* get_orig_data() const;

private:
    ov::Tensor get_tensor() const;

    std::vector<std::pair<TransformType, Transform>> m_transforms;
    void* m_orig_data;
    ov::Shape m_orig_shape;
    ov::element::Type m_orig_type;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
