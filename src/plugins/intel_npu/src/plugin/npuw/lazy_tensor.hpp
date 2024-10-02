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

enum class TransformType : int { TENSOR, PERMUTE, CONVERT, CONCAT };

// Forward declaration
class LazyTensor;

using ConcatMeta = std::pair<std::vector<LazyTensor>, std::size_t>;
using ConstPtr = std::shared_ptr<ov::op::v0::Constant>;
using LTData = std::variant<ConstPtr, ov::Tensor>;

// LazyTensor owns Constant's memory
using Transform = std::variant<LTData, std::vector<std::size_t>, std::monostate, ConcatMeta>;

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

    ov::Tensor get_orig_tensor() const;

    bool has_transformations() const;

private:
    std::vector<std::pair<TransformType, Transform>> m_transforms;
    void* m_orig_data = nullptr;
    ov::Shape m_orig_shape;
    ov::element::Type m_orig_type;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
