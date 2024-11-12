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

enum class TransformType : int { THIS, PERMUTE, CONVERT, CONCAT, UNPACK };

// Forward declaration
class LazyTensor;
struct LazyTensorImpl;

using ConcatMeta = std::pair<std::vector<LazyTensor>, std::size_t>;
using UnpackMeta = std::tuple<LazyTensor, LazyTensor, LazyTensor, ov::Shape, ov::element::Type>;
using ConstPtr = std::shared_ptr<ov::op::v0::Constant>;
using OrigData = std::variant<ConstPtr, ov::Tensor>;

using Transform = std::variant<OrigData, std::vector<std::size_t>, std::monostate, ConcatMeta, UnpackMeta>;

class LazyTensor {
public:
    class Hash {
    public:
        std::size_t operator()(const LazyTensor& lt) const;
    };

    LazyTensor() = default;
    LazyTensor(const TransformType& type, const Transform& transform);

    bool operator==(const LazyTensor& other) const;
    bool operator!=(const LazyTensor& other) const;

    void update(const TransformType& type, const Transform& transform);
    ov::Tensor eval() const;

    ov::Tensor get_orig_tensor() const;
    std::size_t get_hash() const;
    bool has_transformations() const;

private:
    std::shared_ptr<LazyTensorImpl> m_impl = nullptr;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
