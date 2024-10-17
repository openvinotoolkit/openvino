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
// Forward declaration
class LazyTensor;
struct LazyTensorImpl;

class LazyTensor {
public:
    class Hash {
    public:
        std::size_t operator()(const LazyTensor& lt) const;
    };

    LazyTensor() = default;
    LazyTensor(const ov::Tensor& tensor);
    LazyTensor(const std::shared_ptr<ov::op::v0::Constant>& const_ptr);
    LazyTensor(const std::vector<LazyTensor>& to_concat, const std::size_t axis);  // construct from concat
    LazyTensor(const LazyTensor& cw,
               const LazyTensor& cz,
               const LazyTensor& cs,
               const ov::Shape& shape,
               const ov::element::Type& type);  // construct from unpack

    LazyTensor permute(const std::vector<std::size_t>& axes);
    LazyTensor convert();

    bool operator==(const LazyTensor& other) const;
    bool operator!=(const LazyTensor& other) const;

    ov::Tensor eval() const;

    std::size_t get_hash() const;

private:
    std::shared_ptr<LazyTensorImpl> m_impl = nullptr;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
