// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/runtime/tensor.hpp"

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
    LazyTensor(const std::shared_ptr<ov::op::v0::Constant>& const_ptr);
    LazyTensor(const std::vector<LazyTensor>& to_concat, const std::size_t axis);  // construct from concat
    LazyTensor(const LazyTensor& cw,
               const LazyTensor& cz,
               const LazyTensor& cs,
               const ov::element::Type& type,
               const ov::Shape& shape);  // construct from unpack

    LazyTensor permute(const std::vector<std::size_t>& axes);
    LazyTensor convert(const ov::element::Type& type);

    bool operator==(const LazyTensor& other) const;
    bool operator!=(const LazyTensor& other) const;

    ov::Tensor eval() const;
    std::size_t get_hash() const;
    void detach();

private:
    std::shared_ptr<LazyTensorImpl> m_impl = nullptr;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
