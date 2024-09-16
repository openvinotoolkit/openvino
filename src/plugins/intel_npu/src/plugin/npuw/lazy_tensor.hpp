// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace weights {

enum class TransformType {
    ORIG,
    PERMUTE,
    CONVERT,
    CONCAT  // TODO: support
};

class LazyTensor {
public:
    using Transform = std::variant<ov::Tensor, std::vector<std::size_t>, std::monostate>;

    class Hash {
    public:
        std::size_t operator()(const LazyTensor& lt) {
            // FIXME: implement
            return 0;
        }
    };

    explicit LazyTensor() = default;

    bool operator==(const LazyTensor& other) {
        // FIXME: implement
        return false;
    }

private:
    std::list<std::pair<TransformType, Transform>> m_transforms;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
