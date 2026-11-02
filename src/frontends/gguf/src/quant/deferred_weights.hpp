// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <functional>
#include <optional>

#include "openvino/runtime/tensor.hpp"

namespace ov::frontend::gguf {

struct WeightTensors;

bool is_deferred(const ov::Tensor& tensor);
bool is_deferred(const WeightTensors& tensors);
bool is_materialized(const ov::Tensor& tensor);
size_t get_materialization_count(const ov::Tensor& tensor);

ov::Tensor defer_tensor(const ov::element::Type& type, const ov::Shape& shape, std::function<ov::Tensor()> materialize);
std::optional<WeightTensors> concat_deferred_rows(const WeightTensors& lhs, const WeightTensors& rhs);

struct DeferredWeightTensor {
    ov::element::Type type = ov::element::dynamic;
    ov::Shape shape;
};

WeightTensors defer_weight_tensors(const std::array<DeferredWeightTensor, 3>& components,
                                   std::function<void(WeightTensors&)> repack);

}  // namespace ov::frontend::gguf
