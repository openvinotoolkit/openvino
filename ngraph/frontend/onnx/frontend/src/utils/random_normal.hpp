// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/reshape.hpp"
#include "ngraph/output_vector.hpp"

namespace ngraph {
namespace onnx_import {
namespace detail {

OutputVector make_random_normal(Shape shape, element::Type type, float mean, float scale);

OutputVector make_random_normal(Shape shape, element::Type type, float mean, float scale, float seed);

OutputVector box_muller(Shape shape, element::Type type, float mean, float scale, uint64_t op_seed = 0, uint64_t global_seed = 0);

}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
