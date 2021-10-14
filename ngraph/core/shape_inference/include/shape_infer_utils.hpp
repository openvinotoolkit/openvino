// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <ngraph/partial_shape.hpp>

namespace ShapeInfer {
template <class T>
void inline default_work(T& shape) {
    OPENVINO_UNREACHABLE("[ShapeInfer]Not Reachable");
}

template <>
void inline default_work(ov::PartialShape& shape) {
    shape = ov::PartialShape::dynamic();
}
}  // namespace ShapeInfer
