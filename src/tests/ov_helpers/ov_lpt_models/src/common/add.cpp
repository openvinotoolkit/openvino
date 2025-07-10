// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/add.hpp"

namespace ov {
namespace builder {
namespace subgraph {

Add::Add() : isEmpty(true), outPrecision(ov::element::dynamic), constantShapeIsDefined(false) {}

Add::Add(const float value)
    : isEmpty(false),
      values({value}),
      outPrecision(ov::element::dynamic),
      constantShapeIsDefined(false) {}

Add::Add(const std::vector<float>& values)
    : isEmpty(values.empty()),
      values(values),
      outPrecision(ov::element::dynamic),
      constantShapeIsDefined(false) {}

Add::Add(const std::vector<float>& values, const ov::element::Type outPrecision) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShapeIsDefined(false) {
}

Add::Add(
    const std::vector<float>& values,
    const ov::element::Type outPrecision,
    const ov::Shape& constantShape) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShape(constantShape),
    constantShapeIsDefined(true) {
}

bool Add::empty() const noexcept {
    return isEmpty;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
