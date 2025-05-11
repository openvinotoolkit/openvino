// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/constant.hpp"

namespace ov {
namespace builder {
namespace subgraph {

Constant::Constant() : isEmpty(true), outPrecision(ov::element::dynamic), shapeIsDefined(false) {}

Constant::Constant(const float value)
    : isEmpty(false),
      values({value}),
      outPrecision(ov::element::dynamic),
      shapeIsDefined(false) {}

Constant::Constant(const std::vector<float>& values)
    : isEmpty(values.empty()),
      values(values),
      outPrecision(ov::element::dynamic),
      shapeIsDefined(false) {}

Constant::Constant(const std::vector<float>& values, const ov::element::Type outPrecision) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    shapeIsDefined(false) {
}

Constant::Constant(
    const std::vector<float>& values,
    const ov::element::Type outPrecision,
    const ov::Shape& shape) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    shape(shape),
    shapeIsDefined(true) {
}

bool Constant::empty() const noexcept {
    return isEmpty;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
