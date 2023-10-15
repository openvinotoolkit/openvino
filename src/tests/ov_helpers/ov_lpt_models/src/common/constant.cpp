// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/constant.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

Constant::Constant() :
    isEmpty(true),
    outPrecision(ngraph::element::undefined),
    shapeIsDefined(false)
{}

Constant::Constant(const float value) :
    isEmpty(false),
    values({ value }),
    outPrecision(ngraph::element::undefined),
    shapeIsDefined(false) {
}

Constant::Constant(const std::vector<float>& values) :
    isEmpty(values.empty()),
    values(values),
    outPrecision(ngraph::element::undefined),
    shapeIsDefined(false) {
}

Constant::Constant(const std::vector<float>& values, const ngraph::element::Type outPrecision) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    shapeIsDefined(false) {
}

Constant::Constant(
    const std::vector<float>& values,
    const ngraph::element::Type outPrecision,
    const ngraph::Shape& shape) :
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
}  // namespace ngraph
