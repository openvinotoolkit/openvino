// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/add.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

Add::Add() :
    isEmpty(true),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false)
{}

Add::Add(const float value) :
    isEmpty(false),
    values({ value }),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false) {
}

Add::Add(const std::vector<float>& values) :
    isEmpty(values.empty()),
    values(values),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false) {
}

Add::Add(const std::vector<float>& values, const ngraph::element::Type outPrecision) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShapeIsDefined(false) {
}

Add::Add(
    const std::vector<float>& values,
    const ngraph::element::Type outPrecision,
    const ngraph::Shape& constantShape) :
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
}  // namespace ngraph
