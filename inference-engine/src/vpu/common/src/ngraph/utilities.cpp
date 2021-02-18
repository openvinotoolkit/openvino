// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/utilities.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/evaluator.hpp"

#include <numeric>

namespace vpu {

std::shared_ptr<ngraph::Node> shapeToConstant(const ngraph::element::Type& type, const ngraph::Shape& shape) {
    return ngraph::opset5::Constant::create(type, {shape.size()}, shape);
}

std::shared_ptr<ngraph::Node> gatherShapeElements(const ngraph::Output<ngraph::Node>& shape, int startIndex, size_t elemCount) {
    std::vector<int64_t> shapePart(elemCount);
    std::iota(shapePart.begin(), shapePart.end(), startIndex);

    return std::make_shared<ngraph::opset5::Gather>(
        shape,
        ngraph::opset5::Constant::create(ngraph::element::i64, {elemCount}, shapePart),
        ngraph::opset5::Constant::create(ngraph::element::i64, {}, {0}));
}

std::shared_ptr<ngraph::Node> gatherShapeElements(const ngraph::Output<ngraph::Node>& shape, const std::vector<int64_t>& indicesToGather) {
    return std::make_shared<ngraph::opset5::Gather>(
            shape,
            ngraph::opset5::Constant::create(ngraph::element::i64, {indicesToGather.size()}, indicesToGather),
            ngraph::opset5::Constant::create(ngraph::element::i64, {}, {0}));
}

}  // namespace vpu
