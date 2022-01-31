// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/utilities.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset5.hpp"

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

bool fuseTypeToStaticShapeNonMaxSuppression(const std::shared_ptr<ngraph::Node>& node, ngraph::element::Type to, size_t idx) {
    if (auto nms = ngraph::as_type_ptr<ngraph::vpu::op::StaticShapeNonMaxSuppression>(node)) {
        nms->set_output_type(to);
        return true;
    }
    return false;
}

bool fuseTypeToStaticShapeNonZero(const std::shared_ptr<ngraph::Node>& node, ngraph::element::Type to, size_t idx) {
    if (auto nz = ngraph::as_type_ptr<ngraph::vpu::op::StaticShapeNonZero>(node)) {
        nz->set_output_type(to);
        return true;
    }
    return false;
}

bool fuseTypeToStaticShapeTopK(const std::shared_ptr<ngraph::Node>& node, ngraph::element::Type to, size_t idx) {
    if (auto topk = ngraph::as_type_ptr<ngraph::vpu::op::StaticShapeTopK>(node)) {
       if (idx == 1 && (to == ngraph::element::i32 || to == ngraph::element::i64)) {
            topk->set_index_element_type(to);
            return true;
        }
    }
    return false;
}

bool fuseTypeToOutShapeOfReshape(const std::shared_ptr<ngraph::Node>& node, ngraph::element::Type to, size_t idx) {
    if (auto osr = ngraph::as_type_ptr<ngraph::vpu::op::OutShapeOfReshape>(node)) {
        osr->set_output_type(to);
        return true;
    }
    return false;
}
} // namespace vpu
