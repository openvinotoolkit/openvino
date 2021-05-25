// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/tile.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

TileTransformation::TileTransformation(const Params& params) : LayerTransformation(params) {
}

void TileTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Tile>({ make_op_label<opset1::Multiply>() }));
}

bool TileTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op);
    if (dequantization.empty()) {
        return false;
    }

    if (!NetworkHelper::isScalarLike(dequantization.multiplyConstant)) {
        const auto inputShape = op->input(0).get_shape();
        const auto outputShape = op->output(0).get_shape();
        if ((inputShape.size() < 2ul) || (inputShape[0] != outputShape[0]) || (inputShape[1] != outputShape[1])) {
            return false;
        }
    }

    return true;
}

bool TileTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> tile = NetworkHelper::separateInStandaloneBranch(m.get_match_root());
    moveDequantizationAfter(context, tile, NetworkHelper::getDequantization(tile), false, true);
    return true;
}

bool TileTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
