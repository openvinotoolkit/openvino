// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/tile.hpp"
#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::TileTransformation, "TileTransformation", 0);

TileTransformation::TileTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::Tile>({ pattern::wrap_type<opset1::Multiply>(), pattern::any_input() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "TileTransformation");
    this->register_matcher(m, callback);
}

bool TileTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> tile = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(tile, defaultPrecisions);
    dequantization.scalarizeConstants(); // scalarize because Tile can have new channels count on output

    moveDequantizationAfter(context, tile, dequantization, false);
    return true;
}

bool TileTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    if (dequantization.multiply == nullptr || !NetworkHelper::isScalarLike(dequantization.multiplyConstant)) {
        return false;
    }

    if (dequantization.subtract != nullptr && !NetworkHelper::isScalarLike(dequantization.subtractConstant)) {
        return false;
    }

    return true;
}

bool TileTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
