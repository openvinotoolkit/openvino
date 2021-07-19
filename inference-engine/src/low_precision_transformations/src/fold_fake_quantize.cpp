// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fold_fake_quantize.hpp"

#include <memory>
#include <string>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::FoldFakeQuantizeTransformation, "FoldFakeQuantizeTransformation", 0);

FoldFakeQuantizeTransformation::FoldFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    auto fakeQuantize = pattern::wrap_type<opset1::FakeQuantize>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fakeQuantize, "FoldFakeQuantizeTransformation");
    this->register_matcher(m, callback);
}

bool FoldFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    const auto fakeQuantize = as_type_ptr<opset1::FakeQuantize>(m.get_match_root());
    if (fakeQuantize == nullptr) {
        return false;
    }

    if (!canBeTransformed(context, fakeQuantize)) {
        return false;
    }

    const auto resultConstant = NetworkHelper::fold_fake_quantize(fakeQuantize, false);
    if (is_type<opset1::Constant>(resultConstant)) {
        replace_node(fakeQuantize, resultConstant);
        return true;
    }

    return false;
}

bool FoldFakeQuantizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    return NetworkHelper::isConstantPath(op);
}

bool FoldFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
