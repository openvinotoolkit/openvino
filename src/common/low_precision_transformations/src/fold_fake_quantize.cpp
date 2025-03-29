// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fold_fake_quantize.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FoldFakeQuantizeTransformation::FoldFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(FoldFakeQuantizeTransformation);
    auto fakeQuantize = pattern::wrap_type<ov::opset1::FakeQuantize>();

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fakeQuantize, matcher_name);
    this->register_matcher(m, callback);
}

bool FoldFakeQuantizeTransformation::transform(ov::pass::pattern::Matcher &m) {
    const auto fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(m.get_match_root());
    if (fakeQuantize == nullptr) {
        return false;
    }

    if (!canBeTransformed(fakeQuantize)) {
        return false;
    }

    const auto constantShape = fakeQuantize->input(1).get_partial_shape();
    if (constantShape.is_dynamic()) {
        return false;
    }

    const auto resultConstant = NetworkHelper::fold_fake_quantize(fakeQuantize, false);
    if (ov::is_type<ov::opset1::Constant>(resultConstant)) {
        replace_node(fakeQuantize, resultConstant);
        return true;
    }

    return false;
}

bool FoldFakeQuantizeTransformation::isConstantOutput(std::shared_ptr<ov::Node> node) const {
    const auto fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(node);
    if (!fakeQuantize) {
        return false;
    }

    const auto outputLow = as_type_ptr<ov::opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(3));
    const auto outputHigh = as_type_ptr<ov::opset1::Constant>(fakeQuantize->get_input_node_shared_ptr(4));

    if (outputLow == nullptr || outputHigh == nullptr) {
        return false;
    }

    const auto vecLow = outputLow->cast_vector<float>();
    const auto vecHigh = outputHigh->cast_vector<float>();

    return vecLow == vecHigh;
}

bool FoldFakeQuantizeTransformation::canBeTransformed(const std::shared_ptr<Node>& op) const {
    if (!NetworkHelper::isConstantPath(op) && !isConstantOutput(op)) {
        return false;
    }

    const auto fq = ov::as_type_ptr<ov::opset1::FakeQuantize>(op);
    if (!fq) {
        return false;
    }

    for (size_t i = 1; i < fq->get_input_size(); ++i) {
        const auto& shape = fq->get_input_shape(i);
        if (std::count_if(shape.begin(), shape.end(), [](size_t x) { return x > 1; }) > 1) {
            return false;
        }
    }

    return true;
}

bool FoldFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
