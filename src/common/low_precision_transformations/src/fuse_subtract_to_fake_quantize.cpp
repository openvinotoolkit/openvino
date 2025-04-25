// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include <memory>

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"
#include "low_precision/rt_info/disable_cleanup_attribute.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FuseSubtractToFakeQuantizeTransformation::FuseSubtractToFakeQuantizeTransformation(const Params& params)
    : FuseElementwiseToFakeQuantizeTransformation(params) {
    MATCHER_SCOPE(FuseSubtractToFakeQuantizeTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Subtract>();

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool FuseSubtractToFakeQuantizeTransformation::transform(ov::pass::pattern::Matcher &m) {
    const auto subtract = m.get_match_root();
    if (!canBeTransformed(subtract)) {
        return false;
    }

    const auto parent = subtract->input_value(0).get_node_shared_ptr();
    auto fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(parent);
    const auto convert = ov::as_type_ptr<ov::opset1::Convert>(parent);

    if (convert) {
        fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(convert->input_value(0).get_node_shared_ptr());
    }

    const auto subtractConstant = subtract->input_value(1).get_node_shared_ptr();
    if (!ov::is_type<ov::opset1::Constant>(subtractConstant)) {
        return false;
    }

    auto outputLow = foldConvert(fakeQuantize->input_value(3), deqPrecision);
    auto outputHigh = foldConvert(fakeQuantize->input_value(4), deqPrecision);
    const auto subValue = foldConvert(subtractConstant, deqPrecision);

    outputLow = fold<ov::opset1::Subtract>(outputLow, subValue);
    outputHigh = fold<ov::opset1::Subtract>(outputHigh, subValue);

    const auto inputLow = foldConvert(fakeQuantize->input_value(1), deqPrecision);
    const auto inputHigh = foldConvert(fakeQuantize->input_value(2), deqPrecision);
    NetworkHelper::copyInfo(fakeQuantize->input_value(1).get_node_shared_ptr(), inputLow);
    NetworkHelper::copyInfo(fakeQuantize->input_value(2).get_node_shared_ptr(), inputHigh);
    NetworkHelper::copyInfo(fakeQuantize->input_value(3).get_node_shared_ptr(), outputLow);
    NetworkHelper::copyInfo(fakeQuantize->input_value(4).get_node_shared_ptr(), outputHigh);

    auto newFakeQuantize = std::make_shared<ov::op::TypeRelaxed<ov::opset1::FakeQuantize>>(
        ov::opset1::FakeQuantize(
            fakeQuantize->input_value(0),
            inputLow,
            inputHigh,
            outputLow,
            outputHigh,
            fakeQuantize->get_levels()),
        subtract->get_output_element_type(0));

    replace_node(subtract, newFakeQuantize);
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize);

    updateOutput(newFakeQuantize, subtract);
    return true;
}

bool FuseSubtractToFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
