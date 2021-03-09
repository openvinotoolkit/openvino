// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/fake_quantize.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FuseSubtractToFakeQuantizeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Subtract>(pass, context);
}

bool FuseSubtractToFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    const auto subtract = m.get_match_root();
    if (!canBeTransformed(context, subtract)) {
        return false;
    }

    const auto parent = subtract->get_input_node_shared_ptr(0);
    auto fakeQuantize = as_type_ptr<opset1::FakeQuantize>(parent);
    const auto convert = as_type_ptr<opset1::Convert>(parent);

    if (convert) {
        fakeQuantize = as_type_ptr<opset1::FakeQuantize>(convert->get_input_node_shared_ptr(0));
    }

    const auto subtractConstant = subtract->get_input_node_shared_ptr(1);

    auto outputLowConst_f32 = fold<opset1::Convert>(fakeQuantize->get_input_node_shared_ptr(3), deqPrecision);
    auto outputHighConst_f32 = fold<opset1::Convert>(fakeQuantize->get_input_node_shared_ptr(4), deqPrecision);

    const auto value = subtractConstant->get_output_element_type(0) == element::f32 ?
        subtractConstant :
        fold<opset1::Convert>(subtractConstant, deqPrecision);

    outputLowConst_f32 = fold<opset1::Subtract>(outputLowConst_f32, value);
    outputHighConst_f32 = fold<opset1::Subtract>(outputHighConst_f32, value);

    const auto fakeQuantizeParent = fakeQuantize->get_input_node_shared_ptr(0);
    const size_t parentIndex = NetworkHelper::getParentOutputIndex(fakeQuantizeParent, fakeQuantize);

    auto newFakeQuantize = std::make_shared<op::TypeRelaxed<opset1::FakeQuantize>>(
        opset1::FakeQuantize(
            fakeQuantizeParent->output(parentIndex),
            fold<opset1::Convert>(fakeQuantize->input_value(1), deqPrecision),
            fold<opset1::Convert>(fakeQuantize->input_value(2), deqPrecision),
            outputLowConst_f32,
            outputHighConst_f32,
            fakeQuantize->get_levels()),
        subtract->get_output_element_type(0));

    replace_node(subtract, newFakeQuantize);
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize);

    updateOutput(context, newFakeQuantize, subtract);
    return true;
}

bool FuseSubtractToFakeQuantizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!is_type<opset1::Constant>(operation->get_input_node_shared_ptr(1))) {
        return false;
    }

    if (!FakeQuantizeTransformation::checkElementwise(operation)) {
        return false;
    }

    const auto childs = operation->get_output_target_inputs(0);

    for (const auto& target : childs) {
        const auto convolution = is_type<opset1::Convolution>(target.get_node());
        const auto groupConvolution = is_type<opset1::GroupConvolution>(target.get_node());
        if (convolution || groupConvolution) {
            return false;
        }
    }

    const auto parent = operation->get_input_node_shared_ptr(0);
    auto fq = as_type_ptr<opset1::FakeQuantize>(parent);
    const auto convert = as_type_ptr<opset1::Convert>(parent);

    if (convert) {
        fq = as_type_ptr<opset1::FakeQuantize>(convert->get_input_node_shared_ptr(0));
    }

    if (!fq) {
        return false;
    }

    if (fq->get_output_target_inputs(0).size() != 1) {
        return false;
    }

    return true;
}

bool FuseSubtractToFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
