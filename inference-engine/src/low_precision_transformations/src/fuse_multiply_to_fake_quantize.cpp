// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation, "FuseMultiplyToFakeQuantizeTransformation", 0);

FuseMultiplyToFakeQuantizeTransformation::FuseMultiplyToFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::Multiply>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "FuseMultiplyToFakeQuantizeTransformation");
    this->register_matcher(m, callback);
}

bool FuseMultiplyToFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    const auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return false;
    }

    const auto parent = multiply->get_input_node_shared_ptr(0);
    auto fakeQuantize = as_type_ptr<opset1::FakeQuantize>(parent);
    const auto convert = as_type_ptr<opset1::Convert>(parent);

    if (convert) {
        fakeQuantize = as_type_ptr<opset1::FakeQuantize>(convert->get_input_node_shared_ptr(0));
    }

    const auto multiplyConstant = multiply->get_input_node_shared_ptr(1);

    auto outputLowConst_f32 = foldConvert(fakeQuantize->get_input_node_shared_ptr(3), deqPrecision);
    auto outputHighConst_f32 = foldConvert(fakeQuantize->get_input_node_shared_ptr(4), deqPrecision);

    const auto value = multiplyConstant->get_output_element_type(0) == element::f32 ?
        multiplyConstant :
        foldConvert(multiplyConstant, deqPrecision);

    outputLowConst_f32 = fold<opset1::Multiply>(outputLowConst_f32, value);
    outputHighConst_f32 = fold<opset1::Multiply>(outputHighConst_f32, value);

    const auto fakeQuantizeParent = fakeQuantize->get_input_node_shared_ptr(0);
    const size_t parentIndex = NetworkHelper::getParentOutputIndex(fakeQuantizeParent, fakeQuantize);

    const auto inputLow = foldConvert(fakeQuantize->input_value(1), deqPrecision);
    const auto inputHigh = foldConvert(fakeQuantize->input_value(2), deqPrecision);
    NetworkHelper::copyInfo(fakeQuantize->get_input_node_shared_ptr(1), inputLow);
    NetworkHelper::copyInfo(fakeQuantize->get_input_node_shared_ptr(2), inputHigh);
    NetworkHelper::copyInfo(fakeQuantize->get_input_node_shared_ptr(3), outputLowConst_f32);
    NetworkHelper::copyInfo(fakeQuantize->get_input_node_shared_ptr(4), outputHighConst_f32);

    auto newFakeQuantize = std::make_shared<op::TypeRelaxed<opset1::FakeQuantize>>(
        opset1::FakeQuantize(
            fakeQuantizeParent->output(parentIndex),
            inputLow,
            inputHigh,
            outputLowConst_f32,
            outputHighConst_f32,
            fakeQuantize->get_levels()),
        multiply->get_output_element_type(0));

    replace_node(multiply, newFakeQuantize);
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize);

    const auto intervalAlignment = getAttribute<IntervalsAlignmentAttributePtr>(fakeQuantize);
    if ((intervalAlignment != nullptr) && (intervalAlignment->get()->levels != 0ul)) {
        newFakeQuantize->set_levels(intervalAlignment->get()->levels);
    }

    updateOutput(context, newFakeQuantize, multiply);
    return true;
}

bool FuseMultiplyToFakeQuantizeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!is_type<opset1::Constant>(operation->get_input_node_shared_ptr(1))) {
        return false;
    }

    if (!FakeQuantizeTransformation::checkElementwise(operation)) {
        return false;
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

bool FuseMultiplyToFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
