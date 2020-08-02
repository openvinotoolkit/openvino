// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/fuse_fake_quantize.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FuseFakeQuantizeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

void FuseFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::FakeQuantize> fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(m.get_match_root());
    do {
        fakeQuantize = handle(context, fakeQuantize);
    } while (fakeQuantize != nullptr);
}

bool eltwiseWithConstant(const std::shared_ptr<Node> eltwise)  {
    return (eltwise->get_input_size() > 1ul) && is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(1));
}

std::shared_ptr<Node> updateShape(std::shared_ptr<Node> op, const Shape& targetShape) {
    const Shape shape = op->get_output_shape(0);
    if ((shape.size() < targetShape.size()) && (shape.size() > 1ul)) {
        op = fold<opset1::Unsqueeze>(
            op,
            std::make_shared<opset1::Constant>(ngraph::element::i32, Shape{ 1 }, std::vector<size_t>({ 0ul })));
    }
    return op;
}

std::shared_ptr<opset1::FakeQuantize> FuseFakeQuantizeTransformation::handle(
    TransformationContext& context,
    const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const {
    const std::shared_ptr<Node> eltwise = fakeQuantize->get_input_node_shared_ptr(0);

    std::shared_ptr<Node> inputLowConst = fakeQuantize->get_input_node_shared_ptr(1);
    std::shared_ptr<Node> inputHightConst = fakeQuantize->get_input_node_shared_ptr(2);

    if (is_type<opset1::Multiply>(eltwise) && eltwiseWithConstant(eltwise)) {
        const auto value = eltwise->get_input_node_shared_ptr(1)->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            eltwise->get_input_node_shared_ptr(1) :
            fold<opset1::Convert>(eltwise->get_input_node_shared_ptr(1), eltwise->get_output_element_type(0));

        inputLowConst = updateShape(fold<opset1::Divide>(inputLowConst, value), fakeQuantize->get_output_shape(0));
        inputHightConst = updateShape(fold<opset1::Divide>(inputHightConst, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Divide>(eltwise) && eltwiseWithConstant(eltwise)) {
        const auto value = eltwise->get_input_node_shared_ptr(1)->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            eltwise->get_input_node_shared_ptr(1) :
            fold<opset1::Convert>(eltwise->get_input_node_shared_ptr(1), eltwise->get_output_element_type(0));

        inputLowConst = updateShape(fold<opset1::Multiply>(inputLowConst, value), fakeQuantize->get_output_shape(0));
        inputHightConst = updateShape(fold<opset1::Multiply>(inputHightConst, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Subtract>(eltwise) && eltwiseWithConstant(eltwise)) {
        const auto value = eltwise->get_input_node_shared_ptr(1)->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            eltwise->get_input_node_shared_ptr(1) :
            fold<opset1::Convert>(eltwise->get_input_node_shared_ptr(1), eltwise->get_output_element_type(0));

        inputLowConst = updateShape(fold<opset1::Add>(inputLowConst, value), fakeQuantize->get_output_shape(0));
        inputHightConst = updateShape(fold<opset1::Add>(inputHightConst, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Add>(eltwise) && eltwiseWithConstant(eltwise)) {
        if (is_type<opset1::Convolution>(eltwise->get_input_node_shared_ptr(0)) ||
            is_type<opset1::GroupConvolution>(eltwise->get_input_node_shared_ptr(0))) {
            return nullptr;
        }

        const auto value = eltwise->get_input_node_shared_ptr(1)->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            eltwise->get_input_node_shared_ptr(1) :
            fold<opset1::Convert>(eltwise->get_input_node_shared_ptr(1), eltwise->get_output_element_type(0));

        inputLowConst = updateShape(fold<opset1::Subtract>(inputLowConst, value), fakeQuantize->get_output_shape(0));
        inputHightConst = updateShape(fold<opset1::Subtract>(inputHightConst, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Convert>(eltwise)) {
        //
    } else {
        return nullptr;
    }

    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = as_type_ptr<opset1::FakeQuantize>(fakeQuantize->clone_with_new_inputs({
        eltwise->get_input_node_shared_ptr(0),
        inputLowConst,
        inputHightConst,
        fakeQuantize->input_value(3),
        fakeQuantize->input_value(4) }));

    replace_node(fakeQuantize, newFakeQuantize);
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize);
    return newFakeQuantize;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
