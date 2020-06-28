// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/fuse_fake_quantize.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void FuseFakeQuantizeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    // TODO: patterns below doesn't work - explore later:

    // addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<opset1::FakeQuantize>({ make_op_label<opset1::Multiply>() }));

    // addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<opset1::FakeQuantize>({ make_op_label<opset1::Subtract>() }));

    // addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<opset1::FakeQuantize>({ make_op_label<opset1::Convert>() }));

    // TODO: workaround
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

void FuseFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

    std::shared_ptr<opset1::FakeQuantize> fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(m.get_match_root());
    const FakeQuantizeDequantization dequantization = ngraph::pass::low_precision::getDequantization(fakeQuantize->shared_from_this());
    if (dequantization.empty()) {
        return;
    }

    std::cout << "FuseFakeQuantizeTransformation::transform: STARTED: " << fakeQuantize->get_friendly_name() << std::endl;

    //if (dequantization.multiply != nullptr) {
    //    std::shared_ptr<opset1::Constant> multiplyConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));

    //    if (dequantization.multiply->get_input_shape(1).size() > 1ul) {
    //        if (!multiplyConst->get_all_data_elements_bitwise_identical()) {
    //            // TODO: can we fuse not only per tensor dequantization values?
    //            std::cout << "FuseFakeQuantizeTransformation::transform: " << fakeQuantize->get_friendly_name() << ": " << dequantization.multiply->get_input_shape(1) << std::endl;
    //            return;
    //        }
    //    }
    //}

    std::shared_ptr<Node> parent;
    std::shared_ptr<Node> inputLowConst = fakeQuantize->get_input_node_shared_ptr(1);
    std::shared_ptr<Node> inputHightConst = fakeQuantize->get_input_node_shared_ptr(2);
    // std::shared_ptr<Node> outputLowConst = fakeQuantize->get_input_node_shared_ptr(3);
    // std::shared_ptr<Node> outputHightConst = fakeQuantize->get_input_node_shared_ptr(4);

    if (dequantization.multiply != nullptr) {
        inputLowConst = fold<opset1::Divide>(inputLowConst, dequantization.multiply->get_input_node_shared_ptr(1));
        inputHightConst = fold<opset1::Divide>(inputHightConst, dequantization.multiply->get_input_node_shared_ptr(1));

        // TODO: dims hardcoded: only for 3D
        Shape shape = inputLowConst->get_output_shape(0);
        if ((shape.size() < fakeQuantize->get_output_shape(0).size()) && (shape.size() > 0)) {
            inputLowConst = fold<opset1::Unsqueeze>(
                inputLowConst,
                std::make_shared<opset1::Constant>(
                    ngraph::element::i32,
                    Shape{ 1 },
                    std::vector<size_t>({ 0ul })));

            inputHightConst = fold<opset1::Unsqueeze>(
                inputHightConst,
                std::make_shared<opset1::Constant>(
                    ngraph::element::i32,
                    Shape{ 1 },
                    std::vector<size_t>({ 0ul })));
        }
        parent = dequantization.multiply->get_input_node_shared_ptr(0);
    }

    if (dequantization.subtract != nullptr) {
        // TODO: NO TESTS!!!
        inputLowConst = fold<opset1::Add>(inputLowConst, dequantization.subtract->get_input_node_shared_ptr(1));
        inputHightConst = fold<opset1::Add>(inputHightConst, dequantization.subtract->get_input_node_shared_ptr(1));
        parent = dequantization.subtract->get_input_node_shared_ptr(0);
    }

    if (dequantization.convert != nullptr) {
        parent = dequantization.convert->get_input_node_shared_ptr(0);
    }

    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = std::make_shared<ngraph::op::TypeRelaxed<opset1::FakeQuantize>>(
        parent,
        inputLowConst,
        inputHightConst,
        fakeQuantize->input_value(3),
        fakeQuantize->input_value(4),
        fakeQuantize->get_levels(),
        fakeQuantize->get_auto_broadcast());
    replace_node(fakeQuantize, newFakeQuantize);
    newFakeQuantize->set_friendly_name(fakeQuantize->get_friendly_name());

    std::cout << "FuseFakeQuantizeTransformation::transform: DONE: " << fakeQuantize->get_friendly_name() << std::endl;

    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
