// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/avg_pool.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_ops/multiply_add.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void AvgPoolTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::AvgPool>({ make_op_label<opset1::Multiply>() }));
}

void AvgPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> pooling = m.get_match_root();

    // TODO: mote to TransparentBaseTransformation implementation AFTER TESTS
    const FakeQuantizeDequantization dequantization = ngraph::pass::low_precision::getDequantization(pooling->shared_from_this());
    if (dequantization.empty()) {
        return;
    }

    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

    std::shared_ptr<Node> dataNode =
        dequantization.convert != nullptr ? dequantization.convert->get_input_node_shared_ptr(0) :
        (dequantization.subtract != nullptr ?
            dequantization.subtract->get_input_node_shared_ptr(0) :
            dequantization.multiply->get_input_node_shared_ptr(0));

    // removing dequantization operations
    std::shared_ptr<Node> newPooling = pooling->clone_with_new_inputs({ dataNode });
    pass::low_precision::NetworkHelper::setOutDataPrecision(newPooling, dequantization.multiply->get_output_element_type(0));
    replace_node(pooling, newPooling);
    newPooling->set_friendly_name(pooling->get_friendly_name());

    for (int i = 0; i < newPooling->get_output_size(); ++i) {
        const auto childInputs = newPooling->get_output_target_inputs(i);
        for (const auto childInput : childInputs) {
            std::shared_ptr<Node> source = newPooling;
            std::shared_ptr<Node> destination = childInput.get_node()->shared_from_this();

            if (dequantization.subtract != nullptr) {
                // TODO: why this line is not working?
                // insert_new_node_between(source, destination, dequantization.subtract);

                std::shared_ptr<ngraph::opset1::Subtract> subtract = std::make_shared<ngraph::opset1::Subtract>(
                    source,
                    dequantization.subtract->get_input_node_shared_ptr(1));

                insert_new_node_between(source, destination, subtract);
                source = dequantization.subtract;
            }

            if (dequantization.multiply != nullptr) {
                // TODO: why this line is not working?
                // insert_new_node_between(source, destination, dequantization.multiply);

                std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(
                    source,
                    dequantization.multiply->get_input_node_shared_ptr(1));

                insert_new_node_between(source, destination, multiply);
            }
        }
    }

    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

    // TODO: NAMES!
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
