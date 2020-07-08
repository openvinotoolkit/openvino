// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/fake_quantize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void FakeQuantizeTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

void FakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto layer = std::dynamic_pointer_cast<opset1::FakeQuantize>(m.get_match_root());

    const std::deque<descriptor::Output> outputs = layer->get_outputs();
    const ngraph::element::Type precision = outputs.begin()->get_element_type();
    // TODO: extract to separate method (isQuantized)
    // TODO: use supported precisions
    if ((precision == ngraph::element::i8) || (precision == ngraph::element::u8)) {
        return;
    }

    // FakeQuantize on weights are used without dequantization ScaleShifts
    // TODO: include into the transformation pattern?
     if (NetworkHelper::onWeights(layer)) {
        return;
     }

    if (!QuantizationDetails::outputLayoutIsSupported(layer)) {
        return;
    }

    //// Gather Multiply from the data path
    // if (auto multiply = as_type_ptr<opset1::Multiply>(layer->input_value(0).get_node_shared_ptr())) {
    //    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(multiply->input_value(0).get_node_shared_ptr());
    //    auto data = multiply->input_value(1);
    //    if (!constant) {
    //        constant = as_type_ptr<opset1::Constant>(multiply->input_value(1).get_node_shared_ptr());
    //        data = multiply->input_value(0);
    //    }

    //    if (constant) {
    //        // TODO: Check multiply consumers
    //        // TODO: verify that direct multiplication is correct
    //        auto newInputMin = fold<opset1::Divide>(layer->input_value(1), constant);
    //        auto newInputMax = fold<opset1::Divide>(layer->input_value(2), constant);
    //        // FIXME: workaround for current CPU implementation that has restrictions on shapes:
    //        auto newShape = newInputMin->get_output_shape(0);
    //        // FIXME: eshoguli: workaround for workaround to avoid 5D tensor
    //        if (newShape.size() != 4ul) {
    //            newShape.insert(newShape.begin(), 1);
    //        }
    //        newInputMin = fold_reshape<opset1::Reshape>(newInputMin, std::make_shared<opset1::Constant>(element::i64, Shape{4}, newShape), false);
    //        newInputMax = fold_reshape<opset1::Reshape>(newInputMax, std::make_shared<opset1::Constant>(element::i64, Shape{4}, newShape), false);
    //        auto newFQ = layer->copy_with_new_inputs({data, newInputMin, newInputMax, layer->input_value(3), layer->input_value(4)});
    //        replace_node(layer, newFQ);
    //        layer = as_type_ptr<opset1::FakeQuantize>(newFQ);
    //    }
    // }

    //// TODO: can we handle it by marking FQs that we wanted to exclude in RTinfo
    ////       (in previous passes where quantizedFakeQuantizeNames has been populated)
    // const std::string layerName = layer->get_friendly_name();
    // if (context.quantizedFakeQuantizeNames.find(layerName) != context.quantizedFakeQuantizeNames.end()) {
    //    return;
    // }

    if (!QuantizationDetails::isSupportedLevel(layer->get_levels())) {
        return;
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    const DataPrecision dataPrecision = getDataPrecision(layer, quantizationDetails, false, supportAsymmetricQuantization);
    if (dataPrecision.precision == element::undefined) {
        return;
    }

#if 0 // replaced by decomposeFakeQuantize
    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    fillFromQuantizationDetails(
            quantizationDetails,
            dataPrecision,
            dequantizationScales,
            dequantizationShifts);
#endif

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationValues(dequantizationScales, dequantizationShifts);
#endif

    // Split FakeQuantize to two parts: Quantize and Dequantize
    auto QDQ = decomposeFakeQuantize(
        as_type_ptr<opset1::FakeQuantize>(layer),
        dataPrecision.precision,
        dataPrecision.min,
        dataPrecision.max,
        updatePrecisions);

    std::vector<std::shared_ptr<ngraph::Function>> transformedModule{ context.network };
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);

    // To disable application of the same transform twice on the same node
    // TODO: Handle it through node property
    std::shared_ptr<opset1::FakeQuantize> quantize = as_type_ptr<opset1::FakeQuantize>(std::get<0>(QDQ));
    if (quantize != nullptr) {
        auto quantizeConvert = as_type_ptr<opset1::Convert>(quantize->get_output_target_inputs(0).begin()->get_node()->shared_from_this());
        if (quantizeConvert != nullptr) {
            // Remove the first Convert and built convert directly to FQ by modifying output type
            NetworkHelper::setOutDataPrecision(quantize, quantizeConvert->get_output_element_type(0));
            NetworkHelper::removeLayer(quantizeConvert);
        }
    }

    // TODO: hardcoded
    // NetworkHelper::setOutDataPrecision(quantize, element::u8);

    // TODO: for debuging only - remove later
    auto dequantize = as_type_ptr<ngraph::Node>(std::get<1>(QDQ));
    // dequantize->set_friendly_name(layer->get_friendly_name());


    // TODO: Get rid of this.
    // const std::string friendlyName = layer->get_friendly_name();
    // context.quantizedFakeQuantizeNames.insert(friendlyName);

    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);

    // std::cout << "FakeQuantizeTransformation::transform: " << layer->get_friendly_name() << std::endl;

    updateOutput(context, dequantize, layer);
}

bool FakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
