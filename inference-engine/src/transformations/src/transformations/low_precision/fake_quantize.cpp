// Copyright (C) 2020 Intel Corporation
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

namespace ngraph {
namespace pass {
namespace low_precision {

void FakeQuantizeTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

bool FakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::FakeQuantize> layer = std::dynamic_pointer_cast<opset1::FakeQuantize>(m.get_match_root());

    const ngraph::element::Type precision = layer->get_output_element_type(0);
    if ((precision == ngraph::element::i8) || (precision == ngraph::element::u8)) {
        return false;
    }

    // FakeQuantize on weights are used without dequantization ScaleShifts
    if (NetworkHelper::onWeights(layer)) {
        return false;
    }

    if (!QuantizationDetails::outputLayoutIsSupported(layer)) {
        return false;
    }

    if (!QuantizationDetails::isSupportedLevel(layer->get_levels())) {
        return false;
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    const DataPrecision dataPrecision = getDataPrecision(layer, quantizationDetails, false, supportAsymmetricQuantization);
    if (dataPrecision.precision == element::undefined) {
        return false;
    }

    // Split FakeQuantize to two parts: Quantize and Dequantize
    auto QDQ = NetworkHelper::decomposeFakeQuantize(
        as_type_ptr<opset1::FakeQuantize>(layer),
        dataPrecision.precision,
        dataPrecision.min,
        dataPrecision.max,
        dataPrecision.hasZeroPoint,
        updatePrecisions);

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    {
        const std::shared_ptr<opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(std::get<1>(QDQ));
        const std::shared_ptr<opset1::Constant> multiplyConst = as_type_ptr<opset1::Constant>(multiply->get_input_node_shared_ptr(1));
        const std::vector<float> dequantizationScales = multiplyConst->cast_vector<float>();

        const std::shared_ptr<opset1::Subtract> subtract = as_type_ptr<opset1::Subtract>(multiply->get_input_node_shared_ptr(0));
        std::vector<float> dequantizationShifts;
        if (subtract != nullptr) {
            const std::shared_ptr<opset1::Constant> subtractConst = as_type_ptr<opset1::Constant>(subtract->get_input_node_shared_ptr(1));
            dequantizationShifts = subtractConst->cast_vector<float>();
        } else {
            dequantizationShifts = std::vector<float>(dequantizationScales.size());
        }

        printDequantizationValues(dequantizationScales, dequantizationShifts);
    }
#endif

    std::shared_ptr<ngraph::Node> dequantize = std::get<1>(QDQ);
    updateOutput(context, dequantize, layer);
    return true;
}

bool FakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
