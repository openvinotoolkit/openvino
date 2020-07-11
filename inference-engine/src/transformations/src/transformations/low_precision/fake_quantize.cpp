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

namespace ngraph {
namespace pass {
namespace low_precision {

void FakeQuantizeTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

void FakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::FakeQuantize> layer = std::dynamic_pointer_cast<opset1::FakeQuantize>(m.get_match_root());

    const std::deque<descriptor::Output> outputs = layer->get_outputs();
    const ngraph::element::Type precision = outputs.begin()->get_element_type();
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

    if (!QuantizationDetails::isSupportedLevel(layer->get_levels())) {
        return;
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    const DataPrecision dataPrecision = getDataPrecision(layer, quantizationDetails, false, supportAsymmetricQuantization);
    if (dataPrecision.precision == element::undefined) {
        return;
    }

    // Split FakeQuantize to two parts: Quantize and Dequantize
    auto QDQ = decomposeFakeQuantize(
        as_type_ptr<opset1::FakeQuantize>(layer),
        dataPrecision.precision,
        dataPrecision.min,
        dataPrecision.max,
        updatePrecisions);

    std::vector<std::shared_ptr<ngraph::Function>> transformedModule{ context.network };

    std::shared_ptr<opset1::FakeQuantize> quantize = as_type_ptr<opset1::FakeQuantize>(std::get<0>(QDQ));
    if (quantize != nullptr) {
        auto quantizeConvert = as_type_ptr<opset1::Convert>(quantize->get_output_target_inputs(0).begin()->get_node()->shared_from_this());
        if (quantizeConvert != nullptr) {
            // Remove the first Convert and built convert directly to FQ by modifying output type
            NetworkHelper::setOutDataPrecision(quantize, quantizeConvert->get_output_element_type(0));
            NetworkHelper::removeLayer(quantizeConvert);
        }
    }

    std::shared_ptr<ngraph::Node> dequantize = as_type_ptr<ngraph::Node>(std::get<1>(QDQ));
    updateOutput(context, dequantize, layer);
}

bool FakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
