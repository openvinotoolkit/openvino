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

    std::shared_ptr<opset1::FakeQuantize> fakeQuantize = layer;

    do {
        layer = fakeQuantize;
        fakeQuantize = handle(context, fakeQuantize);
    } while (fakeQuantize != nullptr);

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
    const DataPrecision dataPrecision = getDataPrecision(layer, quantizationDetails, false);
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

static std::shared_ptr<Node> updateShape(std::shared_ptr<Node> op, const Shape& targetShape) {
    const Shape shape = op->get_output_shape(0);
    if ((shape.size() < targetShape.size()) && (shape.size() > 1ul)) {
        op = fold<opset1::Unsqueeze>(
            op,
            std::make_shared<opset1::Constant>(ngraph::element::i32, Shape{ 1 }, std::vector<size_t>({ 0ul })));
    }
    return op;
}

static std::shared_ptr<Node> getData(const std::shared_ptr<Node>& eltwise) {
    if (!is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(0))) {
        return eltwise->get_input_node_shared_ptr(0);
    }

    if (!is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(1))) {
        return eltwise->get_input_node_shared_ptr(1);
    }

    return nullptr;
}

static std::shared_ptr<opset1::Constant> getConstant(const std::shared_ptr<Node>& eltwise) {
    if (eltwise->get_input_size() != 2) {
        return nullptr;
    }

    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(1));
    if (constant != nullptr) {
        return constant;
    }

    return as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(0));
}

static bool eltwiseWithConstant(const std::shared_ptr<Node>& eltwise) {
    std::shared_ptr<opset1::Constant> constant = getConstant(eltwise);
    if (constant == nullptr) {
        return false;
    }

    Shape shape = constant->get_output_shape(0);
    if ((!shape.empty()) && (shape_size(shape) != 1ul)) {
        const Shape eltwiseShape = eltwise->get_output_shape(0);
        if ((eltwiseShape.size() - shape.size()) > 1) {
            return false;
        }

        if ((eltwiseShape.size() - shape.size()) == 1ul) {
            shape.insert(shape.begin(), 1ul);
        }

        for (size_t i = 2ul; i < shape.size(); ++i) {
            if (shape[i] != 1ul) {
                return false;
            }
        }
    }

    return getData(eltwise) != nullptr;
}

std::shared_ptr<opset1::FakeQuantize> FakeQuantizeTransformation::handle(
    TransformationContext& context,
    const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const {
    const std::shared_ptr<Node> eltwise = fakeQuantize->get_input_node_shared_ptr(0);

    std::shared_ptr<Node> inputLowConst = fakeQuantize->get_input_node_shared_ptr(1);
    std::shared_ptr<Node> inputHightConst = fakeQuantize->get_input_node_shared_ptr(2);

    std::shared_ptr<opset1::Constant> constant = getConstant(eltwise);
    if (is_type<opset1::Multiply>(eltwise) && eltwiseWithConstant(eltwise)) {
        const auto value = constant->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            constant :
            fold<opset1::Convert>(constant, eltwise->get_output_element_type(0));

        inputLowConst = updateShape(fold<opset1::Divide>(inputLowConst, value), fakeQuantize->get_output_shape(0));
        inputHightConst = updateShape(fold<opset1::Divide>(inputHightConst, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Divide>(eltwise) && eltwiseWithConstant(eltwise)) {
        const auto value = constant->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            constant :
            fold<opset1::Convert>(constant, eltwise->get_output_element_type(0));

        inputLowConst = updateShape(fold<opset1::Multiply>(inputLowConst, value), fakeQuantize->get_output_shape(0));
        inputHightConst = updateShape(fold<opset1::Multiply>(inputHightConst, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Subtract>(eltwise) && eltwiseWithConstant(eltwise)) {
        const auto value = constant->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            constant :
            fold<opset1::Convert>(constant, eltwise->get_output_element_type(0));

        inputLowConst = updateShape(fold<opset1::Add>(inputLowConst, value), fakeQuantize->get_output_shape(0));
        inputHightConst = updateShape(fold<opset1::Add>(inputHightConst, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Add>(eltwise) && eltwiseWithConstant(eltwise)) {
        if (is_type<opset1::Convolution>(getData(eltwise)) ||
            is_type<opset1::GroupConvolution>(getData(eltwise))) {
            return nullptr;
        }

        const auto value = constant->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            constant :
            fold<opset1::Convert>(constant, eltwise->get_output_element_type(0));

        inputLowConst = updateShape(fold<opset1::Subtract>(inputLowConst, value), fakeQuantize->get_output_shape(0));
        inputHightConst = updateShape(fold<opset1::Subtract>(inputHightConst, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Convert>(eltwise)) {
        //
    } else {
        return nullptr;
    }

    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = as_type_ptr<opset1::FakeQuantize>(fakeQuantize->clone_with_new_inputs({
        getData(eltwise),
        inputLowConst,
        inputHightConst,
        fakeQuantize->input_value(3),
        fakeQuantize->input_value(4) }));

    replace_node(fakeQuantize, newFakeQuantize);
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize);

    return newFakeQuantize;
}

bool FakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
