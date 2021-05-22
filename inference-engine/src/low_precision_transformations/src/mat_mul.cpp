// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/mat_mul.hpp"

#include <numeric>
#include <memory>
#include <string>
#include <vector>

#include "low_precision/network_helper.hpp"
#include "low_precision/common/dequantization_op.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

bool MatMulTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::MatMul> matMul = as_type_ptr<opset1::MatMul>(m.get_match_root());
    if ((matMul == nullptr) || !canBeTransformed(context, matMul)) {
        return false;
    }

    matMul = as_type_ptr<opset1::MatMul>(NetworkHelper::separateInStandaloneBranch(matMul));
    if (!support3DTensorOnActivations && (matMul->input(0).get_shape().size() == 3ul)) {
        return false;
    }

    const auto dequantization1 = NetworkHelper::getDequantization(matMul, 0);
    auto dequantization2 = NetworkHelper::getDequantization(matMul, 1);

    if (dequantization2.empty()) {
        const std::shared_ptr<opset1::FakeQuantize> fakeQuantize =
            as_type_ptr<opset1::FakeQuantize>(dequantization2.data.get_node_shared_ptr());
        if (fakeQuantize != nullptr) {
            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
            const DataPrecision dataPrecision = getDataPrecision(fakeQuantize, quantizationDetails, true);

            auto tuple = NetworkHelper::decomposeFakeQuantize(
                fakeQuantize,
                dataPrecision.precision,
                dataPrecision.min,
                dataPrecision.max,
                dataPrecision.hasZeroPoint,
                updatePrecisions);

            dequantization2 = NetworkHelper::getDequantization(matMul, 1);
        }
    }

    if (dequantization2.subtract != nullptr) {
        NetworkHelper::optimizeSubtract(dequantization2.subtract);
        dequantization2 = NetworkHelper::getDequantization(matMul, 1);
    }

    const std::shared_ptr<opset1::MatMul> newMatMul = std::make_shared<ngraph::op::TypeRelaxed<opset1::MatMul>>(
        std::vector<element::Type>({ deqPrecision, deqPrecision }), std::vector<element::Type>({ deqPrecision }),
        ngraph::op::TemporaryReplaceOutputType(dequantization1.data, deqPrecision).get(),
        ngraph::op::TemporaryReplaceOutputType(dequantization2.data, deqPrecision).get(),
        matMul->get_transpose_a(),
        matMul->get_transpose_b());
    NetworkHelper::copyInfo(matMul, newMatMul);

    std::shared_ptr<Node> parent = newMatMul;

    // dequantization with subtract on activations & constant weights
    if (dequantization1.subtract) {
        auto broadcastShape = NetworkHelper::isScalarLike(as_type_ptr<opset1::Constant>(dequantization1.subtractConstant)) ?
            Shape(dequantization1.subtract->get_shape().size(), 1) :
            dequantization1.subtractConstant->get_shape();
        const size_t lastIdx = matMul->get_transpose_a() ? broadcastShape.size() - 2 : broadcastShape.size() - 1;
        broadcastShape[lastIdx] = dequantization1.subtract->get_shape()[lastIdx];

        // broadcasted sub const to form [1, ..., 1, Y]
        const auto broadcastedConst = fold<opset1::Broadcast>(
            dequantization1.subtractConstant,
            opset1::Constant::create(ngraph::element::i32, { broadcastShape.size() }, broadcastShape));

        // multiply by weights: [1, ..., 1, Y] x [Y, Z] => [1, ..., 1, Z]
        const auto newSubConst = NetworkHelper::toScalarIfPossible(fold<opset1::MatMul>(
            broadcastedConst,
            foldConvert(newMatMul->get_input_node_shared_ptr(1), newMatMul->get_element_type()),
            newMatMul->get_transpose_a(),
            newMatMul->get_transpose_b()));

        const auto newSubtract = std::make_shared<DequantizationSubtract>(newMatMul, newSubConst);
        newSubtract->set_friendly_name(newMatMul->get_friendly_name() + "/DequantizationSubtract");
        copy_runtime_info({ newSubtract, matMul }, newSubtract);

        parent = newSubtract;
    }

    auto transpose = [](const std::shared_ptr<Node>& node) -> std::shared_ptr<Node> {
        const Shape outputShape = node->get_output_shape(0);
        if (outputShape.size() < 2ul) {
            return node;
        }

        std::vector<uint32_t> transposeConstant(outputShape.size());
        std::iota(transposeConstant.begin(), transposeConstant.end(), 0);
        std::swap(*(transposeConstant.end() - 1), *(transposeConstant.end() - 2));

        auto order = opset1::Constant::create(element::u32, Shape{ transposeConstant.size() }, transposeConstant);
        std::shared_ptr<Node> transposedConstant = fold<opset1::Transpose>(node, order);
        return transposedConstant;
    };

    const auto mulConst1 = matMul->get_transpose_a() ? transpose(dequantization1.multiplyConstant) : dequantization1.multiplyConstant;
    auto mulConst2 = matMul->get_transpose_b() ? transpose(dequantization2.multiplyConstant) : dequantization2.multiplyConstant;

    if (NetworkHelper::isScalarLike(as_type_ptr<opset1::Constant>(mulConst2))) {
        mulConst2 = NetworkHelper::toScalar(as_type_ptr<opset1::Constant>(mulConst2));
    } else {
        auto constShape = mulConst2->get_shape();
        auto inputShape = matMul->get_input_shape(0);

        // unsqueeze from the left side to make both shapes of the same rank
        if (constShape.size() < inputShape.size()) {
            Shape unsqueezeConstantShape(inputShape.size() - constShape.size());
            std::iota(unsqueezeConstantShape.begin(), unsqueezeConstantShape.end(), 0ul);

            mulConst2 = fold<opset1::Unsqueeze>(
                mulConst2,
                op::Constant::create(element::i32, Shape{ unsqueezeConstantShape.size() }, unsqueezeConstantShape));
        }
    }

    const auto newMulConst = NetworkHelper::toScalarIfPossible(fold<ngraph::opset1::Multiply>(
            mulConst1,
            foldConvert(mulConst2, element::f32)));

    const auto newMultiply = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
        std::vector<element::Type>{ deqPrecision, deqPrecision },
        std::vector<element::Type>{ dequantization1.multiply->get_output_element_type(0) },
        ngraph::op::TemporaryReplaceOutputType(parent, deqPrecision).get(),
        ngraph::op::TemporaryReplaceOutputType(newMulConst, deqPrecision).get());

    newMultiply->set_friendly_name(newMatMul->get_friendly_name() + "/DequantizationMultiply");

    replace_node(matMul, newMultiply);
    copy_runtime_info({ newMultiply, matMul }, newMultiply);

    updateOutput(context, newMultiply, matMul);

    return true;
}

void MatMulTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Multiply>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>() }));
}

bool MatMulTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

bool MatMulTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformedSpatialDimension(context, layer)) {
        return false;
    }

    std::shared_ptr<opset1::MatMul> matMul = as_type_ptr<opset1::MatMul>(layer);
    if (matMul == nullptr) {
        return false;
    }

    const auto dequantization1 = NetworkHelper::getDequantization(layer, 0);
    if (!dequantization1.empty()) {
        if (updatePrecisions && !dequantization1.isLowPrecision()) {
            return false;
        }

        if (!NetworkHelper::isScalarLike(dequantization1.multiplyConstant)) {
            const auto constantShape = dequantization1.multiplyConstant->get_shape();
            const auto mulShape = dequantization1.multiply->get_shape();
            const size_t columnsIdx = matMul->get_transpose_a() ? mulShape.size() - 2ul : mulShape.size() - 1ul;

            // dequantization scales by columns in tensor A can't be propagate
            if ((constantShape.size() == mulShape.size()) && (constantShape[columnsIdx] != 1)) {
                return false;
            }
        }

        if (!NetworkHelper::checkZeroPoint(dequantization1.subtract)) {
            return false;
        }
    }

    const auto dequantization2 = NetworkHelper::getDequantization(layer, 1);
    if (!dequantization2.empty()) {
        if ((updatePrecisions && !dequantization2.isLowPrecision())) {
            return false;
        }

        if (dequantization2.subtract) {
            const auto roundedConst = NetworkHelper::round(dequantization2.subtractConstant, dequantization2.data.get_element_type());
            if (!NetworkHelper::isZeroConst(roundedConst)) {
                return false;
            }
        }

        if (!NetworkHelper::isScalarLike(dequantization2.multiplyConstant)) {
            const auto constantShape = dequantization2.multiplyConstant->get_shape();
            const auto mulShape = dequantization2.multiply->get_shape();
            const size_t rowsIdx = matMul->get_transpose_b() ? mulShape.size() - 1ul : mulShape.size() - 2ul;

            // dequantization scales by rows in tensor B can't be propagate
            if ((constantShape.size() == mulShape.size()) && (constantShape[rowsIdx] != 1)) {
                return false;
            }
        }
    }

    const auto fakeQuantize = as_type_ptr<opset1::FakeQuantize>(layer->get_input_node_shared_ptr(1));
    if (fakeQuantize) {
        if (!QuantizationDetails::outputLayoutIsSupported(fakeQuantize)) {
            return false;
        }

        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
        const DataPrecision dataPrecision = getDataPrecision(fakeQuantize, quantizationDetails, true);
        if (dataPrecision.hasZeroPoint) {
            return false;
        }

        const auto outLowShape = fakeQuantize->get_input_node_shared_ptr(3)->get_shape();
        const auto outHighShape = fakeQuantize->get_input_node_shared_ptr(4)->get_shape();
        const auto fakeQuantizeShape = fakeQuantize->get_shape();
        const size_t rowsIdx = matMul->get_transpose_b() ? fakeQuantizeShape.size() - 1 : fakeQuantizeShape.size() - 2;

        // dequantization scales by rows in tensor B can't be propagate
        if (((outLowShape.size() == fakeQuantizeShape.size()) && (outLowShape[rowsIdx] != 1)) ||
            ((outHighShape.size() == fakeQuantizeShape.size()) && (outHighShape[rowsIdx] != 1))) {
            return false;
        }
    }

    if ((!NetworkHelper::isConstantPath(layer->get_input_node_shared_ptr(1))) && (dequantization1.subtract)) {
        return false;
    }

    return true;
}
