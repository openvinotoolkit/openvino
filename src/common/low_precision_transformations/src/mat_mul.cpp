// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/mat_mul.hpp"

#include <numeric>
#include <memory>
#include <string>
#include <vector>

#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "openvino/util/log.hpp"
#include "itt.hpp"

using namespace ov;
using namespace ov::pass;
using namespace ov::pass::low_precision;

MatMulTransformation::MatMulTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(MatMulTransformation);
    auto mul1 = pattern::wrap_type<ov::opset1::Multiply>();
    auto mul2 = pattern::wrap_type<ov::opset1::Multiply>();
    auto fq2 = pattern::wrap_type<ov::opset1::FakeQuantize>();
    auto matcher = pattern::wrap_type<ov::opset1::MatMul>({ mul1, std::make_shared<pass::pattern::op::Or>(OutputVector{ mul2, fq2 })});

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool MatMulTransformation::transform(ov::pass::pattern::Matcher &m) {
    std::shared_ptr<ov::opset1::MatMul> matMul = ov::as_type_ptr<ov::opset1::MatMul>(m.get_match_root());
    if ((matMul == nullptr) || !canBeTransformed(matMul)) {
        return false;
    }

    matMul = ov::as_type_ptr<ov::opset1::MatMul>(NetworkHelper::separateInStandaloneBranch(matMul, defaultPrecisions));
    const auto dequantization1 = NetworkHelper::getDequantization(matMul, defaultPrecisions, 0);
    auto dequantization2 = NetworkHelper::getDequantization(matMul, defaultPrecisions, 1);

    if (dequantization2.empty()) {
        const std::shared_ptr<ov::opset1::FakeQuantize> fakeQuantize =
            ov::as_type_ptr<ov::opset1::FakeQuantize>(dequantization2.data.get_node_shared_ptr());
        if (fakeQuantize != nullptr) {
            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);

            const auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(fakeQuantize);
            const auto precisions = precisionsAttribute.empty() ?
                defaultPrecisions :
                precisionsAttribute.as<PrecisionsAttribute>().value();
            const DataPrecision dataPrecision = getDataPrecision(fakeQuantize, quantizationDetails, precisions);
            if (dataPrecision.empty()) {
                return false;
            }

            auto tuple = NetworkHelper::decomposeFakeQuantize(
                fakeQuantize,
                dataPrecision.precision,
                dataPrecision.min,
                dataPrecision.max,
                dataPrecision.hasZeroPoint,
                updatePrecisions);

            dequantization2 = NetworkHelper::getDequantization(matMul, defaultPrecisions, 1);
        }
    }

    if (dequantization2.subtract != nullptr) {
        NetworkHelper::optimizeSubtract(dequantization2.subtract);
        dequantization2 = NetworkHelper::getDequantization(matMul, defaultPrecisions, 1);
    }

    const std::shared_ptr<ov::opset1::MatMul> newMatMul = std::make_shared<ov::op::TypeRelaxed<ov::opset1::MatMul>>(
        std::vector<element::Type>({ deqPrecision, deqPrecision }), std::vector<element::Type>({ deqPrecision }),
        ov::op::TemporaryReplaceOutputType(dequantization1.data, deqPrecision).get(),
        ov::op::TemporaryReplaceOutputType(dequantization2.data, deqPrecision).get(),
        matMul->get_transpose_a(),
        matMul->get_transpose_b());
    NetworkHelper::copyInfo(matMul, newMatMul);

    std::shared_ptr<Node> parent = newMatMul;

    // dequantization with subtract on activations & constant weights
    if (dequantization1.subtract) {
        auto broadcastShape = NetworkHelper::isScalarLike(ov::as_type_ptr<ov::opset1::Constant>(dequantization1.subtractConstant)) ?
            Shape(dequantization1.subtract->get_output_partial_shape(0).rank().get_length(), 1) :
            dequantization1.subtractConstant->get_shape();

        const auto weightsPShape = newMatMul->get_input_partial_shape(1);
        assert(weightsPShape.is_static());
        const auto weightsShape = weightsPShape.to_shape();

        const size_t firstWeightsIdx = matMul->get_transpose_b() ? weightsShape.size() - 1ul : weightsShape.size() - 2ul;
        const size_t lastDataIdx = matMul->get_transpose_a() ? broadcastShape.size() - 2 : broadcastShape.size() - 1;
        broadcastShape[lastDataIdx] = weightsShape[firstWeightsIdx];

        // broadcasted sub const to form [1, ..., 1, Y]
        const auto broadcastedConst = fold<ov::opset1::Broadcast>(
            dequantization1.subtractConstant,
            ov::opset1::Constant::create(ov::element::i32, { broadcastShape.size() }, broadcastShape));

        // multiply by weights: [1, ..., 1, Y] x [Y, Z] => [1, ..., 1, Z]
        const auto newSubConst = NetworkHelper::toScalarIfPossible(fold<ov::opset1::MatMul>(
            foldConvert(broadcastedConst, newMatMul->get_element_type()),
            foldConvert(newMatMul->input_value(1), newMatMul->get_element_type()),
            newMatMul->get_transpose_a(),
            newMatMul->get_transpose_b()));

        const auto newSubtract = std::make_shared<ov::opset1::Subtract>(newMatMul, newSubConst);
        newSubtract->set_friendly_name(newMatMul->get_friendly_name() + "/DequantizationSubtract");
        copy_runtime_info({ newSubtract, matMul }, newSubtract);

        parent = newSubtract;
    }

    auto transpose = [](const std::shared_ptr<ov::opset1::Constant>& node) -> std::shared_ptr<Node> {
        const Shape outputShape = node->get_shape();
        if (outputShape.size() < 2ul) {
            return node;
        }

        std::vector<uint32_t> transposeConstant(outputShape.size());
        std::iota(transposeConstant.begin(), transposeConstant.end(), 0);
        std::swap(*(transposeConstant.end() - 1), *(transposeConstant.end() - 2));

        auto order = ov::opset1::Constant::create(element::u32, Shape{ transposeConstant.size() }, transposeConstant);
        std::shared_ptr<Node> transposedConstant = fold<ov::opset1::Transpose>(node, order);
        return transposedConstant;
    };

    const auto mulConst1 = matMul->get_transpose_a() ? transpose(dequantization1.multiplyConstant) : dequantization1.multiplyConstant;
    auto mulConst2 = matMul->get_transpose_b() ? transpose(dequantization2.multiplyConstant) : dequantization2.multiplyConstant;

    if (NetworkHelper::isScalarLike(ov::as_type_ptr<ov::opset1::Constant>(mulConst2))) {
        mulConst2 = NetworkHelper::toScalar(ov::as_type_ptr<ov::opset1::Constant>(mulConst2));
    } else {
        const auto constShape = mulConst2->get_shape();
        const size_t inputRank = matMul->get_input_partial_shape(0).rank().get_length();

        // unsqueeze from the left side to make both shapes of the same rank
        if (constShape.size() < inputRank) {
            Shape unsqueezeConstantShape(inputRank - constShape.size());
            std::iota(unsqueezeConstantShape.begin(), unsqueezeConstantShape.end(), 0ul);

            mulConst2 = fold<ov::opset1::Unsqueeze>(
                mulConst2,
                ov::op::v0::Constant::create(element::i32, Shape{ unsqueezeConstantShape.size() }, unsqueezeConstantShape));
        }
    }

    const auto newMulConst = NetworkHelper::toScalarIfPossible(fold<ov::opset1::Multiply>(
            foldConvert(mulConst1, element::f32),
            foldConvert(mulConst2, element::f32)));

    const auto newMultiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
        std::vector<element::Type>{ deqPrecision, deqPrecision },
        std::vector<element::Type>{ dequantization1.multiply->get_output_element_type(0) },
        ov::op::TemporaryReplaceOutputType(parent, deqPrecision).get(),
        ov::op::TemporaryReplaceOutputType(newMulConst, deqPrecision).get());

    newMultiply->set_friendly_name(newMatMul->get_friendly_name() + "/DequantizationMultiply");

    NetworkHelper::insertDequantizationAfter(matMul, newMultiply, newMatMul);
    copy_runtime_info({ newMultiply, matMul }, newMultiply);

    updateOutput(newMultiply, newMatMul);

    OPENVINO_DEBUG("LPT: done: ", newMatMul);
    return true;
}

bool MatMulTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

bool MatMulTransformation::canBeTransformed(const std::shared_ptr<Node>& layer) const {
    if (!LayerTransformation::canBeTransformedSpatialDimension(layer)) {
        return false;
    }

    std::shared_ptr<ov::opset1::MatMul> matMul = ov::as_type_ptr<ov::opset1::MatMul>(layer);
    if (matMul == nullptr) {
        OPENVINO_DEBUG("LPT: early exit: not MatMul");
        return false;
    }

    const auto dequantization1 = NetworkHelper::getDequantization(layer, defaultPrecisions, 0);
    if (!dequantization1.empty()) {
        if (updatePrecisions && !dequantization1.isLowPrecision()) {
            OPENVINO_DEBUG("LPT: early exit: dequantization before is not in low precision");
            return false;
        }

        if (!NetworkHelper::isScalarLike(dequantization1.multiplyConstant)) {
            const auto constantShape = dequantization1.multiplyConstant->get_shape();
            const auto mulShape = dequantization1.multiply->get_output_partial_shape(0);
            const size_t rank = mulShape.rank().get_length();

            const size_t columnsIdx = matMul->get_transpose_a() ? rank - 2 : rank - 1;

            // dequantization scales by columns in tensor A can't be propagate
            if ((constantShape.size() == rank) && (constantShape[columnsIdx] != 1)) {
                return false;
            }
        }
    } else {
        return false;
    }

    // WA: LPT don't expect Convert on dequantization scale: the convert is usually folded at the previous transformation pipeline steps.
    // However, such subgraphs may arise on Matmul weights since decompression handling logic keeps these subgraphs unchanged
    // TODO: remove this logic when
    //       1. CompressedMatmul is implemented
    //       2. Or convert on scales is supported across the whole LPT pipeline
    if (auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(layer->get_input_node_shared_ptr(1))) {
        if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(mul->get_input_node_shared_ptr(1))) {
            if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0))) {
                auto new_constant = foldConvert(constant, convert->get_destination_type());
                ov::replace_node_update_name(convert, new_constant);
            }
        }
    }
    const auto dequantization2 = NetworkHelper::getDequantization(layer, defaultPrecisions, 1);
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
            const auto mulShape = dequantization2.multiply->get_output_partial_shape(0);
            const size_t rank = mulShape.rank().get_length();

            const size_t rowsIdx = matMul->get_transpose_b() ? rank - 1ul : rank - 2ul;

            // dequantization scales by rows in tensor B can't be propagate
            if ((constantShape.size() == rank) && (constantShape[rowsIdx] != 1)) {
                return false;
            }
        }
    }

    const auto fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(layer->get_input_node_shared_ptr(1));
    if (fakeQuantize) {
        if (!QuantizationDetails::outputLayoutIsSupported(fakeQuantize)) {
            return false;
        }

        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);

        const auto precisionsAttribute = getAttribute<PrecisionsAttribute>(matMul->input(1));
        const auto precisions = precisionsAttribute.empty() ?
            defaultPrecisions :
            precisionsAttribute.as<PrecisionsAttribute>().value();

        const DataPrecision dataPrecision = getDataPrecision(fakeQuantize, quantizationDetails, precisions);
        if (dataPrecision.hasZeroPoint || dataPrecision.empty()) {
            return false;
        }

        const auto outLowShape = fakeQuantize->get_input_node_shared_ptr(3)->get_shape();
        const auto outHighShape = fakeQuantize->get_input_node_shared_ptr(4)->get_shape();
        const auto fakeQuantizeShape = fakeQuantize->get_output_partial_shape(0);
        const size_t rank = fakeQuantizeShape.rank().get_length();

        const size_t rowsIdx = matMul->get_transpose_b() ? rank - 1 : rank - 2;

        // dequantization scales by rows in tensor B can't be propagate
        if (((outLowShape.size() == rank) && (outLowShape[rowsIdx] != 1)) ||
            ((outHighShape.size() == rank) && (outHighShape[rowsIdx] != 1))) {
            return false;
        }
    }

    if (!fakeQuantize && dequantization2.empty()) {
        return false;
    }

    if ((!NetworkHelper::isConstantPath(layer->get_input_node_shared_ptr(1))) && (dequantization1.subtract)) {
        return false;
    }

    return true;
}
