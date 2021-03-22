// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fake_quantize.hpp"

#include <cmath>
#include <memory>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FakeQuantizeTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

bool FakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::FakeQuantize> layer = std::dynamic_pointer_cast<opset1::FakeQuantize>(m.get_match_root());
    if (!NetworkHelper::isQuantizeSupported(layer)) {
        return false;
    }

    std::shared_ptr<opset1::FakeQuantize> fakeQuantize = layer;
    do {
        layer = fakeQuantize;
        fakeQuantize = fuseElementwise(context, fakeQuantize);
    } while (fakeQuantize != nullptr);

    return true;
}

namespace fq {

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

}  // namespace fq

bool FakeQuantizeTransformation::checkElementwise(const std::shared_ptr<Node>& eltwise) {
    const auto eltwiseInputShape = eltwise->get_input_shape(0);
    const auto eltwiseOutputShape = eltwise->get_output_shape(0);
    if (eltwiseInputShape.size() != eltwiseOutputShape.size()) {
        return false;
    }
    for (size_t i = 0; i < eltwiseOutputShape.size(); ++i) {
        if (eltwiseInputShape[i] != eltwiseOutputShape[i]) {
            return false;
        }
    }

    std::shared_ptr<opset1::Constant> constant = fq::getConstant(eltwise);
    if (constant == nullptr) {
        return false;
    }

    Shape shape = constant->get_output_shape(0);
    if ((!shape.empty()) && (shape_size(shape) != 1ul)) {
        if ((eltwiseOutputShape.size() - shape.size()) > 1) {
            return false;
        }

        if ((eltwiseOutputShape.size() - shape.size()) == 1ul) {
            shape.insert(shape.begin(), 1ul);
        }

        for (size_t i = 2ul; i < shape.size(); ++i) {
            if (shape[i] != 1ul) {
                return false;
            }
        }
    }

    return fq::getData(eltwise) != nullptr;
}

std::shared_ptr<opset1::FakeQuantize> FakeQuantizeTransformation::fuseElementwise(
    TransformationContext& context,
    const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const {
    const std::shared_ptr<Node> eltwise = fakeQuantize->get_input_node_shared_ptr(0);

    std::shared_ptr<Node> inputLowConst_f32 = fold<opset1::Convert>(fakeQuantize->get_input_node_shared_ptr(1), deqPrecision);
    std::shared_ptr<Node> inputHighConst_f32 = fold<opset1::Convert>(fakeQuantize->get_input_node_shared_ptr(2), deqPrecision);
    std::shared_ptr<Node> outputLowConst_f32 = fold<opset1::Convert>(fakeQuantize->get_input_node_shared_ptr(3), deqPrecision);
    std::shared_ptr<Node> outputHighConst_f32 = fold<opset1::Convert>(fakeQuantize->get_input_node_shared_ptr(4), deqPrecision);


    std::shared_ptr<opset1::Constant> constant = fq::getConstant(eltwise);
    if (is_type<opset1::Multiply>(eltwise) && checkElementwise(eltwise)) {
        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            fold<opset1::Convert>(constant, deqPrecision);

        const auto valueVec = as_type_ptr<opset1::Constant>(value)->cast_vector<float>();

        // avoid division by zero
        if (std::any_of(valueVec.cbegin(), valueVec.cend(), [](const float value) { return (value == 0.f) || (std::abs(value) < 1.e-32); })) {
            auto inputLowConstValues = as_type_ptr<opset1::Constant>(inputLowConst_f32)->cast_vector<float>();
            auto inputHighConstValues = as_type_ptr<opset1::Constant>(inputHighConst_f32)->cast_vector<float>();
            auto outputLowConstValues =  as_type_ptr<opset1::Constant>(outputLowConst_f32)->cast_vector<float>();
            auto outputHighConstValues = as_type_ptr<opset1::Constant>(outputHighConst_f32)->cast_vector<float>();
            const size_t inputIntervalsConstRank = inputLowConstValues.size();
            const size_t outputIntervalsConstRank = outputLowConstValues.size();
            const size_t multiplyConstRank = valueVec.size();
            const size_t resultRank = std::max(inputIntervalsConstRank, multiplyConstRank);

            const size_t outRank = std::max(resultRank, outputIntervalsConstRank);
            const auto zeroMapping = NetworkHelper::fold_fake_quantize(std::make_shared<opset1::FakeQuantize>(
                std::make_shared<opset1::Constant>(element::f32, Shape{outRank}, std::vector<float>(outRank, 0)),
                fakeQuantize->get_input_node_shared_ptr(1),
                fakeQuantize->get_input_node_shared_ptr(2),
                fakeQuantize->get_input_node_shared_ptr(3),
                fakeQuantize->get_input_node_shared_ptr(4),
                fakeQuantize->get_levels() ));
            auto zeroMappingValues = as_type_ptr<opset1::Constant>(zeroMapping)->cast_vector<float>();

            const bool inputIntervalsConstBroadcasted = inputIntervalsConstRank < resultRank;
            if (inputIntervalsConstBroadcasted) {
                inputLowConstValues = std::vector<float>(resultRank, inputLowConstValues[0]);
                inputHighConstValues = std::vector<float>(resultRank, inputHighConstValues[0]);
            }
            const bool outputIntervalsConstBroadcasted = outputIntervalsConstRank < resultRank;
            if (outputIntervalsConstBroadcasted) {
                outputLowConstValues = std::vector<float>(resultRank, outputLowConstValues[0]);
                outputHighConstValues = std::vector<float>(resultRank, outputHighConstValues[0]);
            }
            const bool multiplyConstBroadcasted = multiplyConstRank < resultRank;
            for (size_t i = 0; i < resultRank; ++i) {
                const float denominator = valueVec[multiplyConstBroadcasted ? 0ul : i];
                if ((denominator == 0.f) || (std::abs(denominator) < 1.e-32)) {
                    outputLowConstValues[i] = zeroMappingValues[i];
                    outputHighConstValues[i] = zeroMappingValues[i];
                } else {
                    inputLowConstValues[i] /= denominator;
                    inputHighConstValues[i] /= denominator;
                }
            }

            const Shape resultShape = multiplyConstBroadcasted ? inputLowConst_f32->get_output_shape(0) : constant->get_output_shape(0);
            inputLowConst_f32 = std::make_shared<opset1::Constant>(element::f32, resultShape, inputLowConstValues);
            inputHighConst_f32 = std::make_shared<opset1::Constant>(element::f32, resultShape, inputHighConstValues);
            outputLowConst_f32 = std::make_shared<opset1::Constant>(element::f32, resultShape, outputLowConstValues);
            outputHighConst_f32 = std::make_shared<opset1::Constant>(element::f32, resultShape, outputHighConstValues);
        } else {
            inputLowConst_f32 = fold<opset1::Divide>(inputLowConst_f32, value);
            inputHighConst_f32 = fold<opset1::Divide>(inputHighConst_f32, value);
        }

        inputLowConst_f32 = fq::updateShape(inputLowConst_f32, fakeQuantize->get_output_shape(0));
        inputHighConst_f32 =  fq::updateShape(inputHighConst_f32, fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Divide>(eltwise) && checkElementwise(eltwise)) {
        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            fold<opset1::Convert>(constant, deqPrecision);

        const auto valueVec = as_type_ptr<opset1::Constant>(value)->cast_vector<float>();

        inputLowConst_f32 = fq::updateShape(fold<opset1::Multiply>(inputLowConst_f32, value), fakeQuantize->get_output_shape(0));
        inputHighConst_f32 = fq::updateShape(fold<opset1::Multiply>(inputHighConst_f32, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Subtract>(eltwise) && checkElementwise(eltwise)) {
        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            fold<opset1::Convert>(constant, deqPrecision);

        inputLowConst_f32 = fq::updateShape(fold<opset1::Add>(inputLowConst_f32, value), fakeQuantize->get_output_shape(0));
        inputHighConst_f32 = fq::updateShape(fold<opset1::Add>(inputHighConst_f32, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Add>(eltwise) && checkElementwise(eltwise)) {
        if (is_type<opset1::Convolution>(fq::getData(eltwise)) ||
            is_type<opset1::GroupConvolution>(fq::getData(eltwise))) {
            return nullptr;
        }

        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            fold<opset1::Convert>(constant, deqPrecision);

        inputLowConst_f32 = fq::updateShape(fold<opset1::Subtract>(inputLowConst_f32, value), fakeQuantize->get_output_shape(0));
        inputHighConst_f32 = fq::updateShape(fold<opset1::Subtract>(inputHighConst_f32, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Convert>(eltwise)) {
        // issue #40611
        if ((eltwise->input(0).get_element_type() == element::i32) &&
            ((eltwise->output(0).get_element_type() == element::f16) || (eltwise->output(0).get_element_type() == element::f32))) {
            return nullptr;
        }
    } else {
        return nullptr;
    }

    // inverted input intervals transfer to output invert
    const auto is_inverted = as_type_ptr<opset1::Constant>(fold<opset1::Greater>(inputLowConst_f32, inputHighConst_f32))->cast_vector<bool>();
    if (std::any_of(is_inverted.begin(), is_inverted.end(), [](const bool value) { return value; })) {
        auto inputLowConstValues = as_type_ptr<opset1::Constant>(inputLowConst_f32)->cast_vector<float>();
        auto inputHighConstValues = as_type_ptr<opset1::Constant>(inputHighConst_f32)->cast_vector<float>();
        auto outputLowConstValues =  as_type_ptr<opset1::Constant>(outputLowConst_f32)->cast_vector<float>();
        auto outputHighConstValues = as_type_ptr<opset1::Constant>(outputHighConst_f32)->cast_vector<float>();
        const size_t inputIntervalsConstRank = inputLowConstValues.size();
        const size_t outputIntervalsConstRank = outputLowConstValues.size();
        const size_t result_rank = is_inverted.size();
        if (inputIntervalsConstRank != result_rank) {
            inputLowConstValues = std::vector<float>(result_rank, inputLowConstValues[0]);
            inputHighConstValues = std::vector<float>(result_rank, inputHighConstValues[0]);
        } else if (outputIntervalsConstRank != result_rank) {
            outputLowConstValues = std::vector<float>(result_rank, outputLowConstValues[0]);
            outputHighConstValues = std::vector<float>(result_rank, outputHighConstValues[0]);
        }
        for (size_t i = 0; i < result_rank; ++i) {
            if (is_inverted[i]) {
                std::swap(inputLowConstValues[i], inputHighConstValues[i]);
                std::swap(outputLowConstValues[i], outputHighConstValues[i]);
            }
        }
        const Shape resultShape = inputIntervalsConstRank != result_rank ? outputLowConst_f32->get_shape() : inputLowConst_f32->get_shape();
        inputLowConst_f32 = std::make_shared<opset1::Constant>(element::f32, resultShape, inputLowConstValues);
        inputHighConst_f32 = std::make_shared<opset1::Constant>(element::f32, resultShape, inputHighConstValues);
        outputLowConst_f32 = std::make_shared<opset1::Constant>(element::f32, resultShape, outputLowConstValues);
        outputHighConst_f32 = std::make_shared<opset1::Constant>(element::f32, resultShape, outputHighConstValues);
    }

    const auto data = fq::getData(eltwise);
    const size_t outputIdx = NetworkHelper::getParentOutputIndex(data, eltwise);
    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = as_type_ptr<opset1::FakeQuantize>(fakeQuantize->clone_with_new_inputs({
        data->output(outputIdx),
        inputLowConst_f32,
        inputHighConst_f32,
        outputLowConst_f32,
        outputHighConst_f32
    }));

    replace_node(fakeQuantize, newFakeQuantize);
    ngraph::copy_runtime_info({ fakeQuantize, eltwise }, newFakeQuantize);
    newFakeQuantize->set_friendly_name(fakeQuantize->get_friendly_name());
    NetworkHelper::cleanRunTimeInfo(newFakeQuantize);
    return newFakeQuantize;
}

bool FakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
