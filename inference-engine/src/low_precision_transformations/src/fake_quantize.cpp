// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fake_quantize.hpp"

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
    std::shared_ptr<opset1::Constant> constant = fq::getConstant(eltwise);
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

    return fq::getData(eltwise) != nullptr;
}

std::shared_ptr<opset1::FakeQuantize> FakeQuantizeTransformation::fuseElementwise(
    TransformationContext& context,
    const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const {
    const std::shared_ptr<Node> eltwise = fakeQuantize->get_input_node_shared_ptr(0);

    std::shared_ptr<Node> inputLowConst_f32 = fold<opset1::Convert>(fakeQuantize->get_input_node_shared_ptr(1), deqPrecision);
    std::shared_ptr<Node> inputHightConst_f32 = fold<opset1::Convert>(fakeQuantize->get_input_node_shared_ptr(2), deqPrecision);

    std::shared_ptr<opset1::Constant> constant = fq::getConstant(eltwise);
    if (is_type<opset1::Multiply>(eltwise) && checkElementwise(eltwise)) {
        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            fold<opset1::Convert>(constant, deqPrecision);

        const auto valueVec = as_type_ptr<opset1::Constant>(value)->cast_vector<float>();
        // TODO: temporary fix for GPU Plugin (inverted intervals)
        for (const float& val : valueVec) {
            if (val < 0) {
                return nullptr;
            }
        }

        inputLowConst_f32 = fq::updateShape(fold<opset1::Divide>(inputLowConst_f32, value), fakeQuantize->get_output_shape(0));
        inputHightConst_f32 = fq::updateShape(fold<opset1::Divide>(inputHightConst_f32, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Divide>(eltwise) && checkElementwise(eltwise)) {
        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            fold<opset1::Convert>(constant, deqPrecision);

        const auto valueVec = as_type_ptr<opset1::Constant>(value)->cast_vector<float>();
        // TODO: temporary fix for GPU Plugin (inverted intervals)
        for (const float& val : valueVec) {
            if (val < 0) {
                return nullptr;
            }
        }

        inputLowConst_f32 = fq::updateShape(fold<opset1::Multiply>(inputLowConst_f32, value), fakeQuantize->get_output_shape(0));
        inputHightConst_f32 = fq::updateShape(fold<opset1::Multiply>(inputHightConst_f32, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Subtract>(eltwise) && checkElementwise(eltwise)) {
        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            fold<opset1::Convert>(constant, deqPrecision);

        inputLowConst_f32 = fq::updateShape(fold<opset1::Add>(inputLowConst_f32, value), fakeQuantize->get_output_shape(0));
        inputHightConst_f32 = fq::updateShape(fold<opset1::Add>(inputHightConst_f32, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Add>(eltwise) && checkElementwise(eltwise)) {
        if (is_type<opset1::Convolution>(fq::getData(eltwise)) ||
            is_type<opset1::GroupConvolution>(fq::getData(eltwise))) {
            return nullptr;
        }

        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            fold<opset1::Convert>(constant, deqPrecision);

        inputLowConst_f32 = fq::updateShape(fold<opset1::Subtract>(inputLowConst_f32, value), fakeQuantize->get_output_shape(0));
        inputHightConst_f32 = fq::updateShape(fold<opset1::Subtract>(inputHightConst_f32, value), fakeQuantize->get_output_shape(0));
    } else if (is_type<opset1::Convert>(eltwise)) {
        // issue #40611
        if ((eltwise->input(0).get_element_type() == element::i32) &&
            ((eltwise->output(0).get_element_type() == element::f16) || (eltwise->output(0).get_element_type() == element::f32))) {
            return nullptr;
        }
    } else {
        return nullptr;
    }

    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = as_type_ptr<opset1::FakeQuantize>(fakeQuantize->clone_with_new_inputs({
        fq::getData(eltwise),
        inputLowConst_f32,
        inputHightConst_f32,
        fold<opset1::Convert>(fakeQuantize->input_value(3), deqPrecision),
        fold<opset1::Convert>(fakeQuantize->input_value(4), deqPrecision) }));

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
