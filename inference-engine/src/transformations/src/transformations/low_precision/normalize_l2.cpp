// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/normalize_l2.hpp"

#include <algorithm>
#include <string>
#include <memory>
#include <cmath>
#include <vector>

#include "ngraph_ops/normalize_ie.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

template<typename T>
std::shared_ptr<ngraph::op::Constant> createNewScalesConst(const ngraph::op::Constant& originalConst) {
    std::vector<T> source = originalConst.cast_vector<T>();

    std::vector<T> newData(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        newData[i] = source[i] < 0 ? -1 : 1;
    }

    const ngraph::element::Type type = originalConst.get_output_element_type(0);
    return ngraph::op::Constant::create(type, originalConst.get_shape(), newData);
}

bool NormalizeL2Transformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    const std::shared_ptr<Node> multiply = operation->get_input_node_shared_ptr(0);
    auto scalesConst = as_type_ptr<ngraph::opset1::Constant>(multiply->get_input_node_shared_ptr(1));
    if (scalesConst == nullptr) {
        scalesConst = as_type_ptr<ngraph::opset1::Constant>(multiply->get_input_node_shared_ptr(0));
    }
    if (scalesConst == nullptr) {
        return false;
    }

    // TODO: Expand transformation for all cases of axes values
    const auto axes = as_type_ptr<opset1::Constant>(operation->get_input_node_shared_ptr(1));
    const std::vector<int64_t> axesAcrossChannels = { 1, 2, 3 };
    const std::vector<int64_t> axesByChannels = { 2, 3 };

    std::vector<int64_t> axesValues = axes->cast_vector<int64_t>();
    if (!(axesValues == axesAcrossChannels || axesValues == axesByChannels)) {
        return false;
    }

    const ngraph::Shape outputShape = scalesConst->get_output_shape(0);
    const size_t size = ngraph::shape_size(outputShape);
    const size_t channels = operation->get_output_shape(0)[1];

    if (size != channels || size != 1) {
        return false;
    }

    if (axesValues == axesAcrossChannels && size == channels && !NetworkHelper::isScalarLike(scalesConst)) {
        return false;
    }

    return true;
}

void NormalizeL2Transformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<ngraph::opset1::NormalizeL2>({
            make_op_label<ngraph::opset1::Multiply>(),
            make_op_label<ngraph::opset1::Constant>()
            }));
}

void NormalizeL2Transformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::Node> operation = m.get_match_root();
    if (!canBeTransformed(context, operation)) {
        return;
    }

    auto normalize = as_type_ptr<opset1::NormalizeL2>(operation);

    normalize = as_type_ptr<opset1::NormalizeL2>(separateInStandaloneBranch(normalize));

    const auto axes = as_type_ptr<opset1::Constant>(normalize->get_input_node_shared_ptr(1));
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(normalize);
    auto scalesConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));
    if (scalesConst == nullptr) {
        scalesConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(0));
    }

    std::shared_ptr<ngraph::opset1::Constant> newScalesConst;
    const auto type = scalesConst->get_output_element_type(0);
    switch (type) {
        case ngraph::element::Type_t::f16: {
            newScalesConst = createNewScalesConst<ngraph::element_type_traits<ngraph::element::Type_t::f16>::value_type>(*scalesConst);
            break;
        }
        case ngraph::element::Type_t::f32: {
            newScalesConst = createNewScalesConst<ngraph::element_type_traits<ngraph::element::Type_t::f32>::value_type>(*scalesConst);
            break;
        }
        default: {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected element type " << type;
        }
    }


    const auto newMultiply = std::make_shared<opset1::Multiply>(
        std::make_shared<opset1::NormalizeL2>(
            dequantization.subtract == nullptr ?
                dequantization.data : dequantization.subtract,
            axes,
            normalize->get_eps(),
            normalize->get_eps_mode()),
        newScalesConst);

    replace_node(normalize, newMultiply);

    updateOutput(context, newMultiply, normalize);
}
