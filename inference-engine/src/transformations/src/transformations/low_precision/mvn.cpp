// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mvn.hpp"

#include <algorithm>
#include <string>
#include <memory>
#include <cmath>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

bool MVNTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    if (!canSubtractBeHandled(operation)) {
        return false;
    }

    auto mvn = as_type_ptr<op::MVN>(operation);

    const std::shared_ptr<Node> multiply = mvn->get_input_node_shared_ptr(0);
    auto scalesConst = as_type_ptr<ngraph::opset1::Constant>(multiply->get_input_node_shared_ptr(1));
    if (scalesConst == nullptr) {
        scalesConst = as_type_ptr<ngraph::opset1::Constant>(multiply->get_input_node_shared_ptr(0));
    }
    if (scalesConst == nullptr) {
        return false;
    }

    const bool acrossChannels = mvn->get_reduction_axes().count(1) > 0;
    const bool normalizeVariance = mvn->get_normalize_variance();

    if (!NetworkHelper::isScalarLike(scalesConst) && acrossChannels) {
        return false;
    }
    return true;
}

void MVNTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<ngraph::op::MVN>({ make_op_label<ngraph::opset1::Multiply>() }));
}

bool MVNTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> operation = m.get_match_root();
    if (!canBeTransformed(context, operation)) {
        return false;
    }

    auto mvn = as_type_ptr<op::MVN>(separateInStandaloneBranch(operation));

    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(mvn);
    auto scalesConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));
    if (scalesConst == nullptr) {
        scalesConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(0));
    }

    const bool acrossChannels = mvn->get_reduction_axes().count(1) > 0;
    const bool normalizeVariance = mvn->get_normalize_variance();

    auto newScalesConst = scalesConst;
    const auto type = scalesConst->get_output_element_type(0);
    if (normalizeVariance) {
        switch (type) {
            case ngraph::element::Type_t::f16: {
                newScalesConst = NetworkHelper::createSignConstant<ngraph::element_type_traits<ngraph::element::Type_t::f16>::value_type>(*scalesConst);
                break;
            }
            case ngraph::element::Type_t::f32: {
                newScalesConst = NetworkHelper::createSignConstant<ngraph::element_type_traits<ngraph::element::Type_t::f32>::value_type>(*scalesConst);
                break;
            }
            default: {
                THROW_TRANSFORMATION_EXCEPTION << "unexpected element type " << type;
            }
        }
    }

    auto newMVN = std::make_shared<op::TypeRelaxed<op::MVN>>(
        op::MVN(dequantization.subtract ?
                    dequantization.subtract :
                    dequantization.data,
                mvn->get_reduction_axes(),
                mvn->get_normalize_variance(),
                mvn->get_eps()),
        type);

    auto newMultiply = std::make_shared<DequantizationMultiply>(newMVN, newScalesConst);
    newMVN->set_friendly_name(mvn->get_friendly_name());

    replace_node(mvn, newMultiply);

    updateOutput(context, newMultiply, newMVN);
    return true;
}

bool MVNTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
