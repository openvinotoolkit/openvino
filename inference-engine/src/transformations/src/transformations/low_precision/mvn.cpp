// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mvn.hpp"

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
        std::cout << newData[i] << std::endl;
    }

    const ngraph::element::Type type = originalConst.get_output_element_type(0);
    return ngraph::op::Constant::create(type, originalConst.get_shape(), newData);
}

bool MVNTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
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
    std::cout << "asd" << std::endl;
    return true;
}

void MVNTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<ngraph::op::MVN>({ make_op_label<ngraph::opset1::Multiply>() }));
}

void MVNTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> operation = m.get_match_root();
    if (!canBeTransformed(context, operation)) {
        removeConvertIfPossible(context, operation);
        return;
    }
    std::cout << "asd" << std::endl;

    auto mvn = as_type_ptr<op::MVN>(separateInStandaloneBranch(operation));

    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(mvn);
    auto scalesConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));
    if (scalesConst == nullptr) {
        scalesConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(0));
        std::cout << "asd" << std::endl;
    }

    const bool acrossChannels = mvn->get_reduction_axes().count(1) > 0;
    const bool normalizeVariance = mvn->get_normalize_variance();

    auto newScalesConst = scalesConst;
    const auto type = scalesConst->get_output_element_type(0);
    if (normalizeVariance) {
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
    }
    std::cout << "asd" << std::endl;

    auto newMVN = std::make_shared<op::TypeRelaxed<op::MVN>>(
        op::MVN(dequantization.subtract ?
                    dequantization.subtract :
                    dequantization.data,
                mvn->get_reduction_axes(),
                mvn->get_normalize_variance(),
                mvn->get_eps()),
        type);

    auto newMultiply = std::make_shared<opset1::Multiply>(newMVN, newScalesConst);
    //newMVN->set_friendly_name(mvn->get_friendly_name());

    replace_node(mvn, newMultiply);

    updateOutput(context, newMultiply, mvn);
}
