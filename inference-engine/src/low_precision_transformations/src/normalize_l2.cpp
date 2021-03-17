// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/normalize_l2.hpp"

#include <string>
#include <memory>
#include <cmath>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/common/dequantization_op.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

namespace normalize_l2 {

template<typename T>
std::shared_ptr<ngraph::op::Constant> createNewScalesConst(const ngraph::op::Constant& originalConst) {
    std::vector<T> source = originalConst.cast_vector<T>();

    std::vector<T> newData(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        newData[i] = source[i] < 0 ? T{-1} : T{1};
    }

    const ngraph::element::Type type = originalConst.get_output_element_type(0);
    return ngraph::op::Constant::create(type, originalConst.get_shape(), newData);
}

} // namespace normalize_l2

bool NormalizeL2Transformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    if (NetworkHelper::getDequantization(operation).subtract != nullptr) {
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
    const std::vector<int64_t> axesAcrossSpatial = { 1 };
    const std::vector<int64_t> axesByChannels = { 1, 2, 3 };

    std::vector<int64_t> axesValues = axes->cast_vector<int64_t>();
    if (!(axesValues == axesAcrossSpatial || axesValues == axesByChannels)) {
        return false;
    }

    const ngraph::Shape outputShape = scalesConst->get_output_shape(0);
    const size_t size = ngraph::shape_size(outputShape);
    const size_t channels = operation->get_output_shape(0)[1];

    if (size != channels && size != 1) {
        return false;
    }

    if (!NetworkHelper::isScalarLike(scalesConst)) {
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

bool NormalizeL2Transformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> operation = m.get_match_root();
    if (!canBeTransformed(context, operation)) {
        return false;
    }

    auto normalize = as_type_ptr<opset1::NormalizeL2>(NetworkHelper::separateInStandaloneBranch(operation));

    const auto axes = as_type_ptr<opset1::Constant>(normalize->get_input_node_shared_ptr(1));
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(normalize);
    auto scalesConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));
    if (scalesConst == nullptr) {
        scalesConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(0));
    }

    std::shared_ptr<opset1::Constant> newScalesConst;
    const auto type = scalesConst->get_output_element_type(0);
    switch (type) {
        case ngraph::element::Type_t::f16: {
            newScalesConst = normalize_l2::createNewScalesConst<ngraph::element_type_traits<ngraph::element::Type_t::f16>::value_type>(*scalesConst);
            break;
        }
        case ngraph::element::Type_t::f32: {
            newScalesConst = normalize_l2::createNewScalesConst<ngraph::element_type_traits<ngraph::element::Type_t::f32>::value_type>(*scalesConst);
            break;
        }
        default: {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected element type " << type;
        }
    }

    auto newNormalize = std::make_shared<op::TypeRelaxed<opset1::NormalizeL2>>(
        std::vector<ngraph::element::Type>{ element::f32, element::f32 },
        std::vector<ngraph::element::Type>{deqPrecision},
        ngraph::op::TemporaryReplaceOutputType(dequantization.subtract == nullptr ? dequantization.data : dequantization.subtract, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(axes->clone_with_new_inputs({}), element::f32).get(),
        normalize->get_eps(),
        normalize->get_eps_mode());
    NetworkHelper::copyInfo(normalize, newNormalize);

    auto newMultiply = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
        std::vector<ngraph::element::Type>{ element::f32, element::f32 },
        std::vector<ngraph::element::Type>{normalize->get_output_element_type(0)},
        ngraph::op::TemporaryReplaceOutputType(newNormalize, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(newScalesConst, element::f32).get());

    replace_node(normalize, newMultiply);
    ngraph::copy_runtime_info({ normalize, newMultiply }, newMultiply);

    updateOutput(context, newMultiply, normalize);
    return true;
}

bool NormalizeL2Transformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
