// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/normalize_l2.hpp"

#include <algorithm>
#include <string>
#include <memory>
#include <vector>

#include "ngraph_ops/multiply_add.hpp"
#include "ngraph_ops/normalize_ie.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

//template<ngraph::element::Type_t ET>
//template<typename T>
//void updateScale<T>(const ngraph::op::Constant* scales) {
//    const void* data = scalesConst->get_data_ptr();
//}

template<typename T>
void setValue(const size_t i, const T* source, T* target) {
    target[i] = std::signbit(source[i]) ? -1.f : 1.f;
}

template<typename T>
std::shared_ptr<ngraph::op::Constant> createNewScalesConst(const ngraph::op::Constant& originalConst) {
    const ngraph::Shape outputShape = originalConst.get_output_shape(0);
    const size_t size = ngraph::shape_size(outputShape);

    const T* source = originalConst.get_data_ptr<T>();
    std::vector<T> newData(size);
    for (size_t i = 0; i < size; ++i) {
        newData[i] = std::signbit(source[i]) ? -1 : 1;
    }

    const ngraph::element::Type type = originalConst.get_output_element_type(0);
    return ngraph::op::Constant::create(type, outputShape, newData);
}

bool NormalizeL2Transformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    const ngraph::op::NormalizeL2* normalizeIe = dynamic_cast<const ngraph::op::NormalizeL2*>(operation.get());
    if (normalizeIe == nullptr) {
        return false;
    }

    const std::shared_ptr<Node> dequantization = operation->input_value(0).get_node_shared_ptr();
    const std::shared_ptr<Node> scales = dequantization->input_value(1).get_node_shared_ptr();
    const ngraph::op::Constant* scalesConst = dynamic_cast<const ngraph::op::Constant*>(scales.get());
    if (scalesConst == nullptr) {
        return false;
    }

    //if (normalizeIe->get_across_spatial() && (!scalesConst->get_all_data_elements_bitwise_identical())) {
    //    return false;
    //}

    return true;
}

void NormalizeL2Transformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<ngraph::op::NormalizeL2>({
            make_op_label<ngraph::op::MultiplyAdd>(),
            make_op_label<ngraph::op::Constant>()
            }));

    //addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<ngraph::op::NormalizeIE>({
    //        make_op_label<ngraph::op::MultiplyAdd>(),
    //        make_op_label<ngraph::op::Constant>()
    //    }));
}

void NormalizeL2Transformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::Node> operation = m.get_match_root();
    if (!canBeTransformed(context, operation)) {
        return;
    }

    const std::shared_ptr<Node> dequantization = operation->input_value(0).get_node_shared_ptr();
    const std::shared_ptr<Node> scales = dequantization->input_value(1).get_node_shared_ptr();
    const ngraph::op::Constant* scalesConst = dynamic_cast<const ngraph::op::Constant*>(scales.get());

    //const ngraph::Shape outputShape = scalesConst->get_output_shape(0);
    //const size_t size = ngraph::shape_size(outputShape);
    //const void* data = static_cast<ngraph::element_type_traits<ngraph::element::f32>>(scalesConst->get_data_ptr());
    //using T = typename element_type_traits<ET>::value_type;
    //runtime::reference::relu<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    //updateValues<int>(scalesConst->get_data_ptr(), 4ul);
    //updateValues<ngraph::element_type_traits<ngraph::element::f32>::value_type>(scalesConst->get_data_ptr(), 4ul);


    std::shared_ptr<ngraph::op::Constant> newScalesConst;
    const ngraph::element::Type type = scalesConst->get_output_element_type(0);
    switch (type) {
        case ngraph::element::Type_t::f16: {
            newScalesConst = createNewScalesConst<ngraph::element_type_traits<ngraph::element::Type_t::f32>::value_type>(*scalesConst);
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

    // TODO: check if shift is not zero then exit

    //moveDequantization(operation, dequantization, newScalesConst);
    moveDequantization(operation, dequantization);
}

//bool NormalizeTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
//    return false;
//}
