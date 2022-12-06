// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/prior_box_clustered.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/prior_box_clustered.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v0::PriorBoxClustered);

ov::op::v0::PriorBoxClustered::PriorBoxClustered(const Output<Node>& layer_shape,
                                                 const Output<Node>& image_shape,
                                                 const Attributes& attrs)
    : Op({layer_shape, image_shape}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void ov::op::v0::PriorBoxClustered::validate_and_infer_types() {
    OV_OP_SCOPE(v0_PriorBoxClustered_validate_and_infer_types);
    // shape node should have integer data type. For now we only allow i64
    auto layer_shape_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          layer_shape_et.is_integral_number(),
                          "layer shape input must be an integral number, but is: ",
                          layer_shape_et);

    auto image_shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          image_shape_et.is_integral_number(),
                          "image shape input must be an integral number, but is: ",
                          image_shape_et);

    auto layer_shape_rank = get_input_partial_shape(0).rank();
    auto image_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          layer_shape_rank.compatible(image_shape_rank),
                          "layer shape input rank ",
                          layer_shape_rank,
                          " must match image shape input rank ",
                          image_shape_rank);

    NODE_VALIDATION_CHECK(this,
                          m_attrs.widths.size() == m_attrs.heights.size(),
                          "Size of heights vector: ",
                          m_attrs.heights.size(),
                          " doesn't match size of widths vector: ",
                          m_attrs.widths.size());

    set_input_is_relevant_to_shape(0);

    if (auto const_shape = get_constant_from_source(input_value(0).get_node_shared_ptr())) {
        NODE_VALIDATION_CHECK(this,
                              shape_size(const_shape->get_shape()) == 2,
                              "Layer shape must have rank 2",
                              const_shape->get_shape());

        auto layer_shape = const_shape->get_shape_val();
        // {Prior boxes, variances-adjusted prior boxes}
        const auto num_priors = m_attrs.widths.size();
        set_output_type(0, element::f32, ov::Shape{2, 4 * layer_shape[0] * layer_shape[1] * num_priors});
    } else {
        set_output_type(0, element::f32, ov::PartialShape{2, Dimension::dynamic()});
    }
}

shared_ptr<Node> ov::op::v0::PriorBoxClustered::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_PriorBoxClustered_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<PriorBoxClustered>(new_args.at(0), new_args.at(1), m_attrs);
}

bool ov::op::v0::PriorBoxClustered::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_PriorBoxClustered_visit_attributes);

    visitor.on_attribute("step", m_attrs.step);
    visitor.on_attribute("step_w", m_attrs.step_widths);
    visitor.on_attribute("step_h", m_attrs.step_heights);
    visitor.on_attribute("width", m_attrs.widths);
    visitor.on_attribute("height", m_attrs.heights);
    visitor.on_attribute("clip", m_attrs.clip);
    visitor.on_attribute("offset", m_attrs.offset);
    visitor.on_attribute("variance", m_attrs.variances);
    return true;
}

namespace prior_box_clustered {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              ov::op::v0::PriorBoxClustered::Attributes attrs) {
    runtime::reference::prior_box_clustered(arg0->get_data_ptr<ET>(),
                                            arg1->get_data_ptr<ET>(),
                                            out->get_data_ptr<float>(),
                                            out->get_shape(),
                                            attrs);
    return true;
}

bool evaluate_prior_box(const HostTensorPtr& arg0,
                        const HostTensorPtr& arg1,
                        const HostTensorPtr& out,
                        const ov::op::v0::PriorBoxClustered::Attributes& attrs) {
    bool rc = true;
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_prior_box, i8, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i16, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i32, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i64, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u8, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u16, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u32, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u64, arg0, arg1, out, attrs);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace prior_box_clustered

bool op::v0::PriorBoxClustered::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_PriorBoxClustered_evaluate);
    return prior_box_clustered::evaluate_prior_box(inputs[0], inputs[1], outputs[0], get_attrs());
}

bool op::v0::PriorBoxClustered::has_evaluate() const {
    OV_OP_SCOPE(v0_PriorBoxClustered_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
        return true;
    default:
        break;
    }
    return false;
}
