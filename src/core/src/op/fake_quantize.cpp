// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/fake_quantize.hpp"

#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/reference/fake_quantize.hpp"

using namespace std;
using namespace ngraph;

op::FakeQuantize::FakeQuantize() : Op(), m_levels() {}

op::FakeQuantize::FakeQuantize(const Output<Node>& data,
                               const Output<Node>& input_low,
                               const Output<Node>& input_high,
                               const Output<Node>& output_low,
                               const Output<Node>& output_high,
                               size_t levels,
                               const AutoBroadcastSpec& auto_broadcast)
    : Op({data, input_low, input_high, output_low, output_high}),
      m_levels(levels),
      m_auto_broadcast(auto_broadcast) {
    constructor_validate_and_infer_types();
}

void op::FakeQuantize::validate_and_infer_types() {
    OV_OP_SCOPE(v0_FakeQuantize_validate_and_infer_types);
    ov::PartialShape data_pshape = get_input_partial_shape(0);

    for (auto i = 1; i <= 4; i++) {
        if (m_auto_broadcast.m_type == op::AutoBroadcastType::NONE) {
            NODE_VALIDATION_CHECK(this,
                                  ov::PartialShape::merge_into(data_pshape, get_input_partial_shape(i)),
                                  "Argument shapes are inconsistent.");
        } else if (m_auto_broadcast.m_type == op::AutoBroadcastType::NUMPY ||
                   m_auto_broadcast.m_type == op::AutoBroadcastType::PDPD) {
            NODE_VALIDATION_CHECK(
                this,
                ov::PartialShape::broadcast_merge_into(data_pshape, get_input_partial_shape(i), m_auto_broadcast),
                "Argument shapes are inconsistent.");
        } else {
            NODE_VALIDATION_CHECK(this, false, "Unsupported auto broadcast specification");
        }
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool ngraph::op::v0::FakeQuantize::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_FakeQuantize_visit_attributes);
    visitor.on_attribute("levels", m_levels);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

shared_ptr<Node> op::FakeQuantize::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_FakeQuantize_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<FakeQuantize>(new_args.at(0),  // X
                                     new_args.at(1),  // input_low
                                     new_args.at(2),  // input_high
                                     new_args.at(3),  // output_low
                                     new_args.at(4),  // output_high
                                     m_levels,
                                     m_auto_broadcast);
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace fakequantizeop {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& arg2,
              const HostTensorPtr& arg3,
              const HostTensorPtr& arg4,
              const HostTensorPtr& out,
              const ngraph::op::FakeQuantize* parent) {
    OV_OP_SCOPE(v0_FakeQuantize_evaluate);
    using T = typename element_type_traits<ET>::value_type;
    out->set_shape(arg0->get_shape());
    out->set_element_type(arg0->get_element_type());
    ov::reference::fake_quantize<T>(arg0->get_data_ptr<const T>(),
                                    arg1->get_data_ptr<const T>(),
                                    arg2->get_data_ptr<const T>(),
                                    arg3->get_data_ptr<const T>(),
                                    arg4->get_data_ptr<const T>(),
                                    out->get_data_ptr<T>(),
                                    arg0->get_shape(),
                                    arg1->get_shape(),
                                    arg2->get_shape(),
                                    arg3->get_shape(),
                                    arg4->get_shape(),
                                    parent->get_levels(),
                                    parent->get_auto_broadcast());
    return true;
}

bool evaluate_fakequantize(const HostTensorPtr& arg0,
                           const HostTensorPtr& arg1,
                           const HostTensorPtr& arg2,
                           const HostTensorPtr& arg3,
                           const HostTensorPtr& arg4,
                           const HostTensorPtr& out,
                           const ngraph::op::FakeQuantize* parent) {
    bool rc = true;
    switch (arg0->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_fakequantize, i32, arg0, arg1, arg2, arg3, arg4, out, parent);
        OPENVINO_TYPE_CASE(evaluate_fakequantize, i64, arg0, arg1, arg2, arg3, arg4, out, parent);
        OPENVINO_TYPE_CASE(evaluate_fakequantize, u32, arg0, arg1, arg2, arg3, arg4, out, parent);
        OPENVINO_TYPE_CASE(evaluate_fakequantize, u64, arg0, arg1, arg2, arg3, arg4, out, parent);
        OPENVINO_TYPE_CASE(evaluate_fakequantize, f16, arg0, arg1, arg2, arg3, arg4, out, parent);
        OPENVINO_TYPE_CASE(evaluate_fakequantize, f32, arg0, arg1, arg2, arg3, arg4, out, parent);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace fakequantizeop

bool ngraph::op::FakeQuantize::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_FakeQuantize_evaluate);
    return fakequantizeop::evaluate_fakequantize(inputs[0],
                                                 inputs[1],
                                                 inputs[2],
                                                 inputs[3],
                                                 inputs[4],
                                                 outputs[0],
                                                 this);
}

bool ngraph::op::FakeQuantize::has_evaluate() const {
    OV_OP_SCOPE(v0_FakeQuantize_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
