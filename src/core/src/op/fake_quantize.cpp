// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_quantize.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/fake_quantize.hpp"

namespace ov {
namespace op {
namespace fake_quantize {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg0,
                             const Tensor& arg1,
                             const Tensor& arg2,
                             const Tensor& arg3,
                             const Tensor& arg4,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const Shape& shape2,
                             const Shape& shape3,
                             const Shape& shape4,
                             const size_t levels,
                             const AutoBroadcastSpec& broadcast_spec) {
        reference::fake_quantize(arg0.data<const T>(),
                                 arg1.data<const T>(),
                                 arg2.data<const T>(),
                                 arg3.data<const T>(),
                                 arg4.data<const T>(),
                                 out.data<T>(),
                                 shape0,
                                 shape1,
                                 shape2,
                                 shape3,
                                 shape4,
                                 levels,
                                 broadcast_spec);
        return true;
    }
};
}  // namespace fake_quantize
namespace v0 {

FakeQuantize::FakeQuantize() : Op(), m_levels() {}

FakeQuantize::FakeQuantize(const Output<Node>& data,
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

void FakeQuantize::validate_and_infer_types() {
    OV_OP_SCOPE(v0_FakeQuantize_validate_and_infer_types);
    auto data_pshape = get_input_partial_shape(0);

    for (auto i = 1; i <= 4; i++) {
        if (m_auto_broadcast.m_type == op::AutoBroadcastType::NONE) {
            NODE_VALIDATION_CHECK(this,
                                  PartialShape::merge_into(data_pshape, get_input_partial_shape(i)),
                                  "Argument shapes are inconsistent.");
        } else if (m_auto_broadcast.m_type == op::AutoBroadcastType::NUMPY ||
                   m_auto_broadcast.m_type == op::AutoBroadcastType::PDPD) {
            NODE_VALIDATION_CHECK(
                this,
                PartialShape::broadcast_merge_into(data_pshape, get_input_partial_shape(i), m_auto_broadcast),
                "Argument shapes are inconsistent.");
        } else {
            NODE_VALIDATION_CHECK(this, false, "Unsupported auto broadcast specification");
        }
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool FakeQuantize::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_FakeQuantize_visit_attributes);
    visitor.on_attribute("levels", m_levels);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

std::shared_ptr<Node> FakeQuantize::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_FakeQuantize_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<FakeQuantize>(new_args.at(0),  // X
                                          new_args.at(1),  // input_low
                                          new_args.at(2),  // input_high
                                          new_args.at(3),  // output_low
                                          new_args.at(4),  // output_high
                                          m_levels,
                                          m_auto_broadcast);
}

bool FakeQuantize::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_FakeQuantize_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 5);

    const auto& shape0 = inputs[0].get_shape();
    outputs[0].set_shape(shape0);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_FakeQuantize_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      fake_quantize::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      inputs[2],
                                      inputs[3],
                                      inputs[4],
                                      outputs[0],
                                      shape0,
                                      inputs[1].get_shape(),
                                      inputs[2].get_shape(),
                                      inputs[3].get_shape(),
                                      inputs[4].get_shape(),
                                      get_levels(),
                                      get_auto_broadcast());
}

bool FakeQuantize::has_evaluate() const {
    OV_OP_SCOPE(v0_FakeQuantize_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
