// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include "element_visitor.hpp"
#include "histc_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/histc.hpp"

namespace ov {
namespace op {
namespace histc {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& input,
                             Tensor& output,
                             int64_t bins,
                             double min_val,
                             double max_val) {
        output.set_shape(Shape{static_cast<size_t>(bins)});
        reference::histc(input.data<const T>(), input.get_size(), bins, min_val, max_val, output.data<T>());
        return true;
    }
};

}  // namespace histc
namespace v17 {

Histc::Histc(const Output<Node>& data, int64_t bins, double min_val, double max_val)
    : Op({data}),
      m_bins(bins),
      m_min_val(min_val),
      m_max_val(max_val) {
    constructor_validate_and_infer_types();
}

bool Histc::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_Histc_visit_attributes);
    visitor.on_attribute("bins", m_bins);
    visitor.on_attribute("min_val", m_min_val);
    visitor.on_attribute("max_val", m_max_val);
    return true;
}

void Histc::validate_and_infer_types() {
    OV_OP_SCOPE(v17_Histc_validate_and_infer_types);

    const auto& data_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_et.is_dynamic() || data_et.is_real(),
                          "The 'data' input must have a floating-point element type. Got: ",
                          data_et);
    NODE_VALIDATION_CHECK(this, m_bins >= 0, "bins must be non-negative. Got: ", m_bins);
    NODE_VALIDATION_CHECK(this,
                          m_max_val >= m_min_val,
                          "max_val must be greater than or equal to min_val. Got min_val=",
                          m_min_val,
                          ", max_val=",
                          m_max_val);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, data_et, output_shapes[0]);

    const auto input_constant = ov::util::get_constant_from_source(input_value(0));
    if (!input_constant) {
        return;
    }

    TensorVector outputs{{data_et, {}}};
    if (!evaluate(outputs, TensorVector{input_constant->get_tensor_view()})) {
        return;
    }

    const auto& out = outputs[0];
    get_output_tensor(0).set_lower_value(out);
    get_output_tensor(0).set_upper_value(out);
}

std::shared_ptr<Node> Histc::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_Histc_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v17::Histc>(new_args.at(0), m_bins, m_min_val, m_max_val);
}

bool Histc::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v17_Histc_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v17_Histc_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(bf16, f16, f32, f64),
                                      histc::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      m_bins,
                                      m_min_val,
                                      m_max_val);
}

bool Histc::has_evaluate() const {
    OV_OP_SCOPE(v17_Histc_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        return false;
    }
}

}  // namespace v17
}  // namespace op
}  // namespace ov
