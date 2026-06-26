// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bincount.hpp"

#include "bincount_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/bincount.hpp"

namespace ov {
namespace {

template <typename TData>
size_t bincount_compute_out_size(const Tensor& data, int64_t minlength) {
    return reference::bincount_output_size(data.template data<const TData>(), data.get_size(), minlength);
}

template <typename TData>
void bincount_eval_unweighted(const Tensor& data, Tensor& output, int64_t minlength, size_t out_size) {
    output.set_shape(Shape{out_size});
    reference::bincount(data.template data<const TData>(),
                        data.get_size(),
                        minlength,
                        output.data<int64_t>(),
                        out_size);
}

template <typename TData, typename TWeight>
void bincount_eval_weighted(const Tensor& data,
                            const Tensor& weights,
                            Tensor& output,
                            int64_t minlength,
                            size_t out_size) {
    output.set_shape(Shape{out_size});
    reference::bincount_weighted(data.template data<const TData>(),
                                 weights.template data<const TWeight>(),
                                 data.get_size(),
                                 minlength,
                                 output.template data<TWeight>(),
                                 out_size);
}

size_t bincount_get_out_size(const Tensor& data, int64_t minlength) {
    switch (data.get_element_type()) {
    case element::i32:
        return bincount_compute_out_size<int32_t>(data, minlength);
    case element::i64:
        return bincount_compute_out_size<int64_t>(data, minlength);
    case element::u8:
        return bincount_compute_out_size<uint8_t>(data, minlength);
    case element::u16:
        return bincount_compute_out_size<uint16_t>(data, minlength);
    case element::u32:
        return bincount_compute_out_size<uint32_t>(data, minlength);
    case element::u64:
        return bincount_compute_out_size<uint64_t>(data, minlength);
    default:
        OPENVINO_THROW("Unsupported Bincount input element type: ", data.get_element_type());
    }
}

bool bincount_dispatch_unweighted(const Tensor& data, Tensor& output, int64_t minlength, size_t out_size) {
    switch (data.get_element_type()) {
    case element::i32:
        bincount_eval_unweighted<int32_t>(data, output, minlength, out_size);
        return true;
    case element::i64:
        bincount_eval_unweighted<int64_t>(data, output, minlength, out_size);
        return true;
    case element::u8:
        bincount_eval_unweighted<uint8_t>(data, output, minlength, out_size);
        return true;
    case element::u16:
        bincount_eval_unweighted<uint16_t>(data, output, minlength, out_size);
        return true;
    case element::u32:
        bincount_eval_unweighted<uint32_t>(data, output, minlength, out_size);
        return true;
    case element::u64:
        bincount_eval_unweighted<uint64_t>(data, output, minlength, out_size);
        return true;
    default:
        return false;
    }
}

template <typename TData>
bool bincount_dispatch_weight_type(const Tensor& data,
                                   const Tensor& weights,
                                   Tensor& output,
                                   int64_t minlength,
                                   size_t out_size,
                                   const element::Type& weights_et) {
    switch (weights_et) {
    case element::f32:
        bincount_eval_weighted<TData, float>(data, weights, output, minlength, out_size);
        return true;
    case element::f64:
        bincount_eval_weighted<TData, double>(data, weights, output, minlength, out_size);
        return true;
    case element::i32:
        bincount_eval_weighted<TData, int32_t>(data, weights, output, minlength, out_size);
        return true;
    case element::i64:
        bincount_eval_weighted<TData, int64_t>(data, weights, output, minlength, out_size);
        return true;
    default:
        return false;
    }
}

bool bincount_dispatch_weighted(const Tensor& data,
                                const Tensor& weights,
                                Tensor& output,
                                int64_t minlength,
                                size_t out_size) {
    const auto& wt = weights.get_element_type();
    switch (data.get_element_type()) {
    case element::i32:
        return bincount_dispatch_weight_type<int32_t>(data, weights, output, minlength, out_size, wt);
    case element::i64:
        return bincount_dispatch_weight_type<int64_t>(data, weights, output, minlength, out_size, wt);
    case element::u8:
        return bincount_dispatch_weight_type<uint8_t>(data, weights, output, minlength, out_size, wt);
    case element::u16:
        return bincount_dispatch_weight_type<uint16_t>(data, weights, output, minlength, out_size, wt);
    case element::u32:
        return bincount_dispatch_weight_type<uint32_t>(data, weights, output, minlength, out_size, wt);
    case element::u64:
        return bincount_dispatch_weight_type<uint64_t>(data, weights, output, minlength, out_size, wt);
    default:
        return false;
    }
}

bool is_supported_data_type(const element::Type& et) {
    return et == element::i32 || et == element::i64 || et == element::u8 || et == element::u16 ||
           et == element::u32 || et == element::u64;
}

bool is_supported_weight_type(const element::Type& et) {
    return et == element::f32 || et == element::f64 || et == element::i32 || et == element::i64;
}

}  // namespace

namespace op {
namespace v17 {

Bincount::Bincount(const Output<Node>& data, int64_t minlength) : Op({data}), m_minlength(minlength) {
    constructor_validate_and_infer_types();
}

Bincount::Bincount(const Output<Node>& data, const Output<Node>& weights, int64_t minlength)
    : Op({data, weights}),
      m_minlength(minlength) {
    constructor_validate_and_infer_types();
}

bool Bincount::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_Bincount_visit_attributes);
    visitor.on_attribute("minlength", m_minlength);
    return true;
}

void Bincount::validate_and_infer_types() {
    OV_OP_SCOPE(v17_Bincount_validate_and_infer_types);

    const auto& data_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_et.is_dynamic() || is_supported_data_type(data_et),
                          "The 'data' input must have one of the following element types: "
                          "i32, i64, u8, u16, u32, u64. Got: ",
                          data_et);

    NODE_VALIDATION_CHECK(this, m_minlength >= 0, "minlength must be non-negative. Got: ", m_minlength);

    element::Type output_et = element::i64;
    if (get_input_size() == 2) {
        const auto& weights_et = get_input_element_type(1);
        NODE_VALIDATION_CHECK(this,
                              weights_et.is_dynamic() || is_supported_weight_type(weights_et),
                              "The 'weights' input must have one of the following element types: "
                              "f32, f64, i32, i64. Got: ",
                              weights_et);
        output_et = weights_et;
    }

    set_input_is_relevant_to_shape(0);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, output_et, output_shapes[0]);

    const auto input_constant = ov::util::get_constant_from_source(input_value(0));
    if (!input_constant) {
        return;
    }

    TensorVector inputs{input_constant->get_tensor_view()};
    if (get_input_size() == 2) {
        const auto weights_constant = ov::util::get_constant_from_source(input_value(1));
        if (!weights_constant) {
            return;
        }
        inputs.push_back(weights_constant->get_tensor_view());
    }

    TensorVector outputs{{output_et, {}}};
    if (!evaluate(outputs, inputs)) {
        return;
    }

    const auto& out = outputs[0];
    set_output_type(0, output_et, out.get_shape());
    get_output_tensor(0).set_lower_value(out);
    get_output_tensor(0).set_upper_value(out);
}

std::shared_ptr<Node> Bincount::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_Bincount_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return std::make_shared<v17::Bincount>(new_args.at(0), m_minlength);
    }
    return std::make_shared<v17::Bincount>(new_args.at(0), new_args.at(1), m_minlength);
}

bool Bincount::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v17_Bincount_evaluate);

    OPENVINO_ASSERT(inputs.size() == 1 || inputs.size() == 2, "Bincount expects 1 or 2 inputs.");
    if (inputs.size() == 2) {
        OPENVINO_ASSERT(inputs[0].get_size() == inputs[1].get_size(),
                        "Bincount 'data' and 'weights' inputs must have the same number of elements.");
    }

    const auto out_size = bincount_get_out_size(inputs[0], m_minlength);
    if (inputs.size() == 1) {
        return bincount_dispatch_unweighted(inputs[0], outputs[0], m_minlength, out_size);
    }
    return bincount_dispatch_weighted(inputs[0], inputs[1], outputs[0], m_minlength, out_size);
}

bool Bincount::has_evaluate() const {
    OV_OP_SCOPE(v17_Bincount_has_evaluate);
    if (!is_supported_data_type(get_input_element_type(0))) {
        return false;
    }
    return get_input_size() == 1 || is_supported_weight_type(get_input_element_type(1));
}

}  // namespace v17
}  // namespace op
}  // namespace ov
