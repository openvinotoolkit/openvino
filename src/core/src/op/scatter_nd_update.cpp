// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_nd_update.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"  // tbr
#include "ngraph/validation_util.hpp"      // tbr
#include "openvino/reference/scatter_nd_update.hpp"

using namespace ngraph;  // tbr

namespace ov {
namespace op {
namespace v3 {
std::shared_ptr<Node> ScatterNDUpdate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ScatterNDUpdate>(new_args.at(util::ScatterNDBase::INPUTS),
                                             new_args.at(util::ScatterNDBase::INDICES),
                                             new_args.at(util::ScatterNDBase::UPDATES));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace scatter {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& arg2,
              const HostTensorPtr& out) {
    using T = typename element_type_traits<ET>::value_type;
    out->set_shape(arg0->get_shape());

    if (arg1->get_element_type() == element::i64) {
        reference::scatterNdUpdate<T, int64_t>(arg0->get_data_ptr<ET>(),
                                               arg1->get_data_ptr<int64_t>(),
                                               arg2->get_data_ptr<ET>(),
                                               out->get_data_ptr<ET>(),
                                               arg0->get_shape(),
                                               arg1->get_shape(),
                                               arg2->get_shape());
    } else if (arg1->get_element_type() == element::i32) {
        reference::scatterNdUpdate<T, int32_t>(arg0->get_data_ptr<ET>(),
                                               arg1->get_data_ptr<int32_t>(),
                                               arg2->get_data_ptr<ET>(),
                                               out->get_data_ptr<ET>(),
                                               arg0->get_shape(),
                                               arg1->get_shape(),
                                               arg2->get_shape());
    } else {
        OPENVINO_THROW("Unexpected type ",
                       arg1->get_element_type().c_type_string(),
                       " for ScatterNDUpdate evaluate method.");
    }

    return true;
}

bool evaluate_scatter(const HostTensorPtr& arg0,
                      const HostTensorPtr& arg1,
                      const HostTensorPtr& arg2,
                      const HostTensorPtr& out) {
    bool rc = true;

    switch (out->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_scatter, i32, arg0, arg1, arg2, out);
        OPENVINO_TYPE_CASE(evaluate_scatter, i64, arg0, arg1, arg2, out);
        OPENVINO_TYPE_CASE(evaluate_scatter, u32, arg0, arg1, arg2, out);
        OPENVINO_TYPE_CASE(evaluate_scatter, u64, arg0, arg1, arg2, out);
        OPENVINO_TYPE_CASE(evaluate_scatter, f16, arg0, arg1, arg2, out);
        OPENVINO_TYPE_CASE(evaluate_scatter, f32, arg0, arg1, arg2, out);
        OPENVINO_TYPE_CASE(evaluate_scatter, boolean, arg0, arg1, arg2, out);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace scatter

bool ScatterNDUpdate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate);
    OPENVINO_ASSERT(!inputs.empty());
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(inputs, 3));
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END

    return scatter::evaluate_scatter(inputs[0], inputs[1], inputs[2], outputs[0]);
}

bool ScatterNDUpdate::has_evaluate() const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_has_evaluate);

    switch (get_output_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
    case element::boolean:
        break;
    default:
        return false;
    }
    switch (get_input_element_type(1)) {
    case element::i32:
    case element::i64:
        return true;
    default:
        return false;
    }
}

bool ScatterNDUpdate::evaluate_lower(TensorVector& output_values) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool ScatterNDUpdate::evaluate_upper(TensorVector& output_values) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool ScatterNDUpdate::evaluate_label(TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_label);
    return default_label_evaluator(this, {0, 2}, output_labels);
}
}  // namespace v3
}  // namespace op
}  // namespace ov
