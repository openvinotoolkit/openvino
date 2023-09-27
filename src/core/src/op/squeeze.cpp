// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/squeeze.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <set>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/reference/copy.hpp"
#include "squeeze_shape_inference.hpp"

using namespace std;
using namespace ngraph;

op::Squeeze::Squeeze() : Op() {}

op::Squeeze::Squeeze(const Output<Node>& data, const Output<Node>& axes) : Op({data, axes}) {
    constructor_validate_and_infer_types();
}

op::Squeeze::Squeeze(const Output<Node>& data) : Op({data}) {
    constructor_validate_and_infer_types();
}

void op::Squeeze::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Squeeze_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool ngraph::op::v0::Squeeze::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Squeeze_visit_attributes);
    return true;
}

shared_ptr<Node> op::Squeeze::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Squeeze_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return make_shared<Squeeze>(new_args.at(0));
    } else if (new_args.size() == 2) {
        return make_shared<Squeeze>(new_args.at(0), new_args.at(1));
    } else {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}

OPENVINO_SUPPRESS_DEPRECATED_START
bool op::v0::Squeeze::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(inputs, inputs.size()));
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END

    if (has_evaluate()) {
        auto input_shapes = std::vector<PartialShape>{inputs[0]->get_partial_shape()};

        if (inputs.size() == 2) {
            input_shapes.push_back(inputs[1]->get_partial_shape());
        }

        auto output_shapes = shape_infer(this, input_shapes, make_tensor_accessor(inputs));

        auto out_shape = output_shapes[0].get_shape();
        outputs[0]->set_shape(out_shape);

        ov::reference::copy(inputs[0]->get_data_ptr<char>(),
                            outputs[0]->get_data_ptr<char>(),
                            shape_size(out_shape) * outputs[0]->get_element_type().size());

        return true;
    }
    return false;
}
OPENVINO_SUPPRESS_DEPRECATED_END

bool op::v0::Squeeze::has_evaluate() const {
    OV_OP_SCOPE(v0_Squeeze_has_evaluate);

    if (get_input_size() == 2) {
        switch (get_input_element_type(1)) {
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
    } else if (get_input_size() == 1) {
        return true;
    } else {
        return false;
    }
}

bool op::v0::Squeeze::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate_lower);
    if (inputs().size() > 1 && !input_value(1).get_tensor().has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v0::Squeeze::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate_upper);
    if (inputs().size() > 1 && !input_value(1).get_tensor().has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v0::Squeeze::evaluate_label(TensorLabelVector& output_labels) const {
    if (get_input_size() > 1 && !get_input_tensor(1).has_and_set_bound())
        return false;
    OPENVINO_SUPPRESS_DEPRECATED_START
    return default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool op::v0::Squeeze::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    OV_OP_SCOPE(v0_Squeeze_constant_fold);
    if (get_output_partial_shape(0).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    const auto& shape = get_output_shape(0);

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        output_values[0] = std::make_shared<op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}

bool op::v0::Squeeze::is_dynamic() const {
    return get_output_partial_shape(0).is_dynamic();
}
