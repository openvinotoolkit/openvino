// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/slice.hpp"

#include <numeric>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "slice_shape_inference.hpp"

using namespace std;
using namespace ngraph;

op::v8::Slice::Slice(const Output<Node>& data,
                     const Output<Node>& start,
                     const Output<Node>& stop,
                     const Output<Node>& step)
    : Op({data, start, stop, step}) {
    constructor_validate_and_infer_types();
}

op::v8::Slice::Slice(const Output<Node>& data,
                     const Output<Node>& start,
                     const Output<Node>& stop,
                     const Output<Node>& step,
                     const Output<Node>& axes)
    : Op({data, start, stop, step, axes}) {
    constructor_validate_and_infer_types();
}

bool op::v8::Slice::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_Slice_visit_attributes);
    return true;
}

std::shared_ptr<op::v0::Constant> op::v8::Slice::get_default_const_axes(const Output<Node>& start) const {
    const auto start_pshape = start.get_partial_shape();
    // Static case
    if (start_pshape.rank().is_static() && start_pshape.rank().get_length() == 1 && start_pshape[0].is_static()) {
        size_t axes_length = start_pshape[0].get_length();
        std::vector<int64_t> axes(axes_length);
        std::iota(axes.begin(), axes.end(), 0);
        return v0::Constant::create(element::i64, Shape{axes_length}, axes);
    } else {
        // Dynamic case
        return {};
    }
}

void op::v8::Slice::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Slice_validate_and_infer_types);

    const auto inputs_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          inputs_size == 4 || inputs_size == 5,
                          "Slice has to have 4 or 5 inputs. Got: ",
                          inputs_size);

    const PartialShape& data_shape = get_input_partial_shape(0);
    const auto& data_rank = data_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          data_rank.is_dynamic() || data_rank.get_length() > 0,
                          "Slice `data` input can't be a scalar.");

    if (get_input_size() < 5) {
        if (auto axes_const = get_default_const_axes(input_value(1))) {
            set_argument(4, axes_const);
        }
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        if (i > 0) {
            NODE_VALIDATION_CHECK(this,
                                  get_input_element_type(i).is_integral_number(),
                                  "Slice `",
                                  slice::shape_names[i - 1],
                                  "` input type must be integer.");
        }

        set_input_is_relevant_to_shape(i);
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape::dynamic()};

    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

std::shared_ptr<Node> op::v8::Slice::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_Slice_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<v8::Slice>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
    } else {
        return std::make_shared<v8::Slice>(new_args.at(0),
                                           new_args.at(1),
                                           new_args.at(2),
                                           new_args.at(3),
                                           new_args.at(4));
    }
}

bool op::v8::Slice::has_evaluate() const {
    OV_OP_SCOPE(v8_Slice_has_evaluate);
    switch (get_input_element_type(1)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
        break;
    default:
        return false;
    }

    if (get_input_size() > 4) {
        switch (get_input_element_type(4)) {
        case ngraph::element::i8:
        case ngraph::element::i16:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u8:
        case ngraph::element::u16:
        case ngraph::element::u32:
        case ngraph::element::u64:
            break;
        default:
            return false;
        }
    }

    return true;
}

bool op::v8::Slice::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v8_Slice_evaluate);
    OPENVINO_ASSERT(inputs.size() >= 4, "Slice evaluate needs at least 4 inputs.");

    // Static HostTensor data shape is needed to clamp and normalize `start` values
    OPENVINO_ASSERT(inputs[0]->get_partial_shape().is_static(),
                    "Can't evaluate Slice elements without static HostTensor data shape.");

    auto constant_data = std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>{};
    auto input_shapes = std::vector<PartialShape>();
    input_shapes.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto&& tensor = inputs[i];
        input_shapes.push_back(tensor->get_partial_shape());
        constant_data.emplace(i, tensor);
    }

    const auto starts = host_tensor_2_vector<int64_t>(inputs[1]);
    const auto stops = host_tensor_2_vector<int64_t>(inputs[2]);
    const auto steps = host_tensor_2_vector<int64_t>(inputs[3]);

    std::vector<int64_t> axes;
    if (inputs.size() < 5) {
        axes.reserve(starts.size());
        std::generate_n(std::back_inserter(axes), starts.size(), SeqGen<int64_t>(0));
    } else {
        axes = host_tensor_2_vector<int64_t>(inputs[4]);
    }

    auto output_shapes = std::vector<PartialShape>(1);
    shape_infer(this, input_shapes, output_shapes, constant_data);
    OPENVINO_ASSERT(output_shapes.front().is_static(), "Can't calculate static output shape for Slice evaluation.");

    outputs[0]->set_shape(output_shapes.front().to_shape());
    outputs[0]->set_element_type(inputs[0]->get_element_type());

    ngraph::runtime::reference::slice(inputs[0]->get_data_ptr<char>(),
                                      inputs[0]->get_shape(),
                                      outputs[0]->get_data_ptr<char>(),
                                      outputs[0]->get_shape(),
                                      inputs[0]->get_element_type().size(),
                                      starts,
                                      steps,
                                      axes);
    return true;
}

namespace {
bool slice_input_check(const ov::Node* node) {
    if (!node->get_input_tensor(1).has_and_set_bound())
        return false;
    if (!node->get_input_tensor(2).has_and_set_bound())
        return false;
    if (!node->get_input_tensor(3).has_and_set_bound())
        return false;
    if (node->get_input_size() == 5 && !node->get_input_tensor(4).has_and_set_bound())
        return false;
    return true;
}
}  // namespace

bool op::v8::Slice::evaluate_lower(ov::TensorVector& output_values) const {
    return slice_input_check(this) && default_lower_bound_evaluator(this, output_values);
}

bool op::v8::Slice::evaluate_upper(ov::TensorVector& output_values) const {
    return slice_input_check(this) && default_upper_bound_evaluator(this, output_values);
}

bool op::v8::Slice::evaluate_label(TensorLabelVector& output_labels) const {
    if (!slice_input_check(this))
        return false;
    OPENVINO_SUPPRESS_DEPRECATED_START
    return default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}
