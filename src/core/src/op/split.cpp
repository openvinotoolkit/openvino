// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/split.hpp"

#include <numeric>
#include <split_shape_inference.hpp>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/split.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::v1::Split::Split(const Output<Node>& data, const Output<Node>& axis, const size_t num_splits)
    : Op({data, axis}),
      m_num_splits{num_splits} {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Split::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Split_visit_attributes);
    visitor.on_attribute("num_splits", m_num_splits);
    return true;
}

void op::v1::Split::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Split_validate_and_infer_types);
    const auto& axis_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          axis_et.is_integral_number(),
                          "Element type of 'axis' input must be integer. Got: ",
                          axis_et);

    NODE_VALIDATION_CHECK(this,
                          m_num_splits > 0,
                          "Attribute 'num_splits' must be greater than zero. Got: ",
                          m_num_splits);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    std::vector<ov::PartialShape> output_shapes;
    shape_infer(this, input_shapes, output_shapes);

    for (size_t i = 0; i < m_num_splits; ++i) {
        set_output_type(i, get_input_element_type(0), output_shapes[i]);
    }

    set_input_is_relevant_to_shape(0);
}

shared_ptr<Node> op::v1::Split::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Split_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Split>(new_args.at(0), new_args.at(1), m_num_splits);
}

bool op::v1::Split::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Split_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, m_num_splits) && validate_host_tensor_vector(inputs, 2));
    OPENVINO_SUPPRESS_DEPRECATED_END

    if (has_evaluate()) {
        const auto& data_tensor = inputs[0];
        const auto& axis_tensor = inputs[1];

        const auto constant_data = std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>{{1, axis_tensor}};
        const auto input_shapes =
            std::vector<PartialShape>{data_tensor->get_partial_shape(), axis_tensor->get_partial_shape()};
        auto output_shapes = std::vector<PartialShape>();

        shape_infer(this, input_shapes, output_shapes, constant_data);

        auto outputs_data = std::vector<char*>(m_num_splits);
        for (size_t i = 0; i < m_num_splits; ++i) {
            outputs[i]->set_shape(output_shapes[i].get_shape());
            outputs_data[i] = outputs[i]->get_data_ptr<char>();
        }

        auto axis = host_tensor_2_vector<int64_t>(axis_tensor)[0];
        OPENVINO_SUPPRESS_DEPRECATED_START
        axis = normalize_axis(this, axis, data_tensor->get_partial_shape().rank());
        OPENVINO_SUPPRESS_DEPRECATED_END

        ngraph::runtime::reference::split(data_tensor->get_data_ptr<char>(),
                                          data_tensor->get_shape(),
                                          data_tensor->get_element_type().size(),
                                          axis,
                                          m_num_splits,
                                          outputs_data.data());
        return true;
    }
    return false;
}

bool op::v1::Split::has_evaluate() const {
    OV_OP_SCOPE(v1_Split_has_evaluate);
    return get_input_element_type(1).is_integral_number();
}

bool op::v1::Split::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Split_evaluate_lower);

    return input(1).get_tensor().has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool op::v1::Split::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Split_evaluate_upper);

    return input(1).get_tensor().has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool op::v1::Split::evaluate_label(TensorLabelVector& output_labels) const {
    OPENVINO_ASSERT(output_labels.size() == get_num_splits());

    OPENVINO_SUPPRESS_DEPRECATED_START
    return input(1).get_tensor().has_and_set_bound() && default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}
