// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/split.hpp"

#include <numeric>
#include <split_shape_inference.hpp>

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

BWDCMP_RTTI_DEFINITION(op::v1::Split);

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
    const element::Type& axis_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          axis_et.is_integral_number(),
                          "Element type of 'axis' input must be integer. Got: ",
                          axis_et);

    NODE_VALIDATION_CHECK(this,
                          m_num_splits > 0,
                          "Attribute 'num_splits' must be greater than zero. Got: ",
                          m_num_splits);

    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0), get_input_partial_shape(1)};
    std::vector<ov::PartialShape> output_shapes;
    shape_infer(this, input_shapes, output_shapes);

    set_output_size(m_num_splits);
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

namespace split {
namespace {
inline bool evaluate(const HostTensorPtr& data_tensor,
                     const HostTensorVector& outputs,
                     const int64_t axis,
                     const int64_t num_splits) {
    ov::Shape output_shape = data_tensor->get_shape();
    std::vector<char*> outputs_data(num_splits);
    output_shape.at(axis) /= num_splits;
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i]->set_shape(output_shape);
        outputs_data[i] = outputs[i]->get_data_ptr<char>();
    }
    ngraph::runtime::reference::split(data_tensor->get_data_ptr<char>(),
                                      data_tensor->get_shape(),
                                      data_tensor->get_element_type().size(),
                                      axis,
                                      num_splits,
                                      outputs_data.data());
    return true;
}

bool evaluate_split(const HostTensorPtr& data_tensor,
                    const HostTensorPtr& axis_tensor,
                    const HostTensorVector& outputs,
                    const int64_t num_splits,
                    const Node* split_node) {
    NGRAPH_CHECK(axis_tensor->get_element_type().is_integral_number(), "axis element type is not integral data type");

    int64_t axis = host_tensor_2_vector<int64_t>(axis_tensor)[0];

    axis = ngraph::normalize_axis(split_node, axis, data_tensor->get_partial_shape().rank());
    evaluate(data_tensor, outputs, axis, num_splits);
    return true;
}
}  // namespace
}  // namespace split

bool op::v1::Split::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Split_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, m_num_splits) && validate_host_tensor_vector(inputs, 2));
    const auto& data = inputs[0];
    const auto& axis = inputs[1];
    return split::evaluate_split(data, axis, outputs, m_num_splits, this);
}

bool op::v1::Split::has_evaluate() const {
    OV_OP_SCOPE(v1_Split_has_evaluate);
    return get_input_element_type(1).is_integral_number();
}
