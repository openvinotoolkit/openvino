// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/batch_to_space.hpp"

#include <batch_to_space_shape_inference.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <ngraph/ops.hpp>
#include <ngraph/validation_util.hpp>
#include <numeric>

#include "itt.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/node.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/shape.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/op/util/slice_plan.hpp"
#include "openvino/reference/reshape.hpp"
#include "openvino/reference/strided_slice.hpp"

using namespace std;
using namespace ngraph;

ngraph::op::v1::BatchToSpace::BatchToSpace(const ngraph::Output<ngraph::Node>& data,
                                           const ngraph::Output<ngraph::Node>& block_shape,
                                           const ngraph::Output<ngraph::Node>& crops_begin,
                                           const ngraph::Output<ngraph::Node>& crops_end)
    : Op({data, block_shape, crops_begin, crops_end}) {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
    ov::mark_as_precision_sensitive(input(3));
    constructor_validate_and_infer_types();
}

void op::v1::BatchToSpace::validate_and_infer_types() {
    OV_OP_SCOPE(v1_BatchToSpace_validate_and_infer_types);

    const auto& data_et = get_input_element_type(0);
    const auto& block_shape_et = get_input_element_type(1);
    const auto& crops_begin_et = get_input_element_type(2);
    const auto& crops_end_et = get_input_element_type(3);

    element::Type inputs_integer_et{};
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(inputs_integer_et, crops_begin_et, crops_end_et) &&
                              element::Type::merge(inputs_integer_et, inputs_integer_et, block_shape_et),
                          "block_shape, crops_begin and crops_end inputs must have same element type. Got: ",
                          block_shape_et,
                          ", ",
                          crops_begin_et,
                          " and ",
                          crops_end_et);

    NODE_VALIDATION_CHECK(this,
                          inputs_integer_et.is_integral_number(),
                          "block_shape and crops inputs must have integer element type. Got: ",
                          inputs_integer_et);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shape = shape_infer(this, get_node_input_partial_shapes(*this)).front();
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, data_et, output_shape);
}

std::shared_ptr<ngraph::Node> ngraph::op::v1::BatchToSpace::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_BatchToSpace_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<BatchToSpace>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::BatchToSpace::visit_attributes(ngraph::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_BatchToSpace_visit_attributes);
    return true;
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace {
bool batch_to_space_evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) {
    auto data = inputs[0];
    const auto elem_size = data->get_element_type().size();

    auto data_shape = data->get_shape();

    auto const block_values_size = shape_size(inputs[1]->get_shape());

    const auto* block_values = inputs[1]->get_data_ptr<int64_t>();
    const auto* crops_begin_values = inputs[2]->get_data_ptr<int64_t>();
    const auto* crops_end_values = inputs[3]->get_data_ptr<int64_t>();

    ov::Shape dispersed_shape(1);
    dispersed_shape.insert(dispersed_shape.end(), data_shape.begin(), data_shape.end());
    std::vector<size_t> axes_order(block_values_size + 1);
    std::vector<size_t> plain_axes_order(block_values_size + 1);
    std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);
    ov::Shape squeezed_shape(data_shape.begin(), data_shape.end());
    if (squeezed_shape.size() > block_values_size) {
        return false;
    }

    auto* flat_data = data->get_data_ptr<char>();
    std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);

    ov::Shape post_transpose_shape(axes_order.size());
    std::vector<char> post_transpose_data(shape_size(data_shape) * elem_size);

    for (size_t block_idx = 1; block_idx < block_values_size; ++block_idx) {
        dispersed_shape[0] = block_values[block_idx];
        dispersed_shape[1] /= block_values[block_idx];
        ov::reference::reshape(flat_data,
                               dispersed_data.data(),
                               data_shape,
                               plain_axes_order,
                               dispersed_shape,
                               elem_size);

        size_t val = 1;
        for (size_t axis_idx = 0; axis_idx <= block_values_size; ++axis_idx) {
            if ((block_idx + 1) == axis_idx) {
                axes_order[axis_idx] = 0;
            } else {
                axes_order[axis_idx] = val;
                val++;
            }
        }
        for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx) {
            post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
        }

        ov::reference::reshape(dispersed_data.data(),
                               post_transpose_data.data(),
                               dispersed_shape,
                               axes_order,
                               post_transpose_shape,
                               elem_size);
        squeezed_shape[0] = dispersed_shape[1];
        squeezed_shape[block_idx] *= block_values[block_idx];
        dispersed_shape[block_idx + 1] = squeezed_shape[block_idx];
        ov::reference::reshape(post_transpose_data.data(),
                               flat_data,
                               post_transpose_shape,
                               plain_axes_order,
                               squeezed_shape,
                               elem_size);
        data_shape = squeezed_shape;
    }

    std::vector<int64_t> upperbounds_values(data_shape.size());
    for (size_t i = 0; i < data_shape.size(); ++i) {
        upperbounds_values[i] = data_shape[i] - crops_end_values[i];
    }

    std::vector<size_t> begin_mask(data_shape.size(), 0);
    std::vector<size_t> end_mask(data_shape.size(), 0);

    std::vector<int64_t> begins(shape_size(inputs[2]->get_shape()));
    begins.assign(crops_begin_values, crops_begin_values + shape_size(inputs[2]->get_shape()));

    std::vector<int64_t> default_strides(begins.size(), 1);
    const auto slice_plan = ov::op::util::make_slice_plan(data_shape,
                                                          begins,
                                                          upperbounds_values,
                                                          default_strides,
                                                          begin_mask,
                                                          end_mask,
                                                          AxisSet(),
                                                          AxisSet(),
                                                          AxisSet());
    ov::reference::strided_slice(flat_data, outputs[0]->get_data_ptr<char>(), data_shape, slice_plan, elem_size);
    return true;
}
}  // namespace

bool ngraph::op::v1::BatchToSpace::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_BatchToSpace_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(inputs, 4));
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END

    if (outputs[0]->get_partial_shape().is_dynamic()) {
        std::vector<ov::PartialShape> input_shapes;
        input_shapes.reserve(inputs.size());

        for (size_t i = 0; i < inputs.size(); ++i) {
            input_shapes.push_back(inputs[i]->get_partial_shape());
            if (input_shapes.back().is_dynamic()) {
                return false;
            }
        }

        const auto output_shape = shape_infer(this, input_shapes, ov::make_tensor_accessor(inputs)).front().to_shape();

        outputs[0]->set_element_type(inputs[0]->get_element_type());
        outputs[0]->set_shape(output_shape);
    }

    return batch_to_space_evaluate(outputs, inputs);
}

bool ngraph::op::v1::BatchToSpace::has_evaluate() const {
    OV_OP_SCOPE(v1_BatchToSpace_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic() && get_input_shape(0).size() >= 2 &&
           get_input_shape(0).size() <= shape_size(get_input_shape(1));
}
