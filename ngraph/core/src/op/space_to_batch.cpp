// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstddef>
#include <memory>
#include <ngraph/validation_util.hpp>
#include <numeric>
#include "itt.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/space_to_batch.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/reference/pad.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::SpaceToBatch::type_info;

ngraph::op::v1::SpaceToBatch::SpaceToBatch(const ngraph::Output<ngraph::Node>& data,
                                           const ngraph::Output<ngraph::Node>& block_shape,
                                           const ngraph::Output<ngraph::Node>& pads_begin,
                                           const ngraph::Output<ngraph::Node>& pads_end)
    : Op({data, block_shape, pads_begin, pads_end})
{
    constructor_validate_and_infer_types();
}

void op::v1::SpaceToBatch::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_SpaceToBatch_validate_and_infer_types);
    PartialShape data_pshape = get_input_partial_shape(0);
    const auto& data_type = get_input_element_type(0);
    const auto& block_shape_type = get_input_element_type(1);
    const auto& pads_begin_type = get_input_element_type(2);
    const auto& pads_end_type = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          block_shape_type.is_integral_number(),
                          "block_shape must be an integral number but got (",
                          block_shape_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_begin_type.is_integral_number(),
                          "crops_begin must be an integral number but got (",
                          pads_begin_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_end_type.is_integral_number(),
                          "crops_end must be an integral number but got (",
                          pads_end_type,
                          ").");

    auto data = input_value(0);
    auto block = input_value(1);
    auto pads_begin = input_value(2);
    auto pads_end = input_value(3);

    const auto& block_const = get_constant_from_source(block);
    const auto& pads_begin_const = get_constant_from_source(pads_begin);
    const auto& pads_end_const = get_constant_from_source(pads_end);

    if (block_const && pads_begin_const && pads_end_const && data_pshape.is_static())
    {
        const auto& data_shape = data.get_shape();

        NODE_VALIDATION_CHECK(
            this,
            (data_shape.size() >= 2),
            "The data tensor with rank lower than 2 is not supported (data rank: ",
            data_shape.size(),
            ")");

        auto block_val = block_const->cast_vector<int64_t>();
        auto pads_begin_val = pads_begin_const->cast_vector<int64_t>();
        auto pads_end_val = pads_end_const->cast_vector<int64_t>();

        int64_t block_prod = 1;
        for (long idx : block_val)
            block_prod *= idx;

        Shape output_shape = {static_cast<size_t>(data_shape[0] * block_prod)};
        for (size_t idx = 1; idx < data_shape.size(); ++idx)
        {
            NODE_VALIDATION_CHECK(
                this, block_val.at(idx) > 0, "block_shape values must be greater than 0");
            NODE_VALIDATION_CHECK(
                this,
                (pads_begin_val.at(idx) + data_shape.at(idx) + pads_end_val.at(idx)) %
                        block_val.at(idx) ==
                    0,
                "The dimension on position: ",
                idx,
                " equal to: ",
                pads_begin_val.at(idx) + data_shape.at(idx) + pads_end_val.at(idx),
                " must be a multiple of block_values[i]: ",
                block_val.at(idx));
            output_shape.push_back(
                static_cast<size_t>(pads_begin_val[idx] + data_shape[idx] + pads_end_val[idx]) /
                block_val[idx]);
        }

        set_output_size(1);
        set_output_type(0, data_type, output_shape);
    }
    else
    {
        set_output_type(0, data_type, PartialShape::dynamic(data_pshape.rank()));
    }
}

std::shared_ptr<Node>
    ngraph::op::v1::SpaceToBatch::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_SpaceToBatch_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<SpaceToBatch>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::SpaceToBatch::visit_attributes(ngraph::AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_SpaceToBatch_visit_attributes);
    return true;
}

bool ngraph::op::v1::SpaceToBatch::evaluate_space_to_batch(const HostTensorVector& outputs,
                                                           const HostTensorVector& inputs) const
{
    const auto& data = inputs[0];
    const auto& out = outputs[0];
    size_t elem_size = data->get_element_type().size();

    if (data->get_partial_shape().is_dynamic())
    {
        return false;
    }
    auto data_shape = data->get_shape();

    if (!(data->get_shape().size() == 4 || data->get_shape().size() == 5))
    {
        return false;
    }

    size_t block_values_size = shape_size(inputs[1]->get_shape());
    const auto* block_values = inputs[1]->get_data_ptr<int64_t>();
    const auto* pads_begin = inputs[2]->get_data_ptr<int64_t>();
    const auto* pads_end = inputs[3]->get_data_ptr<int64_t>();

    const char* pad_value = nullptr;
    const std::vector<char> pad_zero_value(elem_size, 0);
    if (inputs.size() == 4)
    {
        pad_value = inputs[3]->get_data_ptr<char>();
    }
    else
    {
        pad_value = pad_zero_value.data();
    }
    CoordinateDiff pads_begin_vec(shape_size(inputs[2]->get_shape()));
    pads_begin_vec.assign(pads_begin, pads_begin + shape_size(inputs[2]->get_shape()));
    CoordinateDiff pads_end_vec(shape_size(inputs[2]->get_shape()));
    pads_end_vec.assign(pads_end, pads_end + shape_size(inputs[2]->get_shape()));

    Shape padded_shape(data_shape.size());
    for (size_t i = 0; i < data_shape.size(); ++i)
    {
        padded_shape[i] = data_shape[i] + pads_begin_vec[i] + pads_end_vec[i];
    }

    std::vector<char> padded_data(shape_size(padded_shape) * elem_size);
    ngraph::runtime::reference::pad(data->get_data_ptr<char>(),
                                    pad_value,
                                    padded_data.data(),
                                    elem_size,
                                    data_shape,
                                    padded_shape,
                                    pads_begin_vec,
                                    pads_end_vec,
                                    ngraph::op::PadMode::CONSTANT);
    data_shape = padded_shape;

    Shape dispersed_shape(block_values_size + 1);
    std::vector<size_t> axes_order(block_values_size + 1);
    Shape squeezed_shape(data_shape.begin(), data_shape.end());
    std::vector<size_t> plain_axes_order(block_values_size + 1);
    std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);

    std::vector<char> flat_data(padded_data.begin(), padded_data.end());
    std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);
    std::vector<char> post_transpose_data(shape_size(data_shape) * elem_size);

    for (int64_t block_idx = block_values_size - 1; block_idx >= 0; --block_idx)
    {
        int64_t sq_shape_idx = block_values_size - 1;
        int64_t axis_idx = axes_order.size() - 1;
        for (int64_t shape_idx = dispersed_shape.size() - 1; shape_idx >= 0; --shape_idx)
        {
            if (shape_idx == (block_idx + 1))
            {
                dispersed_shape[shape_idx] = block_values[block_idx];
                axes_order[0] = shape_idx;
            }
            else if (shape_idx == block_idx)
            {
                dispersed_shape[shape_idx] = squeezed_shape[sq_shape_idx] / block_values[block_idx];
                axes_order[axis_idx] = shape_idx;
                axis_idx--;
                sq_shape_idx--;
            }
            else
            {
                dispersed_shape[shape_idx] = squeezed_shape[sq_shape_idx];
                axes_order[axis_idx] = shape_idx;
                axis_idx--;
                sq_shape_idx--;
            }
        }

        runtime::opt_kernel::reshape(flat_data.data(),
                                     dispersed_data.data(),
                                     data_shape,
                                     plain_axes_order,
                                     dispersed_shape,
                                     elem_size);
        Shape post_transpose_shape(axes_order.size());
        for (size_t i = 0; i < axes_order.size(); ++i)
        {
            post_transpose_shape[i] = dispersed_shape[axes_order[i]];
        }

        runtime::opt_kernel::reshape(dispersed_data.data(),
                                     post_transpose_data.data(),
                                     dispersed_shape,
                                     axes_order,
                                     post_transpose_shape,
                                     elem_size);
        squeezed_shape[0] *= block_values[block_idx];
        squeezed_shape[block_idx] /= block_values[block_idx];

        runtime::opt_kernel::reshape(post_transpose_data.data(),
                                     flat_data.data(),
                                     post_transpose_shape,
                                     plain_axes_order,
                                     squeezed_shape,
                                     elem_size);
        data_shape = squeezed_shape;
    }

    out->write(flat_data.data(), elem_size * shape_size(out->get_shape()));

    return true;
}

bool ngraph::op::v1::SpaceToBatch::evaluate(const HostTensorVector& outputs,
                                            const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_SpaceToBatch_evaluate);
    return evaluate_space_to_batch(outputs, inputs);
}

bool ngraph::op::v1::SpaceToBatch::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_SpaceToBatch_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic() &&
           (get_input_shape(0).size() == 4 || get_input_shape(0).size() == 5);
}
