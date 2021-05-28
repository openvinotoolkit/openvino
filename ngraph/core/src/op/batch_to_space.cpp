// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstddef>
#include <memory>
#include <ngraph/ops.hpp>
#include <ngraph/validation_util.hpp>
#include <numeric>
#include "itt.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/batch_to_space.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/reference/strided_slice.hpp"
#include "ngraph/slice_plan.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::BatchToSpace::type_info;

ngraph::op::v1::BatchToSpace::BatchToSpace(const ngraph::Output<ngraph::Node>& data,
                                           const ngraph::Output<ngraph::Node>& block_shape,
                                           const ngraph::Output<ngraph::Node>& crops_begin,
                                           const ngraph::Output<ngraph::Node>& crops_end)
    : Op({data, block_shape, crops_begin, crops_end})
{
    constructor_validate_and_infer_types();
}

void op::v1::BatchToSpace::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_BatchToSpace_validate_and_infer_types);
    PartialShape data_pshape = get_input_partial_shape(0);

    const auto& data_type = get_input_element_type(0);
    const auto& block_shape_type = get_input_element_type(1);
    const auto& crops_begin_type = get_input_element_type(2);
    const auto& crops_end_type = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          block_shape_type.is_integral_number(),
                          "block_shape must be an integral number but got (",
                          block_shape_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          crops_begin_type.is_integral_number(),
                          "crops_begin must be an integral number but got (",
                          crops_begin_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          crops_end_type.is_integral_number(),
                          "crops_end must be an integral number but got (",
                          crops_end_type,
                          ").");

    auto data = input_value(0);
    auto block = input_value(1);
    auto crops_begin = input_value(2);
    auto crops_end = input_value(3);

    auto block_const = get_constant_from_source(block);
    auto crops_begin_const = get_constant_from_source(crops_begin);
    auto crops_end_const = get_constant_from_source(crops_end);

    if (block_const && crops_begin_const && crops_end_const && data_pshape.is_static())
    {
        const auto& data_shape = data.get_shape();

        NODE_VALIDATION_CHECK(
            this,
            (data_shape.size() >= 2),
            "The data tensor with rank lower than 2 is not supported (data rank: ",
            data_shape.size(),
            ")");

        auto block_val = block_const->cast_vector<int64_t>();
        auto crops_begin_val = crops_begin_const->cast_vector<int64_t>();
        auto crops_end_val = crops_end_const->cast_vector<int64_t>();

        int64_t block_prod = 1;
        for (long val : block_val)
        {
            NODE_VALIDATION_CHECK(this, val > 0, "block_shape values must be greater than 0");
            block_prod *= val;
        }

        NODE_VALIDATION_CHECK(this,
                              data_shape.at(0) % block_prod == 0,
                              "BatchToSpace: The input data's 'batch' axis size: ",
                              data_shape.at(0),
                              " must be a multiple of ",
                              " product of block_shape values: ",
                              block_prod);

        Shape output_shape = {static_cast<size_t>(data_shape[0] / block_prod)};
        for (size_t idx = 1; idx < data_shape.size(); ++idx)
        {
            output_shape.push_back(static_cast<size_t>(data_shape[idx] * block_val[idx] -
                                                       crops_begin_val[idx] - crops_end_val[idx]));
        }

        set_output_size(1);
        set_output_type(0, data_type, output_shape);
    }
    else
    {
        set_output_type(0, data_type, PartialShape::dynamic(data_pshape.rank()));
    }
}

std::shared_ptr<ngraph::Node>
    ngraph::op::v1::BatchToSpace::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_BatchToSpace_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<BatchToSpace>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::BatchToSpace::visit_attributes(ngraph::AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_BatchToSpace_visit_attributes);
    return true;
}

namespace
{
    bool batch_to_space_evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
    {
        auto data = inputs[0];
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
        const auto* crops_begin_values = inputs[2]->get_data_ptr<int64_t>();
        const auto* crops_end_values = inputs[3]->get_data_ptr<int64_t>();

        Shape dispersed_shape(1);
        dispersed_shape.insert(dispersed_shape.end(), data_shape.begin(), data_shape.end());
        std::vector<size_t> axes_order(block_values_size + 1);
        std::vector<size_t> plain_axes_order(block_values_size + 1);
        std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);
        Shape squeezed_shape(data_shape.begin(), data_shape.end());
        if (squeezed_shape.size() > block_values_size)
        {
            return false;
        }

        auto* flat_data = data->get_data_ptr<char>();
        std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);

        Shape post_transpose_shape(axes_order.size());
        std::vector<char> post_transpose_data(shape_size(data_shape) * elem_size);

        for (size_t block_idx = 1; block_idx < block_values_size; ++block_idx)
        {
            dispersed_shape[0] = block_values[block_idx];
            dispersed_shape[1] /= block_values[block_idx];
            runtime::opt_kernel::reshape(flat_data,
                                         dispersed_data.data(),
                                         data_shape,
                                         plain_axes_order,
                                         dispersed_shape,
                                         elem_size);

            size_t val = 1;
            for (size_t axis_idx = 0; axis_idx <= block_values_size; ++axis_idx)
            {
                if ((block_idx + 1) == axis_idx)
                {
                    axes_order[axis_idx] = 0;
                }
                else
                {
                    axes_order[axis_idx] = val;
                    val++;
                }
            }
            for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx)
            {
                post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
            }

            runtime::opt_kernel::reshape(dispersed_data.data(),
                                         post_transpose_data.data(),
                                         dispersed_shape,
                                         axes_order,
                                         post_transpose_shape,
                                         elem_size);
            squeezed_shape[0] = dispersed_shape[1];
            squeezed_shape[block_idx] *= block_values[block_idx];
            dispersed_shape[block_idx + 1] = squeezed_shape[block_idx];
            runtime::opt_kernel::reshape(post_transpose_data.data(),
                                         flat_data,
                                         post_transpose_shape,
                                         plain_axes_order,
                                         squeezed_shape,
                                         elem_size);
            data_shape = squeezed_shape;
        }

        std::vector<int64_t> upperbounds_values(data_shape.size());
        for (size_t i = 0; i < data_shape.size(); ++i)
        {
            upperbounds_values[i] = data_shape[i] - crops_end_values[i];
        }

        std::vector<size_t> begin_mask(data_shape.size(), 0);
        std::vector<size_t> end_mask(data_shape.size(), 0);

        std::vector<int64_t> begins(shape_size(inputs[2]->get_shape()));
        begins.assign(crops_begin_values, crops_begin_values + shape_size(inputs[2]->get_shape()));

        std::vector<int64_t> default_strides(begins.size(), 1);
        SlicePlan slice_plan = make_slice_plan(data_shape,
                                               begins,
                                               upperbounds_values,
                                               default_strides,
                                               begin_mask,
                                               end_mask,
                                               AxisSet(),
                                               AxisSet(),
                                               AxisSet());
        runtime::reference::strided_slice(
            flat_data, outputs[0]->get_data_ptr<char>(), data_shape, slice_plan, elem_size);
        return true;
    }
} // namespace

bool ngraph::op::v1::BatchToSpace::evaluate(const HostTensorVector& outputs,
                                            const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_BatchToSpace);
    return batch_to_space_evaluate(outputs, inputs);
}

bool ngraph::op::v1::BatchToSpace::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_BatchToSpace_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic() &&
           (get_input_shape(0).size() == 4 || get_input_shape(0).size() == 5) &&
           get_input_shape(0).size() <= shape_size(get_input_shape(1));
}
