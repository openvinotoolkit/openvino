//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cmath>
#include <numeric>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/fused/layer_norm.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::LayerNorm::type_info;
constexpr NodeTypeInfo op::LayerNormBackprop::type_info;

op::LayerNorm::LayerNorm(const Output<Node>& data,
                         const Output<Node>& scale,
                         const Output<Node>& bias,
                         bool keep_stats,
                         int64_t begin_norm_axis,
                         double epsilon)
    : FusedOp({data, scale, bias})
    , m_keep_stats(keep_stats)
    , m_use_affine(true)
    , m_begin_norm_axis(begin_norm_axis)
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

op::LayerNorm::LayerNorm(const Output<Node>& data,
                         bool keep_stats,
                         int64_t begin_norm_axis,
                         double epsilon)
    : FusedOp({data})
    , m_keep_stats(keep_stats)
    , m_use_affine(false)
    , m_begin_norm_axis(begin_norm_axis)
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

// All input shape should be static by this point
NodeVector op::LayerNorm::decompose_op() const
{
    const PartialShape& data_shape = get_input_partial_shape(0);
    if (data_shape.is_dynamic())
    {
        throw ngraph_error("Data needs to have static shape to decompose");
    }
    if (m_use_affine)
    {
        const PartialShape& scale_shape = get_input_partial_shape(1);
        const PartialShape& bias_shape = get_input_partial_shape(2);
        if (scale_shape.is_dynamic())
        {
            throw ngraph_error("Scale needs to have static shape to decompose");
        }
        if (bias_shape.is_dynamic())
        {
            throw ngraph_error("Bias needs to have static shape to decompose");
        }
    }

    // Compute real axis
    auto shape = data_shape.to_shape();
    int64_t n_axis = m_begin_norm_axis >= 0 ? m_begin_norm_axis : shape.size() + m_begin_norm_axis;

    // Get input data
    auto data = input_value(0);

    // Compute mean
    std::vector<size_t> post_reduction_axes(shape.size() - n_axis);
    std::iota(post_reduction_axes.begin(), post_reduction_axes.end(), n_axis);
    auto mean = builder::mean(data, post_reduction_axes);
    AxisSet post_axis_set;
    for (size_t i = static_cast<size_t>(n_axis); i < shape.size(); i++)
    {
        post_axis_set.insert(i);
    }
    auto b_mean = make_shared<ngraph::op::Broadcast>(mean, shape, post_axis_set);

    // Compute variance
    auto var = builder::variance(data, post_reduction_axes);

    // Compute standard deviation with epsilon
    auto epsilon = builder::make_constant(var->get_element_type(), var->get_shape(), m_epsilon);
    auto stddev = make_shared<op::Sqrt>(var + epsilon);
    auto b_stddev = make_shared<op::Broadcast>(stddev, shape, post_axis_set);

    // Get normalized input
    auto norm = (data - b_mean) / b_stddev;

    // Apply affine transformation
    if (m_use_affine)
    {
        AxisSet pre_axis_set;
        for (size_t i = 0; i < static_cast<size_t>(n_axis); i++)
        {
            pre_axis_set.insert(i);
        }
        auto scale = input_value(1);
        auto bias = input_value(2);
        auto scale_shape = get_input_partial_shape(1).to_shape();
        if (shape.size() - n_axis != scale_shape.size())
        {
            Shape reshape_shape(shape.begin() + m_begin_norm_axis, shape.end());
            scale = make_shared<op::Reshape>(scale, AxisVector{0}, reshape_shape);
            bias = make_shared<op::Reshape>(bias, AxisVector{0}, reshape_shape);
        }
        auto b_scale = make_shared<op::Broadcast>(scale, shape, pre_axis_set);
        auto b_bias = make_shared<op::Broadcast>(bias, shape, pre_axis_set);
        norm = norm * b_scale + b_bias;
    }

    // Return output nodes
    NodeVector retval;
    retval.emplace_back(norm);
    if (m_keep_stats)
    {
        retval.emplace_back(mean);
        retval.emplace_back(var);
    }
    return retval;
}

shared_ptr<Node> op::LayerNorm::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 1 && new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    if (!m_use_affine)
    {
        return make_shared<LayerNorm>(new_args.at(0), m_keep_stats, m_begin_norm_axis, m_epsilon);
    }
    else
    {
        return make_shared<LayerNorm>(new_args.at(0),
                                      new_args.at(1),
                                      new_args.at(2),
                                      m_keep_stats,
                                      m_begin_norm_axis,
                                      m_epsilon);
    }
}

void op::LayerNorm::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    const PartialShape& data_shape = get_input_partial_shape(0);
    Rank data_rank = data_shape.rank();
    int64_t d_rank = -1;
    int64_t n_axis = -1;
    if (data_rank.is_static())
    {
        d_rank = data_rank.get_length();
        n_axis = m_begin_norm_axis >= 0 ? m_begin_norm_axis : d_rank + m_begin_norm_axis;
        NODE_VALIDATION_CHECK(
            this, n_axis >= 0 && n_axis < d_rank, "begin_norm_axis is out of range");

        if (m_use_affine)
        {
            const PartialShape& scale_shape = get_input_partial_shape(1);
            const PartialShape& bias_shape = get_input_partial_shape(2);
            Rank scale_rank = scale_shape.rank();
            Rank bias_rank = bias_shape.rank();
            if (scale_rank.is_static() && bias_rank.is_static())
            {
                int64_t s_rank = scale_rank.get_length();
                int64_t b_rank = bias_rank.get_length();
                NODE_VALIDATION_CHECK(this,
                                      s_rank == b_rank &&
                                          ((s_rank == (d_rank - n_axis)) || s_rank == 1),
                                      "Scale and/or bias rank is incorrect");
            }
        }
    }

    if (m_keep_stats)
    {
        set_output_size(3);
        // output shape: data_shape[:begin_norm_axis]
        if (d_rank > 0)
        {
            std::vector<Dimension> stats_dim;
            for (int64_t i = 0; i < n_axis; i++)
            {
                stats_dim.emplace_back(data_shape[i]);
            }
            PartialShape stats_shape(stats_dim);
            set_output_type(1, input_element_type, stats_shape);
            set_output_type(2, input_element_type, stats_shape);
        }
        else // set shape to dynamic
        {
            set_output_type(1, input_element_type, PartialShape::dynamic());
            set_output_type(2, input_element_type, PartialShape::dynamic());
        }
    }
    PartialShape norm_shape{data_shape};
    set_output_type(0, input_element_type, norm_shape);
}

void op::LayerNorm::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);
    auto data = input_value(0);
    if (m_use_affine)
    {
        auto scale = input_value(1);
        auto bias = input_value(2);
        if (m_keep_stats)
        {
            auto mean = outputs()[1];
            auto variance = outputs()[2];
            auto bprop = make_shared<op::LayerNormBackprop>(
                data, delta, mean, variance, scale, m_begin_norm_axis, m_epsilon);
            adjoints.add_delta(data, bprop->outputs()[0]);
            adjoints.add_delta(scale, bprop->outputs()[1]);
            adjoints.add_delta(bias, bprop->outputs()[2]);
        }
        else
        {
            auto bprop = make_shared<op::LayerNormBackprop>(
                data, delta, scale, m_begin_norm_axis, m_epsilon);
            adjoints.add_delta(data, bprop->outputs()[0]);
            adjoints.add_delta(scale, bprop->outputs()[1]);
            adjoints.add_delta(bias, bprop->outputs()[2]);
        }
    }
    else
    {
        if (m_keep_stats)
        {
            auto mean = outputs()[1];
            auto variance = outputs()[2];
            auto bprop = make_shared<op::LayerNormBackprop>(
                data, delta, mean, variance, m_begin_norm_axis, m_epsilon);
            adjoints.add_delta(data, bprop->outputs()[0]);
        }
        else
        {
            auto bprop =
                make_shared<op::LayerNormBackprop>(data, delta, m_begin_norm_axis, m_epsilon);
            adjoints.add_delta(data, bprop->outputs()[0]);
        }
    }
}

op::LayerNormBackprop::LayerNormBackprop(const Output<Node>& data,
                                         const Output<Node>& delta,
                                         const Output<Node>& mean,
                                         const Output<Node>& variance,
                                         const Output<Node>& scale,
                                         int64_t begin_norm_axis,
                                         double epsilon)
    : FusedOp({data, delta, mean, variance, scale})
    , m_use_stats(true)
    , m_use_affine(true)
    , m_begin_norm_axis(begin_norm_axis)
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

op::LayerNormBackprop::LayerNormBackprop(const Output<Node>& data,
                                         const Output<Node>& delta,
                                         const Output<Node>& mean,
                                         const Output<Node>& variance,
                                         int64_t begin_norm_axis,
                                         double epsilon)
    : FusedOp({data, delta, mean, variance})
    , m_use_stats(true)
    , m_use_affine(false)
    , m_begin_norm_axis(begin_norm_axis)
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

op::LayerNormBackprop::LayerNormBackprop(const Output<Node>& data,
                                         const Output<Node>& delta,
                                         const Output<Node>& scale,
                                         int64_t begin_norm_axis,
                                         double epsilon)
    : FusedOp({data, delta, scale})
    , m_use_stats(false)
    , m_use_affine(true)
    , m_begin_norm_axis(begin_norm_axis)
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

op::LayerNormBackprop::LayerNormBackprop(const Output<Node>& data,
                                         const Output<Node>& delta,
                                         int64_t begin_norm_axis,
                                         double epsilon)
    : FusedOp({data, delta})
    , m_use_stats(false)
    , m_use_affine(false)
    , m_begin_norm_axis(begin_norm_axis)
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

// All input shape should be static by this point
NodeVector op::LayerNormBackprop::decompose_op() const
{
    const PartialShape& data_shape = get_input_partial_shape(0);
    if (data_shape.is_dynamic())
    {
        throw ngraph_error("Data needs to have static shape to decompose");
    }
    const PartialShape& delta_shape = get_input_partial_shape(1);
    if (delta_shape.is_dynamic())
    {
        throw ngraph_error("Delta needs to have static shape to decompose");
    }
    if (m_use_stats)
    {
        const PartialShape& mean_shape = get_input_partial_shape(2);
        const PartialShape& var_shape = get_input_partial_shape(3);
        if (mean_shape.is_dynamic())
        {
            throw ngraph_error("Mean needs to have static shape to decompose");
        }
        if (var_shape.is_dynamic())
        {
            throw ngraph_error("Variance needs to have static shape to decompose");
        }
    }
    if (m_use_affine)
    {
        const PartialShape& scale_shape = get_input_partial_shape(m_use_stats ? 4 : 2);
        if (scale_shape.is_dynamic())
        {
            throw ngraph_error("Scale needs to have static shape to decompose");
        }
    }

    // Compute real axis
    auto shape = data_shape.to_shape();
    int64_t n_axis = m_begin_norm_axis >= 0 ? m_begin_norm_axis : shape.size() + m_begin_norm_axis;

    // Get input data
    auto data = input_value(0);

    // Get delta
    auto delta = input_value(1);

    // Get mean
    std::vector<size_t> post_reduction_axes(shape.size() - n_axis);
    std::iota(post_reduction_axes.begin(), post_reduction_axes.end(), n_axis);
    auto mean =
        m_use_stats ? input_value(2) : builder::mean(data, post_reduction_axes)->outputs()[0];

    AxisSet post_axis_set;
    for (size_t i = static_cast<size_t>(n_axis); i < shape.size(); i++)
    {
        post_axis_set.insert(i);
    }
    auto b_mean = make_shared<ngraph::op::Broadcast>(mean, shape, post_axis_set);

    // Get variance
    auto var =
        m_use_stats ? input_value(3) : builder::variance(data, post_reduction_axes)->outputs()[0];

    // Compute standard deviation with epsilon
    auto epsilon = builder::make_constant(var.get_element_type(), var.get_shape(), m_epsilon);
    auto stddev = make_shared<op::Sqrt>(var + epsilon);
    auto b_stddev = make_shared<op::Broadcast>(stddev, shape, post_axis_set);

    // Get normalized input
    auto norm = (data - b_mean) / b_stddev;

    // Get gradient for data
    auto d_data = delta / b_stddev;

    bool scale_flattened = false;
    if (m_use_affine)
    {
        AxisSet pre_axis_set;
        for (size_t i = 0; i < static_cast<size_t>(n_axis); i++)
        {
            pre_axis_set.insert(i);
        }

        size_t scale_idx = m_use_stats ? 4 : 2;
        auto scale = input_value(scale_idx);
        auto scale_shape = get_input_partial_shape(scale_idx).to_shape();
        if (shape.size() - n_axis != scale_shape.size())
        {
            scale_flattened = true;
            Shape reshape_shape(shape.begin() + m_begin_norm_axis, shape.end());
            scale = make_shared<op::Reshape>(scale, AxisVector{0}, reshape_shape);
        }
        auto b_scale = make_shared<op::Broadcast>(scale, shape, pre_axis_set);
        d_data = d_data * b_scale;
    }
    auto d_mean = make_shared<op::Broadcast>(
        builder::mean(-d_data, post_reduction_axes), shape, post_axis_set);
    auto d_stddev =
        norm * make_shared<op::Broadcast>(
                   builder::mean(-d_data * norm, post_reduction_axes), shape, post_axis_set);
    d_data = d_data + d_mean + d_stddev;

    NodeVector retval;
    retval.emplace_back(d_data);

    // Get gradients for affine
    if (m_use_affine)
    {
        std::vector<size_t> pre_reduction_axes(n_axis);
        std::iota(pre_reduction_axes.begin(), pre_reduction_axes.end(), 0);
        auto d_bias = make_shared<op::Sum>(delta, pre_reduction_axes);
        auto d_scale = make_shared<op::Sum>(delta * norm, pre_reduction_axes);
        if (scale_flattened)
        {
            std::vector<size_t> flatten_axes_vector(shape.size() - n_axis);
            std::iota(flatten_axes_vector.begin(), flatten_axes_vector.end(), 0);
            AxisVector flatten_axes = AxisVector(flatten_axes_vector);
            Shape reshape_shape(shape.begin() + m_begin_norm_axis, shape.end());
            size_t reshape_size = shape_size(reshape_shape);
            auto flatten_d_scale =
                make_shared<op::Reshape>(d_scale, flatten_axes, Shape{reshape_size});
            auto flatten_d_bias =
                make_shared<op::Reshape>(d_bias, flatten_axes, Shape{reshape_size});
            retval.emplace_back(flatten_d_scale);
            retval.emplace_back(flatten_d_bias);
        }
        else
        {
            retval.emplace_back(d_scale);
            retval.emplace_back(d_bias);
        }
    }
    return retval;
}

shared_ptr<Node> op::LayerNormBackprop::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() < 2 || new_args.size() > 5)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    if (new_args.size() == 5)
    {
        return make_shared<LayerNormBackprop>(new_args.at(0),
                                              new_args.at(1),
                                              new_args.at(2),
                                              new_args.at(3),
                                              new_args.at(4),
                                              m_begin_norm_axis,
                                              m_epsilon);
    }
    else if (new_args.size() == 4)
    {
        return make_shared<LayerNormBackprop>(new_args.at(0),
                                              new_args.at(1),
                                              new_args.at(2),
                                              new_args.at(3),
                                              m_begin_norm_axis,
                                              m_epsilon);
    }
    else if (new_args.size() == 3)
    {
        return make_shared<LayerNormBackprop>(
            new_args.at(0), new_args.at(1), new_args.at(2), m_begin_norm_axis, m_epsilon);
    }
    else
    {
        return make_shared<LayerNormBackprop>(
            new_args.at(0), new_args.at(1), m_begin_norm_axis, m_epsilon);
    }
}

void op::LayerNormBackprop::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    const PartialShape& data_shape = get_input_partial_shape(0);
    Rank data_rank = data_shape.rank();
    int64_t d_rank = -1;
    int64_t n_axis = -1;
    if (data_rank.is_static())
    {
        d_rank = data_rank.get_length();
        n_axis = m_begin_norm_axis >= 0 ? m_begin_norm_axis : d_rank + m_begin_norm_axis;
        NODE_VALIDATION_CHECK(
            this, n_axis >= 0 && n_axis < d_rank, "begin_norm_axis is out of range");

        const PartialShape& delta_shape = get_input_partial_shape(1);
        Rank delta_rank = delta_shape.rank();
        NODE_VALIDATION_CHECK(this,
                              delta_rank.is_dynamic() || delta_rank.get_length() == d_rank,
                              "Delta rank is incorrect");

        if (m_use_stats)
        {
            const PartialShape& mean_shape = get_input_partial_shape(2);
            const PartialShape& var_shape = get_input_partial_shape(3);
            Rank mean_rank = mean_shape.rank();
            Rank var_rank = var_shape.rank();
            if (mean_rank.is_static() && var_rank.is_static())
            {
                int64_t m_rank = mean_rank.get_length();
                int64_t v_rank = var_rank.get_length();
                NODE_VALIDATION_CHECK(this,
                                      m_rank == v_rank && m_rank == n_axis,
                                      "Mean and/or variance rank is incorrect");
            }
        }

        if (m_use_affine)
        {
            const PartialShape& scale_shape = get_input_partial_shape(m_use_stats ? 4 : 2);
            Rank scale_rank = scale_shape.rank();
            if (scale_rank.is_static())
            {
                int64_t s_rank = scale_rank.get_length();
                NODE_VALIDATION_CHECK(
                    this, (s_rank == (d_rank - n_axis)) || s_rank == 1, "Scale rank is incorrect");
            }
        }
    }

    if (m_use_affine)
    {
        set_output_size(3);
        // output shape: data_shape[begin_norm_axis:]
        if (d_rank > 0)
        {
            std::vector<Dimension> affine_dim;
            for (int64_t i = n_axis; i < d_rank; i++)
            {
                affine_dim.emplace_back(data_shape[i]);
            }
            PartialShape affine_shape(affine_dim);
            set_output_type(1, input_element_type, affine_shape);
            set_output_type(2, input_element_type, affine_shape);
        }
        else // set shape to dynamic
        {
            set_output_type(1, input_element_type, PartialShape::dynamic());
            set_output_type(2, input_element_type, PartialShape::dynamic());
        }
    }
    PartialShape norm_shape{data_shape};
    set_output_type(0, input_element_type, norm_shape);
}
