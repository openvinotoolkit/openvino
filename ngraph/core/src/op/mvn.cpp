// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include "itt.hpp"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/mvn.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V0 ------------------------------

NGRAPH_SUPPRESS_DEPRECATED_START

NGRAPH_RTTI_DEFINITION(op::v0::MVN, "MVN", 0);

op::MVN::MVN()
    : FusedOp()
    , m_across_channels()
    , m_normalize_variance()
    , m_reduction_axes()
{
}

op::MVN::MVN(const Output<Node>& data, bool across_channels, bool normalize_variance, double eps)
    : FusedOp({data})
    , m_eps{eps}
    , m_across_channels{across_channels}
    , m_normalize_variance{normalize_variance}
{
    constructor_validate_and_infer_types();
}

op::MVN::MVN(const Output<Node>& data, AxisSet reduction_axes, bool normalize_variance, double eps)
    : FusedOp({data})
    , m_eps{eps}
    , m_across_channels{false}
    , m_normalize_variance{normalize_variance}
    , m_reduction_axes{reduction_axes}
{
    constructor_validate_and_infer_types();
    const size_t chanelAxis = 1;
    m_across_channels = (m_reduction_axes.count(chanelAxis) > 0);
}

// decompose_op() relies on knowing the data type of input data which might
// not be available at shape inference time. So do direct shape inference
// instead of relying on op decomposition.
void op::MVN::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_MVN_validate_and_infer_types);
    // if m_across_channels is true we should calculate mean and variance per batch
    // else we calculate these per channel
    if (m_reduction_axes.empty() && input_value(0).get_partial_shape().rank().is_static())
    {
        AxisSet reduction_axes;
        size_t start_axis = m_across_channels ? 1 : 2;
        for (int64_t i = start_axis; i < input_value(0).get_partial_shape().rank().get_length();
             ++i)
        {
            reduction_axes.insert(i);
        }
        set_reduction_axes(reduction_axes);
    }

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

OutputVector op::MVN::decompose_op() const
{
    auto data = input_value(0);
    auto data_shape = data.get_shape(); // assume that data has n and c channels.

    // calculate mean normalization
    auto mean = builder::opset1::mean(data, m_reduction_axes);
    auto mean_normalization = std::make_shared<op::v1::Subtract>(
        data, builder::opset1::make_broadcast(mean, data_shape, m_reduction_axes));

    if (!m_normalize_variance)
    {
        return {mean_normalization};
    }
    else
    {
        // calculate variance
        auto variance = builder::opset1::variance(data, m_reduction_axes);
        // add epsilon
        auto eps_node = op::Constant::create(
            data.get_element_type(), Output<Node>(variance).get_shape(), vector<double>{m_eps});
        variance = std::make_shared<op::Sqrt>(std::make_shared<op::v1::Add>(variance, eps_node));
        return OutputVector{std::make_shared<op::v1::Divide>(
            mean_normalization,
            builder::opset1::make_broadcast(variance, data_shape, m_reduction_axes))};
    }
}

shared_ptr<Node> op::MVN::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_MVN_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the MVN op but got ",
                          new_args.size());
    return make_shared<MVN>(new_args.at(0), m_reduction_axes, m_normalize_variance, m_eps);
}

bool op::MVN::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_MVN_visit_attributes);
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("across_channels", m_across_channels);
    visitor.on_attribute("normalize_variance", m_normalize_variance);
    visitor.on_attribute("reduction_axes", m_reduction_axes);
    return true;
}

// ------------------------------ V6 ------------------------------

namespace ngraph
{
    template <>
    NGRAPH_API EnumNames<op::MVNEpsMode>& EnumNames<op::MVNEpsMode>::get()
    {
        static auto enum_names =
            EnumNames<op::MVNEpsMode>("op::MVNEpsMode",
                                      {{"OUTSIDE_SQRT", op::MVNEpsMode::OUTSIDE_SQRT},
                                       {"INSIDE_SQRT", op::MVNEpsMode::INSIDE_SQRT}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::MVNEpsMode>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::MVNEpsMode& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph

NGRAPH_RTTI_DEFINITION(op::v6::MVN, "MVN", 6);

op::v6::MVN::MVN(const Output<Node>& data,
                 const Output<Node>& reduction_axes,
                 bool normalize_variance,
                 float eps,
                 MVNEpsMode eps_mode)
    : Op({data, reduction_axes})
    , m_normalize_variance{normalize_variance}
    , m_eps{eps}
    , m_eps_mode{eps_mode}
{
    constructor_validate_and_infer_types();
}

void op::v6::MVN::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_MVN_validate_and_infer_types);
    const auto data = get_input_partial_shape(0);
    const auto axes = get_input_partial_shape(1);

    if (axes.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              is_vector(axes.to_shape()),
                              "Expected 1D tensor for the 'axes' input. Got: ",
                              axes);

        NODE_VALIDATION_CHECK(
            this,
            data.rank().is_dynamic() ||
                data.rank().get_length() >= static_cast<int64_t>(axes.get_shape()[0]),
            "Expected rank for the 'data' input to be higher than axes shape. Got: ",
            data);
    }

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v6::MVN::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_MVN_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 2,
                          "Expected 2 element in new_args for the MVN op but got ",
                          new_args.size());
    return make_shared<op::v6::MVN>(
        new_args.at(0), new_args.at(1), m_normalize_variance, m_eps, m_eps_mode);
}

bool op::v6::MVN::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_MVN_visit_attributes);
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("normalize_variance", m_normalize_variance);
    visitor.on_attribute("eps_mode", m_eps_mode);
    return true;
}
