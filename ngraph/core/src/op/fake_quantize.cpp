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

#include <memory>

#include "fake_quantize.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

NGRAPH_RTTI_DEFINITION(op::FakeQuantize, "FakeQuantize", 0);

op::FakeQuantize::FakeQuantize(const Output<Node>& data,
                               const Output<Node>& input_low,
                               const Output<Node>& input_high,
                               const Output<Node>& output_low,
                               const Output<Node>& output_high,
                               size_t levels,
                               const AutoBroadcastSpec& auto_broadcast)
    : FusedOp({data, input_low, input_high, output_low, output_high})
    , m_levels(levels)
    , m_auto_broadcast(auto_broadcast)
{
    constructor_validate_and_infer_types();
}

void op::FakeQuantize::validate_and_infer_types()
{
    PartialShape data_pshape = get_input_partial_shape(0);

    for (auto i = 1; i <= 4; i++)
    {
        if (m_auto_broadcast.m_type == op::AutoBroadcastType::NONE)
        {
            NODE_VALIDATION_CHECK(this,
                                  PartialShape::merge_into(data_pshape, get_input_partial_shape(i)),
                                  "Argument shapes are inconsistent.");
        }
        else if (m_auto_broadcast.m_type == op::AutoBroadcastType::NUMPY ||
                 m_auto_broadcast.m_type == op::AutoBroadcastType::PDPD)
        {
            NODE_VALIDATION_CHECK(this,
                                  PartialShape::broadcast_merge_into(
                                      data_pshape, get_input_partial_shape(i), m_auto_broadcast),
                                  "Argument shapes are inconsistent.");
        }
        else
        {
            NODE_VALIDATION_CHECK(this, false, "Unsupported auto broadcast specification");
        }
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool ngraph::op::v0::FakeQuantize::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("levels", m_levels);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

OutputVector op::FakeQuantize::decompose_op() const
{
    Output<Node> data{input_value(0)};
    Output<Node> input_low{input_value(1)};
    Output<Node> input_high{input_value(2)};
    Output<Node> output_low{input_value(3)};
    Output<Node> output_high{input_value(4)};

    if (m_auto_broadcast.m_type == AutoBroadcastType::NUMPY)
    {
        OutputVector broadcasted_nodes = builder::numpy_broadcast_outputs(
            OutputVector{data, input_low, input_high, output_low, output_high});

        data = broadcasted_nodes.at(0);
        input_low = broadcasted_nodes.at(1);
        input_high = broadcasted_nodes.at(2);
        output_low = broadcasted_nodes.at(3);
        output_high = broadcasted_nodes.at(4);
    }
    else if (m_auto_broadcast.m_type == AutoBroadcastType::PDPD)
    {
        OutputVector broadcasted_nodes = builder::pdpd_broadcast(
            OutputVector{data, input_low, input_high, output_low, output_high},
            m_auto_broadcast.m_axis);

        data = broadcasted_nodes.at(0);
        input_low = broadcasted_nodes.at(1);
        input_high = broadcasted_nodes.at(2);
        output_low = broadcasted_nodes.at(3);
        output_high = broadcasted_nodes.at(4);
    }

    const auto input_data_shape = data.get_shape();
    const auto input_data_type = data.get_element_type();

    const auto levels_minus_one =
        Constant::create(input_data_type,
                         input_data_shape,
                         vector<size_t>(shape_size(input_data_shape), m_levels - 1));

    // map the number of quantization levels to the nGraph's quantization and dequantization scales
    const auto quant_scale = (input_high - input_low) / levels_minus_one;
    const auto dequant_scale = (output_high - output_low) / levels_minus_one;

    // zero_point type needs to match the quantization output type
    const auto zero_point = Constant::create(element::Type_t::i32, data.get_shape(), {0.0});
    const auto axes = get_default_order(input_data_shape);

    // clip the input data to the range <input_low;input_high>
    data =
        std::make_shared<op::Minimum>(input_high, std::make_shared<op::Maximum>(input_low, data));

    // shift the input data so that it contains only positive values (and zeros)
    data = data - input_low;

    shared_ptr<Node> quantized_data =
        make_shared<op::Quantize>(data,
                                  quant_scale,
                                  zero_point,
                                  element::Type_t::i32,
                                  axes,
                                  op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN);

    quantized_data = make_shared<op::Convert>(quantized_data, input_data_type);

    // dequantization without using the Dequantize op (just a multiplication by the dequant_scale)
    const auto dequantized_data = quantized_data * dequant_scale;

    // shift the results so that they fall into the <output_low;output_high> range
    return {dequantized_data + output_low};
}

shared_ptr<Node> op::FakeQuantize::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<FakeQuantize>(new_args.at(0), // X
                                     new_args.at(1), // input_low
                                     new_args.at(2), // input_high
                                     new_args.at(3), // output_low
                                     new_args.at(4), // output_high
                                     m_levels,
                                     m_auto_broadcast);
}
