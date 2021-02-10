//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "itt.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/fake_quantize.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

NGRAPH_RTTI_DEFINITION(op::FakeQuantize, "FakeQuantize", 0);

op::FakeQuantize::FakeQuantize()
    : Op()
    , m_levels()
{
}

op::FakeQuantize::FakeQuantize(const Output<Node>& data,
                               const Output<Node>& input_low,
                               const Output<Node>& input_high,
                               const Output<Node>& output_low,
                               const Output<Node>& output_high,
                               size_t levels,
                               const AutoBroadcastSpec& auto_broadcast)
    : Op({data, input_low, input_high, output_low, output_high})
    , m_levels(levels)
    , m_auto_broadcast(auto_broadcast)
{
    constructor_validate_and_infer_types();
}

void op::FakeQuantize::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_FakeQuantize_validate_and_infer_types);
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
    NGRAPH_OP_SCOPE(v0_FakeQuantize_visit_attributes);
    visitor.on_attribute("levels", m_levels);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

shared_ptr<Node> op::FakeQuantize::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_FakeQuantize_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<FakeQuantize>(new_args.at(0), // X
                                     new_args.at(1), // input_low
                                     new_args.at(2), // input_high
                                     new_args.at(3), // output_low
                                     new_args.at(4), // output_high
                                     m_levels,
                                     m_auto_broadcast);
}
