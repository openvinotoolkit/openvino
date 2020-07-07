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

#include "ngraph/pass/constant_to_broadcast.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

bool pass::ConstantToBroadcast::run_on_node(shared_ptr<Node> node)
{
    const size_t minimum_size_of_interest = 32;
    bool modified = false;
    if (node->description() == "Constant")
    {
        auto constant = static_pointer_cast<op::Constant>(node);
        size_t size = shape_size(constant->get_shape());
        if (size > minimum_size_of_interest)
        {
            if (constant->get_all_data_elements_bitwise_identical())
            {
                auto scalar_constant = make_shared<op::Constant>(
                    constant->get_element_type(), Shape{}, constant->get_data_ptr());
                AxisSet broadcast_axes;
                for (size_t i = 0; i < constant->get_output_shape(0).size(); i++)
                {
                    broadcast_axes.insert(i);
                }
                auto broadcast = make_shared<op::v0::Broadcast>(
                    scalar_constant, constant->get_output_shape(0), broadcast_axes);
                replace_node(constant, broadcast);
            }
        }
    }
    return modified;
}
