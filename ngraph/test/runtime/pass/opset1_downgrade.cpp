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

#include <algorithm>

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/provenance.hpp"
#include "ngraph/validation_util.hpp"
#include "opset1_downgrade.hpp"

using namespace std;
using namespace ngraph;

namespace opset1_downgrade
{
    shared_ptr<Node> op_cast(shared_ptr<op::v3::Broadcast> node)
    {
        const auto data = node->input_value(0).get_node_shared_ptr();
        const auto target_shape = node->input_value(1).get_node_shared_ptr();

        shared_ptr<Node> replacement_node;
        switch (node->get_broadcast_spec().m_type)
        {
        case op::BroadcastType::BIDIRECTIONAL:
        {
            const auto const_filled_with_ones = make_shared<op::v1::Broadcast>(
                op::Constant::create(data->get_element_type(), {}, {1}), target_shape);
            if (const_filled_with_ones->get_element_type() == element::Type_t::boolean)
            {
                replacement_node = make_shared<op::v1::LogicalOr>(data, const_filled_with_ones);
            }
            else
            {
                replacement_node = make_shared<op::v1::Multiply>(data, const_filled_with_ones);
            }
            break;
        }
        case op::BroadcastType::EXPLICIT:
        {
            const auto axes_mapping = node->input_value(2).get_node_shared_ptr();
            replacement_node = make_shared<op::v1::Broadcast>(
                data, target_shape, axes_mapping, op::AutoBroadcastType::EXPLICIT);
            break;
        }
        case op::BroadcastType::NUMPY:
        {
            replacement_node =
                make_shared<op::v1::Broadcast>(data, target_shape, op::AutoBroadcastType::NUMPY);
            break;
        }
        case op::BroadcastType::PDPD:
        {
            op::AutoBroadcastSpec broadcast_spec;
            broadcast_spec.m_type = op::AutoBroadcastType::PDPD;
            broadcast_spec.m_axis = node->get_broadcast_spec().m_axis;
            replacement_node = make_shared<op::v1::Broadcast>(data, target_shape, broadcast_spec);
            break;
        }
        default:
        {
            NGRAPH_CHECK(
                true,
                "Not supported broadcast type during Broadcast:v3 to Broadcast:v1 conversion. ",
                "Node: ",
                *node);
        }
        }
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v3::TopK> node)
    {
        const auto data = node->input_value(0);
        const auto k = node->input_value(1);
        const auto replacement_node = make_shared<op::v1::TopK>(data,
                                                                k,
                                                                node->get_axis(),
                                                                node->get_mode(),
                                                                node->get_sort_type(),
                                                                node->get_index_element_type());
        replace_node(node, replacement_node);
        return replacement_node;
    }

    using DispatchMap = map<NodeTypeInfo, std::function<bool(shared_ptr<Node> node)>>;

    template <typename T>
    bool op_cast_thunk(shared_ptr<Node> node)
    {
        auto downgraded_node = op_cast(as_type_ptr<T>(node));
        if (downgraded_node)
        {
            if (ngraph::get_provenance_enabled())
            {
                const std::string provenance_tag =
                    "<Opset1_Downgrade (v3 " + std::string(node->get_type_name()) + ")>";
                downgraded_node->add_provenance_tags_above(node->input_values(), {provenance_tag});
            }
            return true;
        }
        return false;
    }

    DispatchMap& get_dispatch_map()
    {
        static DispatchMap dispatch_map{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, op_cast_thunk<NAMESPACE::NAME>},
            NGRAPH_OP(Broadcast, op::v3) NGRAPH_OP(TopK, op::v3)
#undef NGRAPH_OP
        };
        return dispatch_map;
    }
} // namespace opset1_downgrade

bool pass::Opset1Downgrade::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;
    auto& dispatch_map = opset1_downgrade::get_dispatch_map();
    auto it = dispatch_map.find(node->get_type_info());
    if (it != dispatch_map.end())
    {
        modified = it->second(node);
    }
    return modified;
}
