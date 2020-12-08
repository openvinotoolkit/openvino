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
#include <cstdint>
#include <functional>
#include <numeric>

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/provenance.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/type.hpp"
#include "ngraph/validation_util.hpp"
#include "op/avg_pool.hpp"
#include "pass/implicit_broadcast_elimination.hpp"
#include "pass/opset0_downgrade.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

namespace opset0_downgrade
{
    template <typename OpV0, typename OpV1>
    shared_ptr<Node> op_cast_binary_elementwise_node(const shared_ptr<OpV1>& node)
    {
        const auto input_arg0 = node->input_value(0);
        const auto input_arg1 = node->input_value(1);
        const auto autob = node->get_autob();
        auto replacement_node = make_shared<OpV0>(input_arg0, input_arg1, autob);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    template <typename OpV0, typename OpV1>
    shared_ptr<Node> op_cast_reduction_node(const shared_ptr<OpV1>& node)
    {
        auto replacement_node = make_shared<OpV0>(node->input_value(0), node->input_value(1));
        if (node->get_keep_dims())
        {
            string v1_op_name = string{node->get_type_name()} + ":v1";
            string v0_op_name = string{OpV0{}.get_type_name()} + ":v0";

            NGRAPH_CHECK(node->reduction_axes_constant(),
                         "Unable to convert ",
                         v1_op_name,
                         "to ",
                         v0_op_name,
                         " if reduction axes are not constant (for keep_dims=true). Node: ",
                         *node);
            auto output_pshape = replacement_node->get_output_partial_shape(0);
            NGRAPH_CHECK(output_pshape.is_static(),
                         "Unable to convert ",
                         v1_op_name,
                         "to ",
                         v0_op_name,
                         " if output shape is dynamic (for keep_dims=true). Node: ",
                         *node);
            const auto output_shape = output_pshape.to_shape();
            auto reshaped_output_shape = output_shape;
            for (const auto& axis : node->get_reduction_axes())
            {
                reshaped_output_shape.insert(reshaped_output_shape.begin() + axis, 1);
            }
            auto shape_pattern = op::Constant::create(
                element::u64, {reshaped_output_shape.size()}, reshaped_output_shape);
            auto reshaped_product =
                make_shared<op::v1::Reshape>(replacement_node->output(0), shape_pattern, false);
            return reshaped_product;
        }
        else
        {
            return replacement_node;
        }
    }

    // Default is that we did nothing
    shared_ptr<Node> op_cast(shared_ptr<Node> node) { return nullptr; }
    shared_ptr<Node> op_cast(shared_ptr<op::v1::LogicalXor> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Xor, op::v1::LogicalXor>(node);
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
                    "<Opset0_Downgrade (v1 " + std::string(node->get_type_name()) + ")>";
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
#include "ngraph/opsets/opset1_tbl.hpp"
#undef NGRAPH_OP
        };
        return dispatch_map;
    }
} // namespace opset0_downgrade

bool pass::Opset0Downgrade::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;
    auto& dispatch_map = opset0_downgrade::get_dispatch_map();
    auto it = dispatch_map.find(node->get_type_info());
    if (it != dispatch_map.end())
    {
        modified = it->second(node);
    }
    return modified;
}
