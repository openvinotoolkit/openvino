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
#include "op/convolution.hpp"
#include "op/group_conv.hpp"
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
    shared_ptr<Node> op_cast(shared_ptr<op::v1::Add> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Add, op::v1::Add>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::AvgPool> node)
    {
        auto const input_arg = node->input_value(0);
        const auto ceil_mode = static_cast<bool>(node->get_rounding_type());
        const auto include_padding_in_avg_computation = !node->get_exclude_pad();
        const auto pad_type = node->get_auto_pad();
        const auto padding_below = node->get_pads_begin();
        const auto padding_above = node->get_pads_end();
        const auto window_movement_strides = node->get_strides();
        const auto window_shape = node->get_kernel();

        auto replacement_node = make_shared<op::v0::AvgPool>(input_arg,
                                                             window_shape,
                                                             window_movement_strides,
                                                             padding_below,
                                                             padding_above,
                                                             include_padding_in_avg_computation,
                                                             pad_type,
                                                             ceil_mode);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Convolution> node)
    {
        const auto data_arg = node->input_value(0);
        const auto filters_arg = node->input_value(1);
        const auto strides = node->get_strides();
        const size_t num_spatial_dims = strides.size();
        auto replacement_node = make_shared<op::v0::Convolution>(data_arg,
                                                                 filters_arg,
                                                                 node->get_strides(),
                                                                 node->get_dilations(),
                                                                 node->get_pads_begin(),
                                                                 node->get_pads_end(),
                                                                 Strides(num_spatial_dims, 1),
                                                                 node->get_auto_pad());
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::ConvolutionBackpropData> node)
    {
        const auto data_arg = node->input_value(0);
        const auto filters_arg = node->input_value(1);

        auto data_pshape = data_arg.get_partial_shape();
        auto filters_pshape = filters_arg.get_partial_shape();

        NGRAPH_CHECK(data_pshape.rank().is_static() && data_pshape[0].is_static() &&
                         filters_pshape.rank().is_static() && filters_pshape[1].is_static(),
                     "Unable to convert ConvolutionBackpropData:v1 to ConvolutionBackpropData:v0 "
                     "if data shape N and filters shape C dimensions are not static. Node: ",
                     *node);

        const size_t num_spatial_dims = data_pshape.rank().get_length() - 2;

        const PartialShape output_pshape{node->get_output_partial_shape(0)};
        NGRAPH_CHECK(output_pshape.is_static(),
                     "Unable to convert ConvolutionBackpropData:v1 to ConvolutionBackpropData:v0 "
                     "if output shape is dynamic. Node: ",
                     *node);
        Shape output_shape = output_pshape.to_shape();

        auto replacement_node =
            make_shared<op::v0::ConvolutionBackpropData>(output_shape,
                                                         filters_arg,
                                                         data_arg,
                                                         node->get_strides(),
                                                         node->get_dilations(),
                                                         node->get_pads_begin(),
                                                         node->get_pads_end(),
                                                         Strides(num_spatial_dims, 1));
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Divide> node)
    {
        const auto input_arg0 = node->input_value(0);
        const auto input_arg1 = node->input_value(1);
        const auto autob = node->get_autob();
        const bool pydiv = node->is_pythondiv();
        auto replacement_node = make_shared<op::v0::Divide>(input_arg0, input_arg1, pydiv, autob);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Equal> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Equal, op::v1::Equal>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Greater> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Greater, op::v1::Greater>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::GreaterEqual> node)
    {
        return op_cast_binary_elementwise_node<op::v0::GreaterEq, op::v1::GreaterEqual>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::GroupConvolution> node)
    {
        const auto data_arg = node->input_value(0);
        const auto filters_arg = node->input_value(1);
        const auto strides = node->get_strides();
        const size_t num_spatial_dims = strides.size();
        auto replacement_node = make_shared<op::v0::GroupConvolution>(data_arg,
                                                                      filters_arg,
                                                                      node->get_strides(),
                                                                      node->get_dilations(),
                                                                      node->get_pads_begin(),
                                                                      node->get_pads_end(),
                                                                      Strides(num_spatial_dims, 1),
                                                                      node->get_auto_pad());
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::GroupConvolutionBackpropData> node)
    {
        const auto data_arg = node->input_value(0);
        const auto filters_arg = node->input_value(1);

        NGRAPH_CHECK(data_arg.get_partial_shape().is_static(),
                     "Unable to convert GroupConvolutionBackpropData:1 to "
                     "GroupConvolutionBackpropData:0 with dynamic data shape. Node: ",
                     *node);

        NGRAPH_CHECK(filters_arg.get_partial_shape().is_static(),
                     "Unable to convert GroupConvolutionBackpropData:1 to "
                     "GroupConvolutionBackpropData:0 with dynamic filters shape. Node: ",
                     *node);

        auto filters_shape = filters_arg.get_shape();
        const size_t groups = filters_shape.at(0);

        const PartialShape output_pshape{node->get_output_partial_shape(0)};
        NGRAPH_CHECK(output_pshape.is_static(),
                     "Unable to convert GroupConvolutionBackpropData:v1 to "
                     "GroupConvolutionBackpropData:v0 "
                     "if output_shape is dynamic. Node: ",
                     *node);
        Shape output_shape = output_pshape.to_shape();

        // Convert filters data layout from [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
        // into [C x M/group x k1 x k2 x ... x kn]
        filters_shape.erase(filters_shape.begin());
        filters_shape[0] *= groups;

        auto reshaped_filters = builder::opset1::reshape(node->input_value(1), filters_shape);

        auto replacement_node = make_shared<op::v0::GroupConvolutionBackpropData>(
            op::Constant::create(data_arg.get_element_type(), output_shape, {0}),
            reshaped_filters,
            data_arg,
            node->get_strides(),
            node->get_dilations(),
            node->get_pads_begin(),
            node->get_pads_end(),
            groups);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Less> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Less, op::v1::Less>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::LessEqual> node)
    {
        return op_cast_binary_elementwise_node<op::v0::LessEq, op::v1::LessEqual>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::LogicalXor> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Xor, op::v1::LogicalXor>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Maximum> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Maximum, op::v1::Maximum>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Minimum> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Minimum, op::v1::Minimum>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Multiply> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Multiply, op::v1::Multiply>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::NotEqual> node)
    {
        return op_cast_binary_elementwise_node<op::v0::NotEqual, op::v1::NotEqual>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Power> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Power, op::v1::Power>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Select> node)
    {
        ngraph::pass::ImplicitBroadcastElimination().run_on_node(node);
        auto replacement_node = make_shared<op::v0::Select>(
            node->input_value(0), node->input_value(1), node->input_value(2));
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Subtract> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Subtract, op::v1::Subtract>(node);
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
