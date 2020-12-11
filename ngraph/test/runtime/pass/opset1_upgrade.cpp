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
#include "opset1_upgrade.hpp"

#include <functional>
#include <iterator>
#include <limits>
#include <numeric>

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/provenance.hpp"
#include "op/avg_pool.hpp"
#include "op/convolution.hpp"
#include "op/group_conv.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

namespace opset1_upgrade
{
    template <typename OpV0, typename OpV1>
    shared_ptr<Node> op_cast_binary_elementwise_node(const shared_ptr<OpV0>& node)
    {
        const auto autob = node->get_autob();
        auto replacement_node =
            make_shared<OpV1>(node->input_value(0), node->input_value(1), autob);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    // Default is that we didn nothing
    shared_ptr<Node> op_cast(shared_ptr<Node> node) { return nullptr; }
    shared_ptr<Node> op_cast(shared_ptr<op::v0::Multiply> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Multiply, op::v1::Multiply>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v0::ConvolutionBackpropData> node)
    {
        auto data_batch_shape = node->get_data_batch_shape();
        auto strides = node->get_window_movement_strides_forward();
        auto dilations = node->get_window_dilation_strides_forward();
        auto pads_begin = node->get_padding_below_forward();
        auto pads_end = node->get_padding_above_forward();
        auto data_dilation_strides = node->get_data_dilation_strides_forward();

        bool is_dds_valid = all_of(data_dilation_strides.begin(),
                                   data_dilation_strides.end(),
                                   [](size_t value) { return value == 1; });

        NGRAPH_CHECK(is_dds_valid,
                     "Unable to convert ConvolutionBackpropData:0 to ConvolutionBackpropData:1 "
                     "with data dilation strides "
                     "other than `1`. Node: ",
                     *node);

        auto replacement_node = make_shared<op::v1::ConvolutionBackpropData>(
            node->input_value(1), // data
            node->input_value(0), // filters
            op::Constant::create(
                element::i64,
                Shape{data_batch_shape.size() - 2},
                vector<size_t>(data_batch_shape.begin() + 2, data_batch_shape.end())),
            strides,
            pads_begin,
            pads_end,
            dilations);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v0::GroupConvolution> node)
    {
        auto strides = node->get_window_movement_strides();
        auto dilations = node->get_window_dilation_strides();
        auto pads_begin = node->get_padding_below();
        auto pads_end = node->get_padding_above();
        auto data_dilation_strides = node->get_data_dilation_strides();
        auto auto_pad = node->get_pad_type();

        bool is_dds_valid = all_of(data_dilation_strides.begin(),
                                   data_dilation_strides.end(),
                                   [](size_t value) { return value == 1; });

        NGRAPH_CHECK(is_dds_valid,
                     "Unable to convert GroupConvolution:0 to GroupConvolution:1"
                     "with data dilation strides other than `1`. Node: ",
                     *node);

        shared_ptr<Node> replacement_node;
        if (node->has_groups_in_filters())
        {
            replacement_node = make_shared<op::v1::GroupConvolution>(node->input_value(0),
                                                                     node->input_value(1),
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     auto_pad);
        }
        else
        {
            NGRAPH_CHECK(node->get_input_partial_shape(1).is_static(),
                         "Unable to convert GroupConvolution:0 to GroupConvolution:1"
                         "with dynamic filters shape. Node: ",
                         *node);

            auto filters_shape = node->get_input_shape(1);
            auto groups = node->get_groups();
            filters_shape[0] /= groups;
            filters_shape.insert(filters_shape.begin(), groups);

            auto reshaped_filters = builder::opset1::reshape(node->input_value(1), filters_shape);

            replacement_node = make_shared<op::v1::GroupConvolution>(node->input_value(0),
                                                                     reshaped_filters,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     auto_pad);
        }
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v0::GroupConvolutionBackpropData> node)
    {
        const auto strides = node->get_window_movement_strides();
        const auto dilations = node->get_window_dilation_strides();
        const auto pads_begin = node->get_padding_below();
        const auto pads_end = node->get_padding_above();

        const auto data_batch_pshape = node->get_input_partial_shape(0);
        const auto filters_pshape = node->get_input_partial_shape(1);

        NGRAPH_CHECK(data_batch_pshape.is_static(),
                     "Unable to convert GroupConvolutionBackpropData:0 to "
                     "GroupConvolutionBackpropData:1 with dynamic data_batch shape. Node: ",
                     *node);
        NGRAPH_CHECK(filters_pshape.is_static(),
                     "Unable to convert GroupConvolutionBackpropData:0 to "
                     "GroupConvolutionBackpropData:1 with dynamic filters shape. Node: ",
                     *node);

        auto data_batch_shape = data_batch_pshape.to_shape();
        // Remove N, C from output shape to preserve only spatial dimentions.
        data_batch_shape.erase(std::begin(data_batch_shape),
                               std::next(std::begin(data_batch_shape), 2));
        auto filters_shape = filters_pshape.to_shape();
        auto groups = node->get_groups();

        filters_shape[0] /= groups;
        filters_shape.insert(filters_shape.begin(), groups);
        auto reshaped_filters = builder::opset1::reshape(node->input_value(1), filters_shape);

        auto replacement_node = make_shared<op::v1::GroupConvolutionBackpropData>(
            node->input_value(2),
            reshaped_filters,
            op::Constant::create(element::i64, Shape{data_batch_shape.size()}, data_batch_shape),
            strides,
            pads_begin,
            pads_end,
            dilations);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Xor> node)
    {
        auto replacement_node = make_shared<op::v1::LogicalXor>(
            node->input_value(0), node->input_value(1), node->get_autob());
        replace_node(node, replacement_node);
        return replacement_node;
    }

    using DispatchMap = map<NodeTypeInfo, std::function<bool(shared_ptr<Node> node)>>;

    template <typename T>
    bool op_cast_thunk(shared_ptr<Node> node)
    {
        auto upgraded_node = op_cast(as_type_ptr<T>(node));
        if (upgraded_node)
        {
            if (ngraph::get_provenance_enabled())
            {
                const std::string provenance_tag =
                    "<Opset1_Upgrade (v0 " + std::string(node->get_type_name()) + ")>";
                upgraded_node->add_provenance_tags_above(node->input_values(), {provenance_tag});
            }
            return true;
        }
        return false;
    }

    DispatchMap& get_dispatch_map()
    {
        NGRAPH_SUPPRESS_DEPRECATED_START
        static DispatchMap dispatch_map{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, op_cast_thunk<NAMESPACE::NAME>},
#include "opset0_tbl.hpp"
#undef NGRAPH_OP
        };
        return dispatch_map;
        NGRAPH_SUPPRESS_DEPRECATED_END
    }
} // namespace opset1_upgrade

bool pass::Opset1Upgrade::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;
    auto& dispatch_map = opset1_upgrade::get_dispatch_map();
    auto it = dispatch_map.find(node->get_type_info());
    if (it != dispatch_map.end())
    {
        modified = it->second(node);
    }
    return modified;
}
