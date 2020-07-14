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
#include "ngraph/ops.hpp"
#include "ngraph/provenance.hpp"
#include "op/and.hpp"
#include "op/atan2.hpp"
#include "op/avg_pool.hpp"

using namespace std;
using namespace ngraph;

namespace
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
    shared_ptr<Node> op_cast(shared_ptr<op::Add> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Add, op::v1::Add>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v0::And> node)
    {
        return op_cast_binary_elementwise_node<op::v0::And, op::v1::LogicalAnd>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Broadcast> node)
    {
        auto replacement_node = ngraph::builder::opset1::make_broadcast(
            node->input_value(0), node->get_broadcast_shape(), node->get_broadcast_axes());
        replace_node(node, replacement_node.get_node_shared_ptr());
        return replacement_node.get_node_shared_ptr();
    }

    shared_ptr<Node> op_cast(shared_ptr<op::BroadcastLike> node) { return nullptr; }
    shared_ptr<Node> op_cast(shared_ptr<op::Convolution> node)
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
                     "Unable to convert Convolution:0 to Convolution:1 with data dilation strides "
                     "other than `1`. Node: ",
                     *node);

        auto replacement_node = make_shared<op::v1::Convolution>(node->input_value(0),
                                                                 node->input_value(1),
                                                                 strides,
                                                                 pads_begin,
                                                                 pads_end,
                                                                 dilations,
                                                                 auto_pad);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::ConvolutionBackpropData> node)
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

    shared_ptr<Node> op_cast(shared_ptr<op::Divide> node)
    {
        const auto autob = node->get_autob();
        const bool pydiv = node->is_pythondiv();
        auto replacement_node =
            make_shared<op::v1::Divide>(node->input_value(0), node->input_value(1), pydiv, autob);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Reshape> node)
    {
        shared_ptr<Node> replacement_node =
            builder::opset1::reshape(node->input_value(0), node->get_reshape_output_shape());
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Equal> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Equal, op::v1::Equal>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Gather> node)
    {
        int64_t axis = node->get_axis();

        auto axis_node = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{axis});
        auto replacement_node =
            make_shared<op::v1::Gather>(node->input_value(0), node->input_value(1), axis_node);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Greater> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Greater, op::v1::Greater>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::GreaterEq> node)
    {
        return op_cast_binary_elementwise_node<op::v0::GreaterEq, op::v1::GreaterEqual>(node);
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

            auto reshaped_filters = builder::reshape(node->input_value(1), filters_shape);

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
        auto reshaped_filters = builder::reshape(node->input_value(1), filters_shape);

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

    shared_ptr<Node> op_cast(shared_ptr<op::Less> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Less, op::v1::Less>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::LessEq> node)
    {
        return op_cast_binary_elementwise_node<op::v0::LessEq, op::v1::LessEqual>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Max> node)
    {
        bool keep_dims = false;
        auto replacement_node =
            make_shared<op::v1::ReduceMax>(node->input_value(0), node->input_value(1), keep_dims);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Maximum> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Maximum, op::v1::Maximum>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::MaxPool> node)
    {
        auto rounding_type =
            node->get_ceil_mode() ? op::RoundingType::CEIL : op::RoundingType::FLOOR;
        auto auto_pad = node->get_pad_type();
        auto pads_begin = node->get_padding_below();
        auto pads_end = node->get_padding_above();
        auto strides = node->get_window_movement_strides();
        auto kernel = node->get_window_shape();

        auto replacement_node = make_shared<op::v1::MaxPool>(
            node->input_value(0), strides, pads_begin, pads_end, kernel, rounding_type, auto_pad);
#if defined(__clang__) && __clang_major__ == 3
        // There are some really by clang 3.9 bugs
        if (node->get_ceil_mode())
        {
            replacement_node->set_rounding_type(op::RoundingType::CEIL);
        }
        else
        {
            replacement_node->set_rounding_type(op::RoundingType::FLOOR);
        }
#endif
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Min> node)
    {
        bool keep_dims = false;
        auto replacement_node =
            make_shared<op::v1::ReduceMin>(node->input_value(0), node->input_value(1), keep_dims);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Minimum> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Minimum, op::v1::Minimum>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Multiply> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Multiply, op::v1::Multiply>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Not> node)
    {
        auto replacement_node = make_shared<op::v1::LogicalNot>(node->input_value(0));
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::NotEqual> node)
    {
        return op_cast_binary_elementwise_node<op::v0::NotEqual, op::v1::NotEqual>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::OneHot> node)
    {
        const auto indices = node->input_value(0).get_node_shared_ptr();
        const auto one_hot_axis = node->get_one_hot_axis();

        const auto output_pshape = node->get_output_partial_shape(0);
        NGRAPH_CHECK(output_pshape[one_hot_axis].is_static(),
                     "OneHot:v0 one hot axis dimension must be static ",
                     *node);
        const auto depth = output_pshape[one_hot_axis].get_length();
        const auto depth_node = op::Constant::create(element::i64, Shape{}, {depth});

        const auto on_value = op::Constant::create(element::i64, Shape{}, {1});
        const auto off_value = op::Constant::create(element::i64, Shape{}, {0});

        auto replacement_node =
            make_shared<op::v1::OneHot>(indices, depth_node, on_value, off_value, one_hot_axis);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Or> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Or, op::v1::LogicalOr>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Pad> node)
    {
        auto padding_below = node->get_padding_below();
        auto pads_begin_node =
            make_shared<op::Constant>(element::i64, Shape{padding_below.size()}, padding_below);
        auto padding_above = node->get_padding_above();
        auto pads_end_node =
            make_shared<op::Constant>(element::i64, Shape{padding_above.size()}, padding_above);

        auto replacement_node = make_shared<op::v1::Pad>(node->input_value(0),
                                                         pads_begin_node,
                                                         pads_end_node,
                                                         node->input_value(1),
                                                         node->get_pad_mode());

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Power> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Power, op::v1::Power>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Product> node)
    {
        bool keep_dims = false;
        auto replacement_node =
            make_shared<op::v1::ReduceProd>(node->input_value(0), node->input_value(1), keep_dims);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Reverse> node)
    {
        // creates a Constant node from the v0::Reverse reversed_axes attribute
        // and uses it as the second input of v1::Reverse
        const auto reversed_axes = node->get_reversed_axes();

        const auto reversed_axes_constant = op::Constant::create(
            element::i64, Shape{reversed_axes.size()}, reversed_axes.to_vector());

        const auto replacement_node = make_shared<op::v1::Reverse>(
            node->input_value(0), reversed_axes_constant, op::v1::Reverse::Mode::INDEX);

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Select> node)
    {
        auto replacement_node = make_shared<op::v1::Select>(node->input_value(0),
                                                            node->input_value(1),
                                                            node->input_value(2),
                                                            op::AutoBroadcastSpec());
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Softmax> node)
    {
        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant(),
                     "axes parameter is expected to be a static constant");

        AxisSet axes = node->get_axes();

        NGRAPH_CHECK(
            axes.size() == 1,
            "Unable to convert Softmax:0 to Softmax:1 with zero or more than one axis. Node: ",
            *node);

        auto replacement_node =
            make_shared<op::v1::Softmax>(node->input_value(0), axes.to_vector()[0]);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Slice> node)
    {
        const auto data = node->input_value(0);
        const auto begin = op::Constant::create(
            element::i64, Shape{node->get_lower_bounds().size()}, node->get_lower_bounds());
        const auto end = op::Constant::create(
            element::i64, Shape{node->get_upper_bounds().size()}, node->get_upper_bounds());
        const auto strides = op::Constant::create(
            element::i64, Shape{node->get_strides().size()}, node->get_strides());
        int64_t input_size = node->get_lower_bounds().size();

        auto replacement_node = make_shared<op::v1::StridedSlice>(data,
                                                                  begin,
                                                                  end,
                                                                  strides,
                                                                  vector<int64_t>(input_size, 0),
                                                                  vector<int64_t>(input_size, 0));

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Split> node)
    {
        const auto& splits_vec = node->get_splits();
        const auto first_elem = splits_vec.front();

        const bool split_evenly =
            std::all_of(splits_vec.begin(), splits_vec.end(), [first_elem](const size_t split) {
                return split == first_elem;
            });

        std::shared_ptr<Node> replacement_node;
        if (split_evenly)
        {
            replacement_node = make_shared<op::v1::Split>(
                node->input_value(0), node->input_value(1), splits_vec.front());
        }
        else
        {
            const auto split_lengths =
                ngraph::op::Constant::create(element::u64, Shape{splits_vec.size()}, splits_vec);

            replacement_node = make_shared<op::v1::VariadicSplit>(
                node->input_value(0), node->input_value(1), split_lengths);
        }

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Subtract> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Subtract, op::v1::Subtract>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::Sum> node)
    {
        bool keep_dims = false;
        auto replacement_node =
            make_shared<op::v1::ReduceSum>(node->input_value(0), node->input_value(1), keep_dims);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::TopK> node)
    {
        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant(),
                     "parameter k is expected to be a static constant");
        NGRAPH_CHECK(node->input_value(2).get_node_shared_ptr()->is_constant(),
                     "parameter top_k_axis is expected to be a static constant");

        const auto k = node->get_k();
        const auto axis = node->get_top_k_axis();

        std::string sort;
        switch (node->get_sort())
        {
        case op::TopK::SortType::SORT_INDICES: sort = "index"; break;
        case op::TopK::SortType::SORT_VALUES: sort = "value"; break;
        case op::TopK::SortType::NONE: sort = "none"; break;
        }

        std::string mode;
        if (node->get_compute_max())
        {
            mode = "max";
        }
        else
        {
            mode = "min";
        }

        const auto k_constant = op::Constant::create(element::i64, Shape{}, {k});
        auto replacement_node =
            make_shared<op::v1::TopK>(node->input_value(0), k_constant, axis, mode, sort);

        // indices output will be 0, values 1
        vector<int64_t> output_order{1, 0};
        replace_node(node, replacement_node, output_order);
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
        static DispatchMap dispatch_map{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, op_cast_thunk<NAMESPACE::NAME>},
#include "opset0_tbl.hpp"
#undef NGRAPH_OP
        };
        return dispatch_map;
    }
} // namespace

bool pass::Opset1Upgrade::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;
    auto& dispatch_map = get_dispatch_map();
    auto it = dispatch_map.find(node->get_type_info());
    if (it != dispatch_map.end())
    {
        modified = it->second(node);
    }
    return modified;
}
