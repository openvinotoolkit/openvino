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

#include <array>
#include <map>
#include <memory>
#include <numeric>
#include <stack>
#include <typeindex>
#include <unordered_map>

#include "batch_fusion.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/fused/batch_mat_mul_transpose.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

std::shared_ptr<Node> set_or_check_if_same(std::shared_ptr<Node> oldn, std::shared_ptr<Node> newn)
{
    if (!oldn)
    {
        return newn;
    }
    else
    {
        if (oldn != newn)
        {
            NGRAPH_DEBUG << " different data nodes";
            return nullptr;
        }
        return oldn;
    }
}

static bool is_trivial_convolution(std::shared_ptr<op::Convolution> conv)
{
    Strides stride_1{1, 1};
    CoordinateDiff pad_0{0, 0};
    return conv->get_window_dilation_strides() == stride_1 &&
           conv->get_data_dilation_strides() == stride_1 && conv->get_padding_above() == pad_0 &&
           conv->get_padding_below() == pad_0;
}

std::shared_ptr<Node> fuse_group_convolution(const std::shared_ptr<Node>& n)
{
    Shape win_size_1{1, 1, 1, 1};
    auto data_label = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 4, 9});
    auto weights_label = std::make_shared<pattern::op::Label>(element::f32, Shape{4, 2, 3});

    auto slice_data = std::make_shared<op::Slice>(
        data_label, Coordinate{0, 0, 0}, Coordinate{1, 2, 9}, Strides{1, 1, 1});
    auto slice_weights = std::make_shared<op::Slice>(
        weights_label, Coordinate{0, 0, 0}, Coordinate{2, 2, 3}, Strides{1, 1, 1});

    auto slice_weights_label =
        std::make_shared<pattern::op::Label>(slice_weights, nullptr, NodeVector{slice_weights});
    auto conv = std::make_shared<op::Convolution>(slice_data, slice_weights_label);
    auto matcher = std::make_shared<pattern::Matcher>(conv);

    NGRAPH_DEBUG << "In simplify_concat (group convolution) for " << n->get_name();

    std::shared_ptr<Node> data;
    std::shared_ptr<Node> weights;

    auto concat = std::static_pointer_cast<op::Concat>(n);
    std::shared_ptr<op::Convolution> sconv;

    NodeVector slices;

    const size_t CHANNEL = 1;
    if (concat->get_concatenation_axis() != CHANNEL)
    {
        NGRAPH_DEBUG << "concatenating on an axis different from channel";
        return {nullptr};
    }

    for (auto arg : n->get_arguments())
    {
        if (!matcher->match(arg))
        {
            NGRAPH_DEBUG << arg->get_name() << " doesn't match";
            return {nullptr};
        }

        sconv = std::static_pointer_cast<op::Convolution>(arg);

        if (arg->get_input_shape(0).size() != 4)
        {
            NGRAPH_DEBUG << "convolution data's rank isn't equal to 4";
            return {nullptr};
        }

        if (!is_trivial_convolution(sconv))
        {
            NGRAPH_DEBUG << arg->get_name() << " isn't trivial convolution";
            return {nullptr};
        }

        auto pattern_map = matcher->get_pattern_map();
        data = set_or_check_if_same(data, pattern_map[data_label]);
        weights = set_or_check_if_same(weights, pattern_map[weights_label]);

        if (!data || !weights)
        {
            NGRAPH_DEBUG << "data or weights nodes are different among slices";
            return {nullptr};
        }

        const size_t IC = 1;
        auto slice = pattern_map[slice_weights_label];
        if (weights->get_shape().at(IC) != slice->get_shape().at(IC))
        {
            slices.push_back(slice);
        }
    }

    // TF-flavoured group convolution needs channels re-arranged
    // MKLDNN requires group slicing to be done on OC
    // MKLDNN [4,2,-]
    // ordering w00 w01 w10 w11 w20 w21 w30 w31 produces g00 g01 g10 g11
    // whereas
    // TF    [2,4,-]
    // ordering w00 w01 w02 w03 w10 w11 w12 w13 produces g00 g10 g01 g11
    const size_t CONCAT_AXIS_OC = 0;
    if (!slices.empty())
    {
        weights = std::make_shared<op::Concat>(slices, CONCAT_AXIS_OC);
    }

    auto new_conv = std::make_shared<op::GroupConvolution>(data,
                                                           weights,
                                                           sconv->get_window_movement_strides(),
                                                           sconv->get_window_dilation_strides(),
                                                           sconv->get_padding_below(),
                                                           sconv->get_padding_above(),
                                                           sconv->get_data_dilation_strides(),
                                                           n->get_arguments().size());

    return move(new_conv);
}

std::shared_ptr<Node> fuse_batch_mat_mul_transpose(const std::shared_ptr<Node>& n)
{
    const int num_op_branches = 2;
    std::shared_ptr<pattern::op::Label> input[num_op_branches];
    std::shared_ptr<op::Reshape> reshape[num_op_branches];
    for (int i = 0; i < num_op_branches; ++i)
    {
        input[i] = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 2, 2});
        auto slice =
            std::make_shared<op::Slice>(input[i], Coordinate{0, 0, 0}, Coordinate{1, 2, 2});
        auto skip = std::make_shared<pattern::op::Skip>(slice, pattern::has_class<op::Reshape>());
        reshape[i] = std::make_shared<op::Reshape>(skip, AxisVector{0, 1, 2}, Shape{2, 2});
    }
    auto dot = std::make_shared<op::Dot>(reshape[0], reshape[1]);
    auto final_reshape = std::make_shared<op::Reshape>(dot, AxisVector{0, 1}, Shape{1, 2, 2});

    auto matcher = std::make_shared<pattern::Matcher>(final_reshape);
    std::shared_ptr<Node> fuse_input[num_op_branches];
    bool transpose[num_op_branches] = {false, false};
    const int num_expected_reshape_with_trans = 3;

    // check each input arg matches the pattern
    for (auto arg : n->get_arguments())
    {
        if (matcher->match(arg))
        {
            auto pattern_map = matcher->get_pattern_map();
            int reshape_count[num_op_branches] = {0, 0};
            // we found a match, determine whether we have to transpose for each input by
            // counting the number of reshapes in each branch, if transpose is applied, there
            // should be 3 reshapes.
            for (int i = 0; i < num_op_branches; ++i)
            {
                auto iter = matcher->get_match_root();
                auto& input_node = pattern_map[input[i]];
                do
                {
                    if (is_type<op::Reshape>(iter))
                    {
                        ++reshape_count[i];
                        if (reshape_count[i] == num_expected_reshape_with_trans)
                        {
                            transpose[i] = true;
                            break;
                        }
                    }
                    // branch to either input 0 or 1 depending on which one we are traversing
                    iter =
                        iter->get_input_size() > 1 ? iter->get_argument(i) : iter->get_argument(0);
                } while (iter != input_node);
            }
            // keep track of the input data, make sure they all match
            for (int i = 0; i < num_op_branches; ++i)
            {
                auto& input_node = pattern_map[input[i]];
                if (fuse_input[i] == nullptr)
                {
                    fuse_input[i] = input_node;
                }
                // found different input nodes between different args, can't fuse.
                else if (fuse_input[i] != input_node)
                {
                    return {nullptr};
                }
            }
        }
    }
    if (fuse_input[0] && fuse_input[1])
    {
        return std::make_shared<op::BatchMatMulTranspose>(
            fuse_input[0], fuse_input[1], transpose[0], transpose[1]);
    }
    return {nullptr};
}

bool ngraph::pass::BatchFusion::run_on_function(std::shared_ptr<Function> func)
{
    bool modified = false;

    for (auto n : func->get_ordered_ops())
    {
        const Node& node = *n;
        if (TI(node) == TI(op::Concat))
        {
            if (m_fusion_type.is_set(FusionType::DIFFERENTIABLE_FUSIONS))
            {
                if (auto fused_node = fuse_batch_mat_mul_transpose(n))
                {
                    func->replace_node(n, fused_node);
                    modified = true;
                }
            }
            if (m_fusion_type.is_set(FusionType::REGULAR_FUSIONS))
            {
                if (auto fused_conv = fuse_group_convolution(n))
                {
                    func->replace_node(n, fused_conv);
                    modified = true;
                }
            }
        }
    }
    return modified;
}
