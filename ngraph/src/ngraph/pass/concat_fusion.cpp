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

#include "concat_fusion.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

namespace
{
    bool check_self_concat_op(const std::shared_ptr<Node>& op)
    {
        auto input_args = op->get_arguments();
        std::set<std::shared_ptr<Node>> input_args_set(input_args.begin(), input_args.end());
        return (input_args_set.size() == 1);
    }

    bool check_concat_axis_dim_value(const std::shared_ptr<Node>& concat_op)
    {
        auto input_shape = concat_op->get_input_shape(0);
        size_t concat_axis =
            std::static_pointer_cast<op::Concat>(concat_op)->get_concatenation_axis();

        return (input_shape[concat_axis] == 1);
    }

    bool check_concat_has_no_fan_out(const std::shared_ptr<Node>& op)
    {
        auto no_fan_out = ngraph::pass::get_no_fan_out_function();
        return no_fan_out(op);
    }

    bool valid_self_concat(const std::shared_ptr<Node>& Op)
    {
        if (!check_self_concat_op(Op))
        {
            NGRAPH_DEBUG << "self_concat_fusion: Matcher matched " << Op->get_name()
                         << " but it is not a self concat\n";
            return false;
        }

        if (!check_concat_axis_dim_value(Op))
        {
            NGRAPH_DEBUG << "self_concat_fusion: Input shape value along concat axis of "
                         << Op->get_name() << " is not equal to 1\n";
            return false;
        }

        return true;
    }

    std::vector<size_t> get_concatenation_axis_vector(const NodeVector& bounded_concat_ops)
    {
        std::vector<size_t> concat_axis_vec;
        for (auto iter : bounded_concat_ops)
        {
            auto concat_op = std::static_pointer_cast<op::Concat>(iter);
            concat_axis_vec.push_back(concat_op->get_concatenation_axis());
        }
        return concat_axis_vec;
    }
}

void pass::ConcatElimination::construct_concat_elimination()
{
    auto op_label = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3});
    auto concat = std::make_shared<op::Concat>(NodeVector{op_label}, 0);
    auto concat_label = std::make_shared<pattern::op::Label>(concat, nullptr, NodeVector{concat});

    auto callback = [op_label](pattern::Matcher& m) {
        NGRAPH_DEBUG
            << "concat_elimination: In callback for construct_concat_elimination against node = "
            << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto op = pattern_map[op_label];

        auto root = as_type_ptr<op::Concat>(m.get_match_root());
        if (root && (root->get_input_shape(0) == root->get_output_shape(0)))
        {
            NGRAPH_DEBUG << " eliminated " << m.get_match_root() << "\n";
            replace_node(m.get_match_root(), op);

            return true;
        }
        NGRAPH_DEBUG << " Incorrect match in callback\n";
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(concat_label, "ConcatElimination");
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

bool ngraph::pass::SelfConcatFusion::run_on_function(std::shared_ptr<Function> function)
{
    bool modify_graph = false;
    auto has_multiple_inputs = [](std::shared_ptr<Node> n) {
        auto input_size = n->get_input_size();
        auto root = as_type_ptr<op::Concat>(n);
        return (root && input_size > 1);
    };

    auto print_state_of_bounded_vectors = [this]() -> std::string {
        std::stringstream ss;
        ss << "-----------------------------------------------------------" << std::endl;
        ss << "State of bounded pattern node vectors: " << std::endl;
        ss << "-----------------------------------------------------------" << std::endl;
        ss << "Number of pattern node vectors: " << this->m_concat_pattern_vectors.size()
           << std::endl;
        size_t c = 0;
        for (auto iter : this->m_concat_pattern_vectors)
        {
            ss << "For vector " << c << std::endl;
            auto iter_node_vec = iter;
            ss << "concat_op_vector: ";
            for (auto it : iter_node_vec)
            {
                ss << it->get_name() << " ";
            }
            ss << std::endl;
            c++;
        }
        ss << "-----------------------------" << std::endl;
        return ss.str();
    };

    auto concat_op_label =
        std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3}, has_multiple_inputs);
    auto matcher = std::make_shared<pattern::Matcher>(concat_op_label);
    for (auto n : function->get_ordered_ops())
    {
        construct_concat_patterns(matcher, concat_op_label, n);
    }

    NGRAPH_DEBUG << print_state_of_bounded_vectors();

    for (auto concat_op_pattern_node_vector : this->m_concat_pattern_vectors)
    {
        modify_graph = replace_patterns(concat_op_pattern_node_vector);
    }

    return modify_graph;
}

void ngraph::pass::SelfConcatFusion::construct_concat_patterns(
    const std::shared_ptr<pattern::Matcher>& matcher,
    const std::shared_ptr<pattern::op::Label>& concat_op_label,
    const std::shared_ptr<Node>& n)
{
    if (matcher->match(n))
    {
        auto concat_op = matcher->get_pattern_map()[concat_op_label];
        if (!is_type<op::Concat>(concat_op))
        {
            NGRAPH_DEBUG << "self_concat_fusion: Pattern matcher matched incorrect op. Matched "
                         << concat_op->get_name() << " instead of a self concat";
            return;
        }
        if (!valid_self_concat(concat_op))
        {
            NGRAPH_DEBUG << "self_concat_fusion: " << concat_op->get_name()
                         << " is not a valid self concat\n";
            return;
        }
        else
        {
            NGRAPH_DEBUG << "self_concat_fusion: " << concat_op->get_name()
                         << " is a VALID self concat\n";
        }

        auto& concat_vectors = this->m_concat_pattern_vectors;
        if (concat_vectors.empty())
        {
            concat_vectors.push_back(NodeVector{concat_op});
        }
        else
        {
            update_concat_pattern_vectors(concat_op);
        }
    }
}

void ngraph::pass::SelfConcatFusion::update_concat_pattern_vectors(
    const std::shared_ptr<Node>& concat_op)
{
    bool concat_source_found = false;
    for (auto& concat_pattern_vec : this->m_concat_pattern_vectors)
    {
        auto last_op_in_pattern_vec = concat_pattern_vec.back();
        if ((concat_op->get_argument(0) == last_op_in_pattern_vec) &&
            (check_concat_has_no_fan_out(last_op_in_pattern_vec)))
        {
            concat_pattern_vec.push_back(concat_op);
            concat_source_found = true;
            break;
        }
    }

    if (!concat_source_found)
    {
        this->m_concat_pattern_vectors.push_back(NodeVector{concat_op});
    }
}

void ngraph::pass::SelfConcatFusion::remove_single_concat_op_pattern()
{
    auto iter = m_concat_pattern_vectors.begin();
    while (iter != m_concat_pattern_vectors.end())
    {
        if (iter->size() == 1)
        {
            iter = m_concat_pattern_vectors.erase(iter);
        }
        else
        {
            iter++;
        }
    }
}

bool ngraph::pass::SelfConcatFusion::replace_patterns(const NodeVector& bounded_concat_ops)
{
    auto scalarize_dim = [](std::vector<size_t> concat_axis_vector,
                            const Shape& input_shape) -> Shape {

        Shape scalarized_shape;
        for (size_t i = 0; i < input_shape.size(); i++)
        {
            auto it = std::find(concat_axis_vector.begin(), concat_axis_vector.end(), i);
            if (it == concat_axis_vector.end())
            {
                scalarized_shape.push_back(input_shape[i]);
            }
        }
        return scalarized_shape;
    };

    auto concat_axis_vector = get_concatenation_axis_vector(bounded_concat_ops);

    auto& first_bounded_concat = (*bounded_concat_ops.begin());
    auto driver_op = first_bounded_concat->get_argument(0);
    const Shape& input_shape = first_bounded_concat->get_input_shape(0);

    auto scalarized_shape = scalarize_dim(concat_axis_vector, input_shape);
    AxisVector axis_order = get_default_order(input_shape);
    auto reshape = std::make_shared<op::Reshape>(driver_op, axis_order, scalarized_shape);
    auto last_bounded_concat_op = bounded_concat_ops.back();
    auto broadcast_out_shape = last_bounded_concat_op->get_shape();
    auto broadcast =
        std::make_shared<op::v0::Broadcast>(reshape, broadcast_out_shape, concat_axis_vector);

    replace_node(last_bounded_concat_op, broadcast);
    return true;
}
