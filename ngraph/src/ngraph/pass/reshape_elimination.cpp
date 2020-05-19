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

#include "reshape_elimination.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void pass::ReshapeElimination::construct_identity_reshape_pattern()
{
    Shape shape_op{3};
    Shape shape_r1{1, 3};

    auto op = make_shared<pattern::op::Label>(element::f32, shape_op);
    auto reshape1 = make_shared<op::Reshape>(op, AxisVector{0}, shape_r1);

    auto callback = [op](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_identity_reshape_pattern against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_value_map();
        auto gop = pattern_map[op];

        auto r1 = as_type_ptr<op::Reshape>(m.get_match_root());

        if (r1->get_shape() != gop.get_shape())
        {
            NGRAPH_DEBUG << "Not a no-op; Shapes are different!";
            return false;
        }

        auto do_r1 = get_default_order(r1->get_shape());

        if (do_r1 != r1->get_input_order())
        {
            NGRAPH_DEBUG << "Not a no-op; Not in default input order!";
            return false;
        }

        m.get_match_value().replace(gop);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(reshape1);
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

void pass::ReshapeElimination::construct_reshapex2_pattern()
{
    Shape shape_op{3};
    Shape shape_r1{1, 3};

    auto op = make_shared<pattern::op::Label>(element::f32, shape_op);
    auto reshape1 = make_shared<op::Reshape>(op, AxisVector{0}, shape_r1);
    auto reshape2 = make_shared<op::Reshape>(reshape1, AxisVector{0, 1}, shape_op);

    auto callback = [op](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_reshapex2_pattern against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto gop = pattern_map[op];

        auto r2 = static_pointer_cast<op::Reshape>(m.get_match_root());
        auto r1 = static_pointer_cast<op::Reshape>(r2->get_argument(0));

        if (gop->get_shape() != m.get_match_root()->get_shape())
        {
            // First reshape transposes and second reshape only changes shape
            // Replace with a transpose that changes shape
            if (apply_permutation(gop->get_shape(), r1->get_input_order()) == r2->get_shape() &&
                r2->get_input_order() == get_default_order(r1->get_shape()) &&
                r1->get_users().size() == 1)
            {
                replace_node(m.get_match_root(),
                             make_shared<op::Reshape>(gop, r1->get_input_order(), r2->get_shape()));
                return true;
            }
            else
            {
                NGRAPH_DEBUG << "Operand shape doesn't match the shape of the second reshape!";
                NGRAPH_DEBUG << "gop " << gop->get_name()
                             << "shape = " << vector_to_string(gop->get_shape());
                NGRAPH_DEBUG << "match_root " << m.get_match_root()->get_name()
                             << "shape = " << vector_to_string(m.get_match_root()->get_shape());
                return false;
            }
        }

        // Check for sequence of reshapes/transposes that cancel out.
        auto do_r2 = get_default_order(r1->get_shape());
        auto do_r1 = get_default_order(gop->get_shape());

        NGRAPH_DEBUG << "r1's i/o = " << vector_to_string(r1->get_input_order())
                     << "do_r1 = " << vector_to_string(do_r1);
        NGRAPH_DEBUG << "r2's i/o = " << vector_to_string(r2->get_input_order())
                     << "do_r2 = " << vector_to_string(do_r2);

        if (r1->get_input_order() == do_r1 && r2->get_input_order() == do_r2)
        {
            NGRAPH_DEBUG << "Two reshapes were removed!";
            replace_node(m.get_match_root(), gop);
            return true;
        }

        auto perm1 = apply_permutation(do_r1, r1->get_input_order());
        auto perm2 = apply_permutation(perm1, r2->get_input_order());
        if (perm2 == do_r1)
        {
            NGRAPH_DEBUG << "Two transposes were removed!";
            replace_node(m.get_match_root(), gop);
            return true;
        }

        return false;
    };
    auto m = make_shared<pattern::Matcher>(reshape2);
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

void pass::ReshapeElimination::construct_dot_transpose_pattern()
{
    // dot(A,B).T = dot (B.T, A.T)
    auto dot_pred = [](shared_ptr<Node> n) { return is_type<op::Dot>(n); };

    auto pdot = make_shared<pattern::op::Label>(element::f32, Shape{2, 1}, dot_pred);
    auto preshape = make_shared<op::Reshape>(pdot, AxisVector{1, 0}, Shape{1, 2});

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_dot_transpose_pattern against node = "
                     << m.get_match_root()->get_name();

        auto mtranspose = static_pointer_cast<op::Reshape>(m.get_match_root());
        // this also checks the rank
        if (mtranspose->get_input_order() != AxisVector{1, 0})
        {
            NGRAPH_DEBUG << "Reshape isn't transpose. "
                         << vector_to_string(mtranspose->get_input_order());
            return false;
        }

        auto mdot = mtranspose->get_argument(0);
        if (mdot->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << "Dot has the wrong shape. " << vector_to_string(mdot->get_shape());
            return false;
        }

        auto arg0 = mdot->get_argument(0);
        if (arg0->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << "Arg0 has the wrong shape. " << vector_to_string(arg0->get_shape());
            return false;
        }
        auto reshape0_shape = Shape{arg0->get_shape().at(1), arg0->get_shape().at(0)};
        auto reshape0 = make_shared<op::Reshape>(arg0, AxisVector{1, 0}, reshape0_shape);

        auto arg1 = mdot->get_argument(1);
        if (arg1->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << "Arg1 has the wrong shape. " << vector_to_string(arg1->get_shape());
            return false;
        }
        auto reshape1_shape = Shape{arg1->get_shape().at(1), arg1->get_shape().at(0)};
        auto reshape1 = make_shared<op::Reshape>(arg1, AxisVector{1, 0}, reshape1_shape);

        auto tdot = shared_ptr<Node>(new op::Dot(reshape1, reshape0));
        replace_node(m.get_match_root(), tdot);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(preshape);
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

void pass::RecurrentReshapeElimination::construct_recurrent_reshape()
{
    Shape shape_op{3};
    Shape shape_r{1, 3};

    auto op = make_shared<pattern::op::Label>(element::f32, shape_op);
    auto reshape = make_shared<op::Reshape>(op, AxisVector{0}, shape_r);
    auto reshape_label =
        make_shared<pattern::op::Label>(reshape, get_no_fan_out_function(), NodeVector{reshape});

    auto callback = [op, reshape_label](pattern::RecurrentMatcher& m) {
        NGRAPH_DEBUG << "In callback for construct_recurrent_reshape against node = "
                     << reshape_label->get_argument(0)->get_name();
        auto reshape_node_vector = m.get_bound_nodes_for_pattern(reshape_label);

        // The bound node vector is in reverse order. It is convenient to have the
        // bound node vector in the correct order
        std::reverse(std::begin(reshape_node_vector), std::end(reshape_node_vector));

        auto first_bound_reshape_op = reshape_node_vector.front();
        auto driver_op = first_bound_reshape_op->get_argument(0);
        auto last_bound_reshape_op = reshape_node_vector.back();

        // Need to check if the user of the last bound op is a reshape since the last reshape is
        // allowed to have fan-out but the matcher will discard any reshape if it has fan-out
        auto user_of_last_bound_reshape_op = last_bound_reshape_op->get_users(true)[0];
        if (is_type<op::Reshape>(user_of_last_bound_reshape_op))
        {
            reshape_node_vector.push_back(user_of_last_bound_reshape_op);
            last_bound_reshape_op = reshape_node_vector.back();
        }

        // Return if the recurrent matcher matches only one reshape
        if (reshape_node_vector.size() == 1)
        {
            return false;
        }

        // The complete reshape node vector may not contain contiguous reshapes that can be
        // fused. Only the subset of reshapes with a reshape(any axis order) followed by reshapes
        // with default axis order can be fused. Creating such subpatterns here:
        std::vector<NodeVector> sub_patterns{NodeVector{first_bound_reshape_op}};
        for (auto it = std::next(reshape_node_vector.begin()); it != reshape_node_vector.end();
             it++)
        {
            auto r = as_type_ptr<op::Reshape>(*it);

            // Check that the input to r is the last reshape stored in the
            // subpattern vector
            if (!r)
            {
                NGRAPH_DEBUG
                    << "Incorrect match. Something went wrong. Non-reshape op has been matched";
                return false;
            }

            auto default_order_r = get_default_order(r->get_input_shape(0));
            if (r->get_input_order() == default_order_r)
            {
                sub_patterns.back().push_back(r);
            }
            else
            {
                NGRAPH_DEBUG << r->get_name() << "does not have default axis order. "
                             << "It might be part of a different subpattern";
                sub_patterns.push_back(NodeVector{r});
            }
        }

        bool modify_graph = false;

        // Replace the patterns
        for (auto sub_pattern : sub_patterns)
        {
            // Do not consider subpatterns with just one reshape in them
            if (sub_pattern.size() == 1)
            {
                continue;
            }

            auto first_reshape = as_type_ptr<op::Reshape>(sub_pattern.front());
            auto input_to_first_reshape = first_reshape->get_argument(0);
            auto last_reshape = as_type_ptr<op::Reshape>(sub_pattern.back());

            auto new_input_order = first_reshape->get_input_order();
            auto new_out_shape = last_reshape->get_shape();

            auto new_reshape = std::make_shared<op::Reshape>(
                input_to_first_reshape, new_input_order, new_out_shape);

            replace_node(last_reshape, new_reshape);
            modify_graph = true;
        }

        return modify_graph;
    };
    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m =
        std::make_shared<pattern::RecurrentMatcher>(reshape_label, op, empty_correlated_matches);
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}
