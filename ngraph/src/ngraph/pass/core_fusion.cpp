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
#include <unordered_set>

#include "ngraph/pass/core_fusion.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"

using namespace ngraph;
using namespace std;

static shared_ptr<Node> construct_constant_node(int n)
{
    return op::Constant::create(element::f32, Shape{}, {n});
}

void pass::CoreFusion::construct_relu()
{
    auto iconst0 = construct_constant_node(0);
    auto val = make_shared<pattern::op::Label>(iconst0);
    auto zero = make_shared<pattern::op::Label>(iconst0, nullptr, NodeVector{iconst0});

    auto skip_broadcast = make_shared<pattern::op::Skip>(zero, pattern::has_class<op::Broadcast>());
    auto max = make_shared<op::Maximum>(skip_broadcast, val);

    auto callback = [val, zero](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_relu against "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto mzero = m.get_pattern_map()[zero];
        if (!is_zero(mzero))
        {
            NGRAPH_DEBUG << "zero constant = " << mzero->get_name() << " not equal to 0\n";
            return false;
        }
        auto mpattern = m.get_match_root();

        auto cg = shared_ptr<Node>(new op::Relu(pattern_map[val]));
        replace_node(m.get_match_root(), cg);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(max, "CoreFusion.Relu");
    this->add_matcher(m, callback, all_pass_property_off);
}

void pass::CoreFusion::construct_sigmoid()
{
    // construct variance
    auto input = make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = make_shared<op::Negative>(input);
    auto exp_neg_input = make_shared<op::Exp>(neg_input);

    auto constant = make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto skip_broadcast =
        make_shared<pattern::op::Skip>(constant, pattern::has_class<op::Broadcast>());

    auto add_exp = make_shared<op::Add>(exp_neg_input, skip_broadcast);
    auto divide_1_over_exp = make_shared<op::Divide>(skip_broadcast, add_exp);

    // Define a call back that needs to called once the DFG matches the pattern
    auto callback = [input, constant](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_fprop_sigmoid pattern against "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        if (m.get_match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }

        if (m.get_match_root()->get_output_size() != pattern_map[input]->get_output_size())
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << "input= " << pattern_map[input]->get_name() << "size dont match!";
            return false;
        }

        if (!is_one(pattern_map[constant]))
        {
            NGRAPH_DEBUG << "Node not constant or not 1";
            return false;
        }
        auto sigmoid_node = make_shared<op::Sigmoid>(pattern_map[input]);
        replace_node(m.get_match_root(), sigmoid_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_1_over_exp, "CoreFusion.Sigmoid");
    this->add_matcher(m, callback, all_pass_property_off);
}

void pass::CoreFusion::construct_reshape_broadcast()
{
    Shape input_shape{10};
    auto input = make_shared<pattern::op::Label>(element::f32, input_shape);
    auto reshape1 = make_shared<op::Reshape>(input, AxisVector{0}, Shape{10, 1});
    auto broadcast = make_shared<op::Broadcast>(reshape1, Shape{10, 1, 20}, AxisSet{2});

    auto callback = [input](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_reshape_broadcast against "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto broadcast_m = static_pointer_cast<op::Broadcast>(m.get_match_root());
        auto reshape1_m =
            static_pointer_cast<op::Reshape>(broadcast_m->input_value(0).get_node_shared_ptr());
        auto input_m = m.get_pattern_value_map()[input];

        // it doesn't seem to make sense to support shapes : [0] or [1]
        if (input_m.get_shape().size() != 1 || input_m.get_shape().at(0) < 2)
        {
            NGRAPH_DEBUG << "input_m isn't a scalar or contains zero dimension";
            return false;
        }

        size_t dim = input_m.get_shape().at(0);

        // We are going to support the most common case where broadcast doesn't add 1-dimensions
        // since it's also very simple to implement
        size_t dim_one_count = 0;
        for (auto d : reshape1_m->get_shape())
        {
            if (d != 1 && d != dim)
            {
                NGRAPH_DEBUG << "Input is reshaped in a way we can't directly broadcast ( shape = "
                             << vector_to_string(reshape1_m->get_shape()) << ")";
                return false;
            }

            if (d == 1)
            {
                dim_one_count++;
            }
        }

        AxisSet new_axes = broadcast_m->get_broadcast_axes();
        auto broadcast_shape = broadcast_m->get_shape();
        for (size_t i = 0; i < broadcast_shape.size(); i++)
        {
            if (broadcast_shape[i] == 1)
            {
                dim_one_count--;
                new_axes.insert(i);
            }
        }

        if (dim_one_count != 0)
        {
            NGRAPH_DEBUG << "Broadcast adds 1-dimensions";
            return false;
        }

        auto new_broadcast =
            make_shared<op::Broadcast>(input_m, broadcast_m->get_shape(), new_axes);
        replace_node(m.get_match_root(), new_broadcast);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(broadcast, "CoreFusion.ReshapeBroadcast");
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

void pass::CoreFusion::construct_reshape_softmax_reshape()
{
    Shape input_shape{10, 20};
    AxisVector io{1, 0};
    auto input = make_shared<pattern::op::Label>(element::f32, input_shape);
    auto reshape1 = make_shared<op::Reshape>(input, io, Shape{20, 10});
    auto softmax = make_shared<op::Softmax>(reshape1, AxisSet{1});
    auto reshape2 = make_shared<op::Reshape>(softmax, io, input_shape);

    auto callback = [input](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_reshape_softmax_reshape against "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto reshape2_m = static_pointer_cast<op::Reshape>(m.get_match_root());
        auto softmax_m =
            static_pointer_cast<op::Softmax>(reshape2_m->input_value(0).get_node_shared_ptr());
        auto reshape1_m =
            static_pointer_cast<op::Reshape>(softmax_m->input_value(0).get_node_shared_ptr());
        auto input_m = m.get_pattern_map()[input];

        if (!reshape2_m->get_is_transpose() || !reshape1_m->get_is_transpose())
        {
            NGRAPH_DEBUG << "we expect reshape2 and reshape1 both be dimshuffles";
            return false;
        }

        if (input_m->get_shape() != reshape2_m->get_shape())
        {
            NGRAPH_DEBUG << "input and reshape2's shape are different";
            return false;
        }

        AxisSet new_axes;
        const auto& axis_order = reshape2_m->get_input_order();
        for (auto axis : softmax_m->get_axes())
        {
            new_axes.insert(axis_order.at(axis));
        }

        auto new_softmax = make_shared<op::Softmax>(input_m, new_axes);
        replace_node(m.get_match_root(), new_softmax);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(reshape2, "CoreFusion.ReshapeSoftmaxReshape");
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}
