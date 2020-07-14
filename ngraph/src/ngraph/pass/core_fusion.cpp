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
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/softmax_crossentropy.hpp"
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

void pass::CoreFusion::construct_softmax_cross_entropy_fprop()
{
    auto param_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{41, 37});
    auto softmax = std::make_shared<ngraph::op::Softmax>(param_1, AxisSet{1});

    // parameter with one-hot encoded values
    auto param_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{41, 37});
    auto log = std::make_shared<ngraph::op::Log>(softmax);
    auto multiply = std::make_shared<ngraph::op::Multiply>(param_2, log);

    auto reduction_axes = ngraph::op::Constant::create(element::i64, Shape{}, {1});
    auto reduction_axes_label = std::make_shared<pattern::op::Label>(reduction_axes);
    auto sum = std::make_shared<ngraph::op::Sum>(multiply, reduction_axes_label);
    auto negative = std::make_shared<ngraph::op::Negative>(sum);
    auto reshape = std::make_shared<ngraph::op::Reshape>(negative, AxisVector{0}, Shape{41, 1});

    auto callback = [reduction_axes_label, param_1, param_2](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_softmax_cross_entropy_fprop against "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto input_to_normalize = pattern_map[param_1];
        auto labels = pattern_map[param_2];
        auto softmax_crossentropy =
            std::make_shared<ngraph::op::SoftmaxCrossEntropy>(input_to_normalize, labels, true);
        ngraph::replace_node(m.get_match_root(), softmax_crossentropy);

        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(reshape, "CoreFusion.SoftmaxCrossEntropy");
    this->add_matcher(m, callback);
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

void pass::CoreFusion::construct_folded_batch_norm()
{
    Shape shape{2, 2, 1, 1};
    auto input = make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = make_shared<pattern::op::Label>(element::f32, shape);

    auto pconv = make_shared<op::Convolution>(input,
                                              filters,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    auto mean_shape = Shape{2};
    auto mean = make_shared<pattern::op::Label>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<pattern::op::Label>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<pattern::op::Label>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<pattern::op::Label>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = make_shared<op::BatchNormInference>(pconv, gamma, beta, mean, var, eps);

    auto callback = [input, filters, mean, var, gamma, beta](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for folded batch norm against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto m_bn = static_pointer_cast<op::BatchNormInference>(m.get_match_root());
        auto m_conv = static_pointer_cast<op::Convolution>(m_bn->get_argument(2));

        if (m_conv->get_users().size() > 1)
        {
            return false;
        }

        if (m_conv->get_shape().size() != 4)
        {
            return false;
        }

        // new weights = old weights * gamma / sqrt(variance + epsilon)
        // new biases = -mean * gamma / sqrt(variance + epsilon) + beta

        auto bn_eps = op::Constant::create(element::f32, Shape{}, {m_bn->get_eps_value()});
        auto var_eps = make_shared<op::Add>(
            pattern_map[var],
            make_shared<op::Broadcast>(bn_eps, pattern_map[var]->get_shape(), AxisSet{0}));
        auto sqrt_var_eps = make_shared<op::Sqrt>(var_eps);

        auto mean_gamma = make_shared<op::Multiply>(pattern_map[mean], pattern_map[gamma]);
        auto new_biases = make_shared<op::Subtract>(
            pattern_map[beta], make_shared<op::Divide>(mean_gamma, sqrt_var_eps));
        auto weight_scaling = make_shared<op::Divide>(pattern_map[gamma], sqrt_var_eps);
        auto new_weights = make_shared<op::Multiply>(
            pattern_map[filters],
            make_shared<op::Broadcast>(
                weight_scaling, pattern_map[filters]->get_shape(), AxisSet{1, 2, 3}));

        auto conv = make_shared<op::Convolution>(pattern_map[input],
                                                 new_weights,
                                                 m_conv->get_window_movement_strides(),
                                                 m_conv->get_window_dilation_strides(),
                                                 m_conv->get_padding_below(),
                                                 m_conv->get_padding_above(),
                                                 m_conv->get_data_dilation_strides());
        auto conv_bias =
            conv + make_shared<op::Broadcast>(new_biases, conv->get_shape(), AxisSet{0, 2, 3});
        replace_node(m.get_match_root(), conv_bias);

        return true;

    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(bn, "CoreFusion.FoldedBatchNorm");
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

void pass::CoreFusion::construct_conv_affine_folding()
{
    // A * Conv (input, filters) + B -> ConvBias (input, filters * A_c, B_c)
    Shape shape{2, 2, 1, 1};
    auto input = make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = make_shared<pattern::op::Label>(element::f32, shape);

    auto conv = make_shared<op::Convolution>(input,
                                             filters,
                                             Strides{1, 1},
                                             Strides{1, 1},
                                             CoordinateDiff{0, 0},
                                             CoordinateDiff{0, 0},
                                             Strides{1, 1});
    auto conv_label = make_shared<pattern::op::Label>(conv, nullptr, NodeVector{conv});

    auto Ac = make_shared<pattern::op::Label>(element::f32, Shape{2});
    auto A = make_shared<op::Broadcast>(Ac, Shape{2, 2, 1, 1}, AxisSet{0, 2, 3});
    auto A_label = make_shared<pattern::op::Label>(A, nullptr, NodeVector{A});
    auto Bc = make_shared<pattern::op::Label>(element::f32, Shape{2});
    auto B = make_shared<op::Broadcast>(Bc, Shape{2, 2, 1, 1}, AxisSet{0, 2, 3});
    auto B_label = make_shared<pattern::op::Label>(B, nullptr, NodeVector{B});
    auto multiply = make_shared<op::Multiply>(conv_label, A_label);
    auto add = make_shared<op::Add>(multiply, B_label);

    auto callback = [input, filters, conv_label, A_label, B_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for conv affine folding against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto conv_m = static_pointer_cast<op::Convolution>(pattern_map[conv_label]);

        if (conv_m->get_users().size() > 1)
        {
            return false;
        }

        if (conv_m->get_shape().size() != 4)
        {
            return false;
        }

        auto A_m = static_pointer_cast<op::Broadcast>(pattern_map[A_label]);
        auto B_m = static_pointer_cast<op::Broadcast>(pattern_map[B_label]);

        // Check if values are being broadcast along channel (2nd) dimension
        auto is_channel_bcast = [](const shared_ptr<op::Broadcast>& bcast) {

            if (bcast->get_input_shape(0).size() == 1 &&
                bcast->get_broadcast_axes() == AxisSet{0, 2, 3})
            {
                return true;
            }

            if (bcast->get_input_shape(0).size() == 2)
            {
                auto input_shape = bcast->get_input_shape(0);
                if (input_shape[0] == 1 && bcast->get_broadcast_axes() == AxisSet{2, 3})
                    return true;
            }
            return false;
        };

        if (!is_channel_bcast(A_m) || !is_channel_bcast(B_m))
        {
            return false;
        }

        auto get_bcast_input = [](const shared_ptr<op::Broadcast>& bcast) {
            if (bcast->get_input_shape(0).size() == 1)
            {
                return bcast->get_argument(0);
            }
            if (bcast->get_input_shape(0).size() == 2)
            {
                Shape bshape{bcast->get_input_shape(0)[1]};
                return static_pointer_cast<Node>(
                    make_shared<op::Reshape>(bcast->get_argument(0), AxisVector{0, 1}, bshape));
            }
            throw ngraph_error("Unexpected shape for bcast input");
        };

        auto Ac_m = get_bcast_input(A_m);

        // new weights = old weights * Ac_m
        // new biases = Bc_m

        auto filters_n = make_shared<op::Multiply>(
            pattern_map[filters],
            make_shared<op::Broadcast>(Ac_m, pattern_map[filters]->get_shape(), AxisSet{1, 2, 3}));

        auto conv_n = make_shared<op::Convolution>(pattern_map[input],
                                                   filters_n,
                                                   conv_m->get_window_movement_strides(),
                                                   conv_m->get_window_dilation_strides(),
                                                   conv_m->get_padding_below(),
                                                   conv_m->get_padding_above(),
                                                   conv_m->get_data_dilation_strides());
        auto convbias_n = conv_n + B_m;
        replace_node(m.get_match_root(), convbias_n);

        return true;

    };

    auto m = make_shared<pattern::Matcher>(add, "CoreFusion.ConvAffineFolding");
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

static bool is_trivial_convolution(shared_ptr<op::Convolution> conv, bool skip_pad_checks = false)
{
    Strides stride_1{1, 1};
    CoordinateDiff pad_0{0, 0};

    return conv->get_window_dilation_strides() == stride_1 &&
           conv->get_data_dilation_strides() == stride_1 &&
           (skip_pad_checks ||
            (conv->get_padding_above() == pad_0 && conv->get_padding_below() == pad_0));
}

static bool are_img_dims_equal(Shape conv_shape, Shape image_shape)
{
    if (conv_shape.size() != 4)
    {
        return false;
    }

    return conv_shape[2] == image_shape[0] && conv_shape[3] == image_shape[1];
}

static shared_ptr<Node> reduce_broadcast(shared_ptr<Node> broadcast)
{
    const size_t H = 2;
    const size_t W = 3;
    auto matched_broadcast_w1 = static_pointer_cast<op::Broadcast>(broadcast);
    Shape shape_w1{matched_broadcast_w1->get_shape()};
    shape_w1[H] /= 2;
    shape_w1[W] /= 2;
    auto new_broadcast_w1 =
        std::make_shared<op::Broadcast>(matched_broadcast_w1->get_argument(0),
                                        shape_w1,
                                        matched_broadcast_w1->get_broadcast_axes());
    return move(new_broadcast_w1);
}

static size_t shape_to_index(Shape shape)
{
    if (shape.size() != 4)
    {
        return 0;
    }
    const size_t HEIGHT_DIM = 2;
    const size_t WIDTH_DIM = 3;

    if (shape.at(HEIGHT_DIM) != shape.at(WIDTH_DIM))
    {
        return 0;
    }

    switch (shape.at(HEIGHT_DIM))
    {
    case 28: return 1;
    case 14: return 2;
    case 7: return 3;
    default: return 0;
    }
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
        auto reshape1_m = static_pointer_cast<op::Reshape>(broadcast_m->get_argument(0));
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

//   conv(56w3s1)                       conv(28w3s2)
//	      |                                |
//   conv(56w1s1)              ==>      conv(28w1s1)
//       |                                 |
// elt------------56               elt------------pool(28s2)
//   |            |                  |               |
// conv(28w1s2) conv(28w1s2)     conv(28w1s1)  conv(28w1s1)
void pass::CoreFusion::construct_optimized_strided_conv()
{
    Shape win_size_1{1, 1, 1, 1};
    auto is_bc = pattern::has_class<op::Broadcast>();
    auto data_stride3 = make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 128, 128});
    auto weights_stride3 = make_shared<pattern::op::Label>(element::f32, win_size_1);

    auto conv_stride3 = make_shared<op::Convolution>(data_stride3, weights_stride3);

    auto conv_stride3_label =
        make_shared<pattern::op::Label>(conv_stride3, nullptr, NodeVector{conv_stride3});

    auto broadcast_w3_label = make_shared<pattern::op::Label>(conv_stride3_label, is_bc);
    auto add_w3 = make_shared<op::Add>(conv_stride3_label, broadcast_w3_label);
    auto relu_w3 = make_shared<op::Relu>(add_w3);

    auto weights_stride1 = make_shared<pattern::op::Label>(element::f32, win_size_1);
    auto conv_stride1 = make_shared<op::Convolution>(relu_w3, weights_stride1);
    auto conv_stride1_label =
        make_shared<pattern::op::Label>(conv_stride1, nullptr, NodeVector{conv_stride1});
    auto broadcast_w1_label = make_shared<pattern::op::Label>(conv_stride1_label, is_bc);
    auto add_w1 = make_shared<op::Add>(conv_stride1_label, broadcast_w1_label);

    auto eltwise_arg_label =
        make_shared<pattern::op::Label>(element::f32, conv_stride1->get_shape());
    auto add_two_convs = make_shared<op::Add>(add_w1, eltwise_arg_label);

    auto relu_two_convs = make_shared<op::Relu>(add_two_convs);

    auto eltwise_label =
        make_shared<pattern::op::Label>(relu_two_convs, nullptr, NodeVector{relu_two_convs});

    auto weights_eltwise = make_shared<pattern::op::Label>(element::f32, win_size_1);
    auto eltwise_conv = make_shared<op::Convolution>(eltwise_label, weights_eltwise);

    auto callback = [win_size_1,
                     eltwise_label,
                     conv_stride1_label,
                     conv_stride3_label,
                     eltwise_arg_label,
                     broadcast_w3_label,
                     broadcast_w1_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_skip against "
                     << m.get_match_root()->get_name();

        if (m.get_match_root()->get_users().empty())
        {
            NGRAPH_DEBUG << m.get_match_root()
                         << " has already been replaced by a preceding callback";
            return false;
        }

        auto pattern_map = m.get_pattern_map();
        auto m_eltwise = pattern_map[eltwise_label];

        vector<shared_ptr<Node>> strided_convs;
        for (auto n : m_eltwise->get_users())
        {
            if (is_used(n.get()))
            {
                if (!is_type<op::Convolution>(n))
                {
                    NGRAPH_DEBUG << "Not all live users of element wise operation are Convolution";
                    return false;
                }
                strided_convs.push_back(n);
            }
        }

        if (strided_convs.size() != 2)
        {
            NGRAPH_DEBUG << "Number of live users of element wise operation isn't equal to 2";
            return false;
        }

        Shape supported_shapes[] = {Shape{56, 56}, Shape{28, 28}, Shape{14, 14}, Shape{7, 7}};
        Shape shape_1{1, 1};
        Shape shape_3{3, 3};
        Strides stride_2{2, 2};
        Strides stride_1{1, 1};
        CoordinateDiff pad_0{0, 0};
        CoordinateDiff pad_1{1, 1};
        Shape win_size_3{1, 1, 3, 3};

        size_t sparse_shape_index = 0;
        NodeVector sconvs;
        for (auto sc : strided_convs)
        {
            if (sc->get_argument(0) != m_eltwise)
            {
                NGRAPH_DEBUG << "element-wise isn't data";
                return false;
            }
            auto sconv = static_pointer_cast<op::Convolution>(sc);
            sparse_shape_index = shape_to_index(sconv->get_shape());
            if (sparse_shape_index == 0)
            {
                NGRAPH_DEBUG << "Unsupported shape of " << sconv->get_name();
                return false;
            }
            if (!are_img_dims_equal(sconv->get_shape(), supported_shapes[sparse_shape_index]) ||
                !are_img_dims_equal(sconv->get_input_shape(1), shape_1) ||
                sconv->get_window_movement_strides() != stride_2 || !is_trivial_convolution(sconv))
            {
                NGRAPH_DEBUG << sconv->get_name() << " and its weights are of the wrong shape (not "
                             << vector_to_string(supported_shapes[sparse_shape_index])
                             << " and 1x1) and strides (2x2)";
                return false;
            }
            sconvs.push_back(sconv);
        }

        const size_t full_shape_index = sparse_shape_index - 1;

        auto m_conv_stride1 = static_pointer_cast<op::Convolution>(pattern_map[conv_stride1_label]);

        if (!are_img_dims_equal(m_conv_stride1->get_shape(), supported_shapes[full_shape_index]) ||
            !are_img_dims_equal(m_conv_stride1->get_input_shape(1), win_size_1) ||
            m_conv_stride1->get_window_movement_strides() != stride_1 ||
            !is_trivial_convolution(m_conv_stride1))
        {
            NGRAPH_DEBUG << m_conv_stride1->get_name()
                         << " and its weights are of the wrong shape (not "
                         << vector_to_string(supported_shapes[full_shape_index])
                         << " and 1x1) and strides (1x1)";
            return false;
        }

        auto m_conv_stride3 = static_pointer_cast<op::Convolution>(pattern_map[conv_stride3_label]);

        if (!are_img_dims_equal(m_conv_stride3->get_shape(), supported_shapes[full_shape_index]) ||
            !are_img_dims_equal(m_conv_stride3->get_input_shape(1), shape_3) ||
            m_conv_stride3->get_window_movement_strides() != stride_1 ||
            !is_trivial_convolution(m_conv_stride3, true))
        {
            NGRAPH_DEBUG << m_conv_stride3->get_name()
                         << " and its weights are of the wrong shape (not "
                         << vector_to_string(supported_shapes[full_shape_index])
                         << " and 3x3) and strides (1x1)";
            return false;
        }

        auto conv_28w3s2 = make_shared<op::Convolution>(m_conv_stride3->get_argument(0),
                                                        m_conv_stride3->get_argument(1),
                                                        stride_2,
                                                        stride_1,
                                                        pad_1,
                                                        pad_1);

        auto new_add_conv_28w3s2 =
            make_shared<op::Add>(conv_28w3s2, reduce_broadcast(pattern_map[broadcast_w3_label]));
        auto new_relu_28w3s2 = make_shared<op::Relu>(new_add_conv_28w3s2);

        auto conv_28w1s1 = make_shared<op::Convolution>(
            new_relu_28w3s2, m_conv_stride1->get_argument(1), stride_1, stride_1);

        auto new_add_conv28s1 =
            make_shared<op::Add>(conv_28w1s1, reduce_broadcast(pattern_map[broadcast_w1_label]));

        auto maxpool =
            make_shared<op::MaxPool>(pattern_map[eltwise_arg_label], Shape{1, 1}, stride_2);
        auto new_add_two_convs = make_shared<op::Add>(new_add_conv28s1, maxpool);
        auto new_relu_two_convs = make_shared<op::Relu>(new_add_two_convs);

        for (auto sconv : sconvs)
        {
            auto sconv_28w1s1 = make_shared<op::Convolution>(
                new_relu_two_convs, sconv->get_argument(1), stride_1, stride_1);
            NGRAPH_DEBUG << "Replacing " << sconv->get_name() << " with "
                         << sconv_28w1s1->get_name();
            replace_node(sconv, sconv_28w1s1);
        }
        return true;
    };

    auto m = make_shared<pattern::Matcher>(eltwise_conv, "CoreFusion.OptimizedStridedConv");
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
        auto softmax_m = static_pointer_cast<op::Softmax>(reshape2_m->get_argument(0));
        auto reshape1_m = static_pointer_cast<op::Reshape>(softmax_m->get_argument(0));
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

static bool
    zero_padded_conv_consistency_check(const std::shared_ptr<ngraph::Node>& match_root,
                                       const std::shared_ptr<ngraph::op::Constant>& pad_value_op,
                                       const std::shared_ptr<ngraph::Node>& pad_input,
                                       const std::shared_ptr<ngraph::op::Pad>& matched_pad,
                                       const ngraph::CoordinateDiff& padding_below,
                                       const ngraph::CoordinateDiff& padding_above,
                                       size_t batch_index,
                                       size_t channel_index)
{
    // Only match float32 convolutions
    if (match_root->get_element_type() != ngraph::element::f32)
    {
        return false;
    }

    // Only match zero padding
    if (pad_value_op->get_vector<float>().at(0) != 0.0f)
    {
        return false;
    }

    // Only match 4D tensors
    if (pad_input->get_shape().size() != 4)
    {
        return false;
    }

    // Only match convolutions with no padding specification
    if (padding_below != ngraph::CoordinateDiff(2) || padding_above != ngraph::CoordinateDiff(2))
    {
        return false;
    }

    // Only match constant padding
    if (matched_pad->get_pad_mode() != ngraph::op::PadMode::CONSTANT)
    {
        return false;
    }

    // Only match no padding in the batch dimension
    if (matched_pad->get_padding_above().at(batch_index) != 0 ||
        matched_pad->get_padding_below().at(batch_index) != 0)
    {
        return false;
    }

    // Only match no padding in the channel dimension
    if (matched_pad->get_padding_above().at(channel_index) != 0 ||
        matched_pad->get_padding_below().at(channel_index) != 0)
    {
        return false;
    }

    return true;
}

void pass::CoreFusion::construct_zero_padded_reshaped_conv()
{
    auto pad_input = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto pad_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto pad =
        std::make_shared<ngraph::op::Pad>(pad_input, pad_value, CoordinateDiff{}, CoordinateDiff{});
    auto pad_label = std::make_shared<pattern::op::Label>(pad, nullptr, NodeVector{pad});

    auto reshape =
        std::make_shared<ngraph::op::Reshape>(pad_label, AxisVector{}, Shape{1, 1, 1, 1});
    auto reshape_label =
        std::make_shared<pattern::op::Label>(reshape, nullptr, NodeVector{reshape});

    auto conv_filter = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});

    auto conv = std::make_shared<ngraph::op::Convolution>(reshape_label,
                                                          conv_filter,
                                                          Strides{1, 1},
                                                          Strides{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          Strides{1, 1});
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, NodeVector{conv});

    auto callback = [pad_input, pad_value, pad_label, reshape_label, conv_filter, conv_label](
        pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto pad_value_op = as_type_ptr<ngraph::op::Constant>(pattern_map[pad_value]);
        if (!pad_value_op)
        {
            NGRAPH_DEBUG << "Pad value must be a constant";
            return false;
        }

        const auto& matched_conv = as_type_ptr<ngraph::op::Convolution>(pattern_map[conv_label]);
        const auto& matched_pad = as_type_ptr<ngraph::op::Pad>(pattern_map[pad_label]);
        const auto& matched_reshape =
            std::static_pointer_cast<ngraph::op::Reshape>(pattern_map[reshape_label]);

        const auto& input_order = matched_reshape->get_input_order();
        auto hoisted_reshape_output_shape =
            ngraph::apply_permutation<Shape>(pattern_map[pad_input]->get_shape(), input_order);

        auto hoisted_reshape = std::make_shared<ngraph::op::Reshape>(
            pattern_map[pad_input],
            input_order,
            Shape(hoisted_reshape_output_shape.begin(), hoisted_reshape_output_shape.end()));

        if (!zero_padded_conv_consistency_check(m.get_match_root(),
                                                pad_value_op,
                                                pattern_map[pad_input],
                                                matched_pad,
                                                matched_conv->get_padding_below(),
                                                matched_conv->get_padding_above(),
                                                input_order[0],
                                                input_order[1]))
        {
            return false;
        }

        CoordinateDiff padding_below{static_cast<CoordinateDiff::value_type>(
                                         matched_pad->get_padding_below().at(input_order[2])),
                                     static_cast<CoordinateDiff::value_type>(
                                         matched_pad->get_padding_below().at(input_order[3]))};
        CoordinateDiff padding_above{static_cast<CoordinateDiff::value_type>(
                                         matched_pad->get_padding_above().at(input_order[2])),
                                     static_cast<CoordinateDiff::value_type>(
                                         matched_pad->get_padding_above().at(input_order[3]))};

        auto zero_padded_conv =
            std::make_shared<ngraph::op::Convolution>(hoisted_reshape,
                                                      pattern_map[conv_filter],
                                                      matched_conv->get_window_movement_strides(),
                                                      matched_conv->get_window_dilation_strides(),
                                                      padding_below,
                                                      padding_above,
                                                      matched_conv->get_data_dilation_strides());

        ngraph::replace_node(m.get_match_root(), zero_padded_conv);
        return true;
    };

    auto m =
        std::make_shared<ngraph::pattern::Matcher>(conv_label, "CoreFusion.ZeroPaddedReshapedConv");
    this->add_matcher(m, callback);
}

void pass::CoreFusion::construct_zero_padded_conv()
{
    auto pad_input = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto pad_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto pad = std::make_shared<ngraph::op::Pad>(
        pad_input, pad_value, CoordinateDiff{0, 0, 0, 0}, CoordinateDiff{0, 0, 0, 0});
    auto pad_label = std::make_shared<pattern::op::Label>(pad, nullptr, NodeVector{pad});

    auto conv_filter = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});

    auto conv = std::make_shared<ngraph::op::Convolution>(pad_label,
                                                          conv_filter,
                                                          Strides{1, 1},
                                                          Strides{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          Strides{1, 1});
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, NodeVector{conv});

    auto callback = [pad_input, pad_value, pad_label, conv_filter, conv_label](
        pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto pad_value_op = as_type_ptr<ngraph::op::Constant>(pattern_map[pad_value]);
        if (!pad_value_op)
        {
            NGRAPH_DEBUG << "Pad value must be a constant";
            return false;
        }

        const auto& matched_conv =
            std::static_pointer_cast<ngraph::op::Convolution>(pattern_map[conv_label]);
        const auto& matched_pad = std::static_pointer_cast<ngraph::op::Pad>(pattern_map[pad_label]);

        if (!zero_padded_conv_consistency_check(m.get_match_root(),
                                                pad_value_op,
                                                pattern_map[pad_input],
                                                matched_pad,
                                                matched_conv->get_padding_below(),
                                                matched_conv->get_padding_above(),
                                                0,
                                                1))
        {
            return false;
        }

        CoordinateDiff padding_below{
            static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_below().at(2)),
            static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_below().at(3))};
        CoordinateDiff padding_above{
            static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_above().at(2)),
            static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_above().at(3))};

        auto zero_padded_conv =
            std::make_shared<ngraph::op::Convolution>(pattern_map[pad_input],
                                                      pattern_map[conv_filter],
                                                      matched_conv->get_window_movement_strides(),
                                                      matched_conv->get_window_dilation_strides(),
                                                      padding_below,
                                                      padding_above,
                                                      matched_conv->get_data_dilation_strides());

        ngraph::replace_node(m.get_match_root(), zero_padded_conv);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_label, "CoreFusion.ZeroPaddedConv");
    this->add_matcher(m, callback);
}

void pass::CoreFusion::construct_conv_bias()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = make_shared<pattern::op::Label>(element::f32, shape);
    auto pbias = make_shared<pattern::op::Label>(element::f32, Shape{});

    auto pbcast = make_shared<op::Broadcast>(pbias, shape, AxisSet{0, 1, 2, 3});
    auto pbcast_label = make_shared<pattern::op::Label>(pbcast, nullptr, NodeVector{pbcast});
    auto reshape_pred = [](shared_ptr<Node> node) -> bool {
        if (auto reshape = as_type_ptr<op::Reshape>(node))
        {
            auto ishape = reshape->get_input_shape(0);
            auto oshape = reshape->get_shape();
            // Callback will check that broadcast happens along channel (1) dimension.
            // Reshape should not alter that
            if (!reshape->get_is_transpose() && ishape.size() > 1 && oshape.size() > 1 &&
                ishape[0] == oshape[0] && ishape[1] == oshape[1])
            {
                return true;
            }
        }
        return false;
    };
    auto pskip = make_shared<pattern::op::Skip>(pbcast_label, reshape_pred);

    auto pconv1 = make_shared<op::Convolution>(data_batch,
                                               filters,
                                               Strides{1, 1},
                                               Strides{1, 1},
                                               CoordinateDiff{0, 0},
                                               CoordinateDiff{0, 0},
                                               Strides{1, 1});
    auto p_conv_bias = pskip + pconv1;

    auto callback = [pbcast_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_conv_bias against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto conv_m = as_type_ptr<op::Convolution>(m.get_match_root()->get_argument(0));

        if (conv_m == nullptr)
        {
            conv_m = static_pointer_cast<op::Convolution>(m.get_match_root()->get_argument(1));
        }

        if (conv_m->get_shape().size() > 5 || conv_m->get_element_type() != element::f32)
        {
            // Most backends are unlikely to efficiently support these convolutions. Skip fusion
            return false;
        }

        auto bcast_m = static_pointer_cast<op::Broadcast>(pattern_map[pbcast_label]);
        // Except for the 2nd axis (channel dimension), we should either be broadcasting
        // to it or the dimension size should be 1.
        auto bcast_axes = bcast_m->get_broadcast_axes();
        for (size_t i = 0; i < bcast_m->get_shape().size(); i++)
        {
            if (i != 1 && bcast_axes.find(i) == bcast_axes.end() && bcast_m->get_shape()[i] != 1)
            {
                return false;
            }
        }

        auto bias = bcast_m->get_argument(0);
        if (bias->get_shape().size() > 1)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << "conv_bias bias shape != 1, requires reshape to match filter count.";
            auto order = get_default_order(bias->get_shape());
            auto bias_reshape =
                make_shared<op::Reshape>(bias, order, Shape{conv_m->get_input_shape(1)[0]});
            auto conv_bias = shared_ptr<Node>(new op::ConvolutionBias(conv_m, bias_reshape));
            replace_node(m.get_match_root(), conv_bias);
        }
        else
        {
            auto conv_bias = shared_ptr<Node>(new op::ConvolutionBias(conv_m, bias));
            replace_node(m.get_match_root(), conv_bias);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_conv_bias, "CoreFusion.ConvBias");
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

void pass::CoreFusion::construct_conv_bias_add()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = make_shared<pattern::op::Label>(element::f32, shape);
    auto bias = make_shared<pattern::op::Label>(element::f32, Shape{shape[0]});

    auto pconv = make_shared<op::ConvolutionBias>(data_batch,
                                                  filters,
                                                  bias,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
    auto add_input = make_shared<pattern::op::Label>(element::f32, pconv->get_shape());
    auto padd = make_shared<op::Add>(add_input, pconv);

    auto callback = [data_batch, filters](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_sum against "
                     << m.get_match_root()->get_name();

        auto add_m = m.get_match_root();
        auto pattern_map = m.get_pattern_map();
        auto conv_m = as_type_ptr<op::ConvolutionBias>(add_m->get_argument(1));
        auto add_input_m = add_m->get_argument(0);

        if (!conv_m)
        {
            conv_m = static_pointer_cast<op::ConvolutionBias>(add_m->get_argument(0));
            add_input_m = add_m->get_argument(1);
        }

        if (get_user_count(conv_m.get()) > 1)
        {
            NGRAPH_DEBUG << "Convolution has more than one user";
            return false;
        }

        auto conv_add = shared_ptr<Node>(new op::ConvolutionBiasAdd(conv_m, add_input_m, false));
        replace_node(m.get_match_root(), conv_add);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(padd, "CoreFusion.ConvBiasAdd");
    this->add_matcher(m, callback, all_pass_property_off);
}
