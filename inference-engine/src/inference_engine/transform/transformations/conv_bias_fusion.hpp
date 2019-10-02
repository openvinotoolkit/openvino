// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph.hpp>

#include "ngraph/pattern/matcher.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/add.hpp"

#include "ngraph_ops/group_conv_bias.hpp"
#include "ngraph/op/fused/group_conv.hpp"

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class ConvBiasFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvBiasFusion: public ngraph::pass::GraphRewrite {
public:
    ConvBiasFusion() : GraphRewrite() {
        construct_conv_bias<ngraph::op::Convolution>();
        construct_conv_bias<ngraph::op::GroupConvolution>();
    }

private:
    template <class T>
    void construct_conv_bias();

    template <class T>
    ngraph::graph_rewrite_callback get_callback();
};


template <class T>
ngraph::graph_rewrite_callback ngraph::pass::ConvBiasFusion::get_callback() {
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) {
        static_assert(std::is_same<T, ngraph::op::Convolution>() || std::is_same<T, ngraph::op::GroupConvolution>(),
                      "This callback works only with ngraph::op::Convolution and ngraph::op::GroupConvolution");

        auto pattern_map = m.get_pattern_map();

        auto conv_m = std::dynamic_pointer_cast<T>(m.get_match_root()->get_argument(0));
        auto bcast_m = std::dynamic_pointer_cast<ngraph::op::DynBroadcast>(m.get_match_root()->get_argument(1));

        if (conv_m == nullptr) {
            conv_m = std::dynamic_pointer_cast<T>(m.get_match_root()->get_argument(1));
            bcast_m = std::dynamic_pointer_cast<ngraph::op::DynBroadcast>(m.get_match_root()->get_argument(0));
        }

        //  Check that DynBroadcast has only Constant inputs
        //  otherwise we can't guaranty that bias fusion will be valid
        for (const auto &inp : bcast_m->get_inputs()) {
            bool is_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(inp.get_output().get_node()) != nullptr;
            if (!is_constant) {
                return false;
            }
        }

        // Except for the 2nd axis (channel dimension), we should either be broadcasting
        // to it or the dimension size should be 1.
        //        auto axes_node = std::dynamic_pointer_cast<ngraph::op::Constant> (bcast_m->get_inputs()[2].get_output().get_node());
        //        auto axes_vector = axes_node->get_vector<axes_node->get_element_type()>();
        //        for (size_t i = 0; i < bcast_m->get_shape().size(); i++) {
        //            if (i != 1 && bcast_axes->get_data_ptr().find(i) == bcast_axes.end() && bcast_m->get_shape()[i] != 1) {
        //                return false;
        //            }
        //        }

        auto bias = bcast_m->get_argument(0);
        auto bias_shape = bias->get_shape();
        std::shared_ptr<ngraph::Node> conv_bias = nullptr;

        if (bias_shape.size() > 1) {
            auto order = ngraph::get_default_order(bias_shape);
            auto bias_reshape = std::make_shared<ngraph::op::Reshape>(bias, order,
                                                                      ngraph::Shape{conv_m->get_input_shape(1)[0]});
            bias = std::dynamic_pointer_cast<ngraph::Node>(bias_reshape);
        }

        if (std::is_same<T, ngraph::op::GroupConvolution>()) {
            auto g_conv = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(conv_m);
            conv_bias = std::shared_ptr<ngraph::Node>(new ngraph::op::GroupConvolutionBias(g_conv,
                                                                                           bias,
                                                                                           g_conv->get_groups(),
                                                                                           g_conv->get_output_shape(0),
                                                                                           false,
                                                                                           1.0));
        } else {
            auto conv = std::dynamic_pointer_cast<ngraph::op::Convolution>(conv_m);
            conv_bias = std::shared_ptr<ngraph::Node>(new ngraph::op::ConvolutionBias(conv, bias, false));
        }

        conv_bias->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), conv_bias);

        return true;
    };
    return callback;
}

template <class T>
void ngraph::pass::ConvBiasFusion::construct_conv_bias() {
    static_assert(std::is_same<T, ngraph::op::Convolution>() || std::is_same<T, ngraph::op::GroupConvolution>(),
                  "This transformation works only with ngraph::op::Convolution and ngraph::op::GroupConvolution");

    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 2, 1, 1});
    auto filters = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 2, 1, 1});
    auto pbias = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto shp = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto axs = std::make_shared<pattern::op::Label>(element::i64, Shape{1});

    auto pbroadcast = std::make_shared<ngraph::op::DynBroadcast>(pbias, shp, axs);

    std::shared_ptr<ngraph::Node> pconv = nullptr;

    if (std::is_same<T, ngraph::op::GroupConvolution>()) {
        auto pconv1 = std::make_shared<ngraph::op::GroupConvolution>(data_batch,
                                                                     filters,
                                                                     Strides{1, 1},
                                                                     Strides{1, 1},
                                                                     CoordinateDiff{0, 0},
                                                                     CoordinateDiff{0, 0},
                                                                     Strides{1, 1}, 1);
        pconv = std::dynamic_pointer_cast<ngraph::Node>(pconv1);
    } else {
        auto pconv1 = std::make_shared<ngraph::op::Convolution>(data_batch,
                                                                filters,
                                                                Strides{1, 1},
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{1, 1});
        pconv = std::dynamic_pointer_cast<ngraph::Node>(pconv1);
    }

    auto p_conv_bias = pbroadcast + pconv;

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_conv_bias, "CPUFusion.ConvBias");
    this->add_matcher(m, get_callback<T>(), PassProperty::CHANGE_DYNAMIC_STATE);
}
