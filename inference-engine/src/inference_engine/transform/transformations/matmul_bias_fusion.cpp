// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "matmul_bias_fusion.hpp"

#include <ngraph_ops/matmul_bias.hpp>
#include <ngraph/builder/make_constant.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/pattern/op/skip.hpp>
#include <ngraph/util.hpp>
#include <ngraph.hpp>

void ngraph::pass::MatMulBiasFusion::construct_matmulbias() {
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{2, 1};
    auto W = std::make_shared<pattern::op::Label>(element::f32, shape_w);
    auto x = std::make_shared<pattern::op::Label>(element::f32, shape_x);
    auto b = std::make_shared<pattern::op::Label>(element::f32, shape_b);

//    auto pbias = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
//    auto shp = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
//    auto axs = std::make_shared<pattern::op::Label>(element::i64, Shape{1});

    auto pmmb = std::make_shared<ngraph::op::Dot>(W, x);  //, nullptr, W->get_shape(), x->get_shape(), false, false);
//    auto pbroadcast = std::make_shared<pattern::op::Label>(b);
    auto padd = pmmb + b;

    ngraph::graph_rewrite_callback callback = [W, x](pattern::Matcher &m) {
        auto mpattern = m.get_match_root();
        auto pattern_map = m.get_pattern_map();

        //  Detect which input
        auto dot_m = std::dynamic_pointer_cast<ngraph::op::Dot>(m.get_match_root()->get_argument(0));
        auto bcast_node = m.get_match_root()->get_argument(1);

        if (dot_m == nullptr) {
            dot_m = std::dynamic_pointer_cast<ngraph::op::Dot>(m.get_match_root()->get_argument(1));
            bcast_node = m.get_match_root()->get_argument(0);
        }

        auto bias_m = bcast_node;
        if (auto bcast_m = std::dynamic_pointer_cast<ngraph::op::DynBroadcast>(bcast_node)) {
            bias_m = bcast_m->get_argument(0);
        }

        if (!std::dynamic_pointer_cast<ngraph::op::Constant>(bias_m)) {
            return false;
        }

        Shape shape_w;
        auto reshapeLayer = std::dynamic_pointer_cast<ngraph::op::DynReshape>(pattern_map[W]);
        if (reshapeLayer) {
            auto constLayer = std::dynamic_pointer_cast<ngraph::op::Constant>(reshapeLayer->get_inputs()[1].get_output().get_node());
            if (!constLayer)
                throw;
            auto constDims = constLayer->get_data_ptr<std::int64_t>();
            for (size_t i = 0; i < ngraph::shape_size(constLayer->get_shape()); i++)
                shape_w.emplace_back(constDims[i]);
        } else {
            shape_w = pattern_map[W]->get_shape();
        }

        auto mmb = std::make_shared<ngraph::op::MatmulBias>(pattern_map[W],
                                                            pattern_map[x],
                                                            bias_m,
                                                            pattern_map[W]->get_shape(),
                                                            pattern_map[x]->get_shape(),
                                                            false, false, AxisSet{0});

        mmb->set_friendly_name(m.get_match_root()->get_friendly_name());

        ngraph::replace_node(m.get_match_root(), mmb);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(padd, "CPUFusion.MatMulBias");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
