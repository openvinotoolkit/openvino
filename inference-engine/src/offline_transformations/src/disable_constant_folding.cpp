// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "disable_constant_folding.hpp"

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/variant.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::DisablePriorBoxConstantFolding, "DisablePriorBoxConstantFolding", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::DisablePriorBoxClusteredConstantFolding, "DisablePriorBoxClusteredConstantFolding", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::DisableShapeOfConstantFolding, "DisableShapeOfConstantFolding", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::EnableShapeOfConstantFolding, "EnableShapeOfConstantFolding", 0);

ngraph::pass::DisableShapeOfConstantFolding::DisableShapeOfConstantFolding() {
    auto shape_of = ngraph::pattern::wrap_type<ngraph::opset5::ShapeOf>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto shape_of = m.get_match_root();
        auto & rt_info = shape_of->get_rt_info();
        rt_info["DISABLED_CONSTANT_FOLDING"] = std::make_shared<ngraph::VariantWrapper<std::string>>("");
        std::cout << "DISABLED CF FOR:" << shape_of->get_friendly_name() << std::endl;
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shape_of, "DisableShapeOfConstantFolding");
    register_matcher(m, callback);
}

ngraph::pass::EnableShapeOfConstantFolding::EnableShapeOfConstantFolding() {
    auto shape_of = ngraph::pattern::wrap_type<ngraph::opset5::ShapeOf>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto shape_of = m.get_match_root();
        auto & rt_info = shape_of->get_rt_info();
        rt_info.erase("DISABLED_CONSTANT_FOLDING");
        std::cout << "ENABLED CF FOR:" << shape_of->get_friendly_name() << std::endl;
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shape_of, "EnableShapeOfConstantFolding");
    register_matcher(m, callback);
}

ngraph::pass::DisablePriorBoxConstantFolding::DisablePriorBoxConstantFolding() {
    auto prior_box = ngraph::pattern::wrap_type<ngraph::opset5::PriorBox>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto find_shape_of = [](Output<Node> output) {
            while (true) {
                auto node = output.get_node_shared_ptr();
                if (auto shape_of = std::dynamic_pointer_cast<opset5::ShapeOf>(node)) {
                    DisableShapeOfConstantFolding().apply(shape_of);
                    return;
                }
                if (node->get_input_size() > 0) {
                    output = node->input_value(0);
                } else {
                    return;
                }
            }
        };
        find_shape_of(node->input_value(0));
        find_shape_of(node->input_value(1));
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prior_box, "DisableShapeOfConstantFolding");
    register_matcher(m, callback);
}

ngraph::pass::DisablePriorBoxClusteredConstantFolding::DisablePriorBoxClusteredConstantFolding() {
    auto prior_box = ngraph::pattern::wrap_type<ngraph::opset5::PriorBoxClustered>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto find_shape_of = [](Output<Node> output) {
            while (true) {
                auto node = output.get_node_shared_ptr();
                if (auto shape_of = std::dynamic_pointer_cast<opset5::ShapeOf>(node)) {
                    DisableShapeOfConstantFolding().apply(shape_of);
                    return;
                }
                if (node->get_input_size() > 0) {
                    output = node->input_value(0);
                } else {
                    return;
                }
            }
        };
        find_shape_of(node->input_value(0));
        find_shape_of(node->input_value(1));
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prior_box, "DisablePriorBoxClusteredConstantFolding");
    register_matcher(m, callback);
}