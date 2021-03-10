// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pruning.hpp"
#include "transformations/rt_info/mask_attribute.hpp"
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/pass/constant_folding.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::Pruning, "Pruning", 0);

bool ngraph::pass::Pruning::run_on_function(std::shared_ptr<Function> f) {
    Manager manager(get_pass_config());
    manager.register_pass<PropagateMasks>();

    auto modifier = [](Node& node, std::vector<std::string>& attributes) {
        std::stringstream ss;
        for (auto input : node.input_values()) {
            if (auto mask = getMask(input)) {
                if (!mask->all_dims_are_empty()) {
                    attributes.emplace_back("color=green");
                    attributes.emplace_back("penwidth=2");
                }
                ss << *mask << "\\n";
            }
        }
        if (!ss.str().empty()) {
            attributes.push_back("label=\"" + ss.str() + "\"");
        }
    };
    manager.register_pass<VisualizeTree>("/tmp/before.svg", modifier);
    manager.register_pass<ShrinkWeights>();
    manager.register_pass<ConstantFolding>();
    manager.register_pass<VisualizeTree>("/tmp/after.svg");
    manager.run_passes(f);
    return true;
}