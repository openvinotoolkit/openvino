// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

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
        size_t index{0};
        for (auto output : node.outputs()) {
            if (auto mask = getMask(output)) {
                if (!mask->all_dims_are_empty()) {
                    attributes.emplace_back("color=green");
                    attributes.emplace_back("penwidth=2");
                }
                ss << "Mask(" << index << ") : " << *mask << "\\n";
            }
            index++;
        }
        if (!ss.str().empty()) {
            auto label = std::find_if(attributes.begin(), attributes.end(),
                                   [](const std::string & value) { return value.find("label=") != std::string::npos; });
            if (label != attributes.end()) {
                label->pop_back();
                *label += "\n" + ss.str() + "\"";
            } else {
                attributes.push_back("label=\"" + ss.str() + "\"");
            }
        }
    };
    manager.register_pass<VisualizeTree>("/tmp/before.svg", modifier);
    manager.register_pass<ShrinkWeights>();
    manager.register_pass<ConstantFolding>();
    manager.register_pass<VisualizeTree>("/tmp/after.svg");
    manager.run_passes(f);
    return true;
}