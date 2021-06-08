// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "pruning.hpp"
#include "mask_attribute.hpp"

#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/log.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::Pruning, "Pruning", 0);

bool ngraph::pass::Pruning::run_on_function(std::shared_ptr<Function> f) {
    Manager manager(get_pass_config());

    // Initialize masks only for Convolutions/GroupConvolutions weights (needed to init mask in source Constant of
    // weights-calculating subgraph). For other node types masks initialized in PropagateMasks pass.
    manager.register_pass<InitMasks>();
    manager.register_pass<PropagateMasks>();


#ifdef NGRAPH_DEBUG_ENABLE
    // VisualizeTree modifier helps to print Masks and mark nodes with masks
    /*
    auto modifier = [](const Node& node, std::vector<std::string>& attributes) {
        std::stringstream ss;
        size_t index{0};
        for (const auto & output : node.outputs()) {
            if (const auto & mask = getMask(output)) {
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
    */

    // Uncomment modifier above and following line and change path to resulting svg file
    // manager.register_pass<VisualizeTree>("/tmp/before.svg", modifier);
#endif

    manager.register_pass<ShrinkWeights>();

#ifdef NGRAPH_DEBUG_ENABLE
    // Uncomment following line and change path to resulting svg file
    // manager.register_pass<VisualizeTree>("/tmp/after.svg");
#endif

    manager.run_passes(f);
    return true;
}