// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pruning.hpp"

#include <algorithm>

#include "mask_attribute.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/log.hpp"

bool ov::pass::Pruning::run_on_model(const std::shared_ptr<ov::Model>& f) {
    Manager manager(get_pass_config());

    // Initialize masks only for Convolutions/GroupConvolutions weights (needed to init mask in source Constant of
    // weights-calculating subgraph). For other node types masks initialized in PropagateMasks pass.
    manager.register_pass<InitMasks>();
    manager.register_pass<PropagateMasks>();

#ifdef ENABLE_OPENVINO_DEBUG
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

#ifdef ENABLE_OPENVINO_DEBUG
    // Uncomment following line and change path to resulting svg file
    // manager.register_pass<VisualizeTree>("/tmp/after.svg");
#endif

    manager.run_passes(f);
    return true;
}
