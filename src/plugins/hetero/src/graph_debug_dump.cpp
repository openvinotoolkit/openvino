// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_debug_dump.hpp"

#include "openvino/pass/visualize_tree.hpp"

namespace ov {
namespace hetero {
namespace debug {
static const std::vector<std::string> colors = {
    "aliceblue",
    "antiquewhite4",
    "aquamarine4",
    "azure4",
    "bisque3",
    "blue1",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk4",
};

void dump_affinities(const std::shared_ptr<ov::Model>& model,
                     const std::map<std::string, std::string>& supported_ops_map,
                     const std::unordered_set<std::string>& devices) {
    const auto& name = model->get_friendly_name();
    // clang-format off
    ov::pass::VisualizeTree{
        "hetero_affinity_" + name + ".dot",
        [&](const ov::Node& node, std::vector<std::string>& attributes) {
            const auto& nodeDevice = supported_ops_map.at(node.get_friendly_name());
            int colorIndex = 0;
            for (auto&& device : devices) {
                if (device == nodeDevice) {
                    attributes.push_back(std::string {"fillcolor="} + colors[colorIndex % colors.size()] +
                                         " style=filled");
                    auto itLabel =
                        std::find_if(std::begin(attributes), std::end(attributes), [](const std::string& str) {
                            return str.find("label") != std::string::npos;
                        });
                    auto label = "\\ndevice=" + supported_ops_map.at(node.get_friendly_name()) + '\"';
                    OPENVINO_ASSERT(itLabel != attributes.end());
                    itLabel->pop_back();
                    (*itLabel) += label;
                    break;
                }
                colorIndex++;
            }
        }}
        .run_on_model(model);
    // clang-format on
}

void dump_subgraphs(const std::shared_ptr<ov::Model>& model,
                    const std::map<std::string, std::string>& supported_ops_map,
                    const std::map<std::string, int>& map_id) {
    const auto& name = model->get_friendly_name();
    // clang-format off
    ov::pass::VisualizeTree{
        "hetero_subgraphs_" + name + ".dot",
        [&](const ov::Node& node, std::vector<std::string>& attributes) {
            attributes.push_back(std::string {"fillcolor="} +
                                 colors[map_id.at(node.get_friendly_name()) % colors.size()] + " style=filled");
            auto itLabel = std::find_if(std::begin(attributes), std::end(attributes), [](const std::string& str) {
                return str.find("label") != std::string::npos;
            });
            auto label = "\\nsubgraph=" + std::to_string(map_id.at(node.get_friendly_name())) + "\\n" +
                         "device=" + supported_ops_map.at(node.get_friendly_name()) + '\"';
            OPENVINO_ASSERT(itLabel != attributes.end());
            itLabel->pop_back();
            (*itLabel) += label;
        }}
        .run_on_model(model);
    // clang-format on
}
}  // namespace debug
}  // namespace hetero
}  // namespace ov
