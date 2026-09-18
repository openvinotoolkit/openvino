// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"

#include <unordered_set>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"

namespace ov::frontend::gguf::pass {
bool AdaptMmprojToGenAI::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const std::string prefix = m_modality == Modality::Vision ? "vision." : "audio.";
    std::shared_ptr<ov::op::v0::Result> selected;
    for (const auto& result : model->get_results()) {
        if (result->input_value(0).get_names().count(prefix + "embeddings"))
            selected = result;
    }
    OPENVINO_ASSERT(selected, "[GGUF] mmproj has no ", prefix, "embeddings output");
    std::unordered_set<ov::Node*> reachable;
    std::vector<std::shared_ptr<ov::Node>> stack{selected};
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!reachable.insert(node.get()).second)
            continue;
        for (const auto& input : node->input_values())
            stack.push_back(input.get_node_shared_ptr());
    }
    const auto results = model->get_results();
    for (const auto& result : results)
        model->remove_result(result);
    auto embeddings = std::make_shared<ov::op::v0::Squeeze>(selected->input_value(0),
                                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    embeddings->output(0).set_names({m_modality == Modality::Vision ? "image_features" : "audio_features"});
    size_t auxiliary_count = 0;
    if (m_modality == Modality::Vision && model->has_rt_info({"gguf_mmproj", "vision.auxiliary_count"}))
        auxiliary_count = std::stoull(model->get_rt_info<std::string>({"gguf_mmproj", "vision.auxiliary_count"}));
    if (auxiliary_count) {
        auto split = std::make_shared<ov::op::v1::Split>(embeddings,
                                                         ov::op::v0::Constant::create(ov::element::i64, {}, {2}),
                                                         auxiliary_count + 1);
        for (size_t i = 0; i <= auxiliary_count; ++i) {
            split->output(i).set_names({i == 0 ? "image_features" : "deepstack_features." + std::to_string(i - 1)});
            model->add_results({std::make_shared<ov::op::v0::Result>(split->output(i))});
        }
    } else {
        model->add_results({std::make_shared<ov::op::v0::Result>(embeddings)});
    }
    const auto params = model->get_parameters();
    for (const auto& p : params) {
        if (!reachable.count(p.get())) {
            model->remove_parameter(p);
        } else {
            auto name = p->get_friendly_name();
            if (name.rfind(prefix, 0) == 0) {
                name = name.substr(prefix.size());
                p->set_friendly_name(name);
                p->output(0).set_names({name});
            }
        }
    }
    model->validate_nodes_and_infer_types();
    return true;
}
}  // namespace ov::frontend::gguf::pass
