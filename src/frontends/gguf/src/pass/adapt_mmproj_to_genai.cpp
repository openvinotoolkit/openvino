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
#include "pass/prune_orphaned_parameters.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::pass {
bool AdaptMmprojToGenAI::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const std::string prefix = m_modality == Modality::Vision ? "vision." : "audio.";
    std::shared_ptr<ov::op::v0::Result> selected;
    for (const auto& result : model->get_results()) {
        if (result->input_value(0).get_names().count(prefix + "embeddings"))
            selected = result;
    }
    OPENVINO_ASSERT(selected, "[GGUF] mmproj has no ", prefix, "embeddings output");
    auto live_before = std::make_shared<std::unordered_set<const ov::Node*>>();
    SnapshotLiveParameters(live_before).run_on_model(model);
    const auto results = model->get_results();
    for (const auto& result : results)
        model->remove_result(result);
    auto embeddings = std::make_shared<ov::op::v0::Squeeze>(selected->input_value(0),
                                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
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
        // Only the tensor that becomes a Result is named; the Split path names output 0 instead.
        embeddings->output(0).set_names({m_modality == Modality::Vision ? "image_features" : "audio_features"});
        model->add_results({std::make_shared<ov::op::v0::Result>(embeddings)});
    }
    PruneParametersOrphanedSince(live_before).run_on_model(model);
    for (const auto& p : model->get_parameters()) {
        const auto& name = p->get_friendly_name();
        if (name.rfind(prefix, 0) == 0)
            name_output(p, name.substr(prefix.size()));
    }
    model->validate_nodes_and_infer_types();
    return true;
}
}  // namespace ov::frontend::gguf::pass
