// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "projector.hpp"

#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"

namespace example {
namespace {
using namespace ov::frontend::gguf;

// A small non-decoder family: project a variable number of embedding vectors. This source uses
// only installed builder API headers and is compiled unchanged in the plugin and catalog tests.
class Projector : public ModelBuilder {
public:
    explicit Projector(const BuildContext& context) : m_context(context) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext graph(m_context);
        const auto weight = graph.tensors().require("projection.weight");
        OPENVINO_ASSERT(weight.ne(0) > 0, "Projector requires a static input width");
        const auto input = graph.add_input("embeddings", ov::element::f32, {1, 1, -1, weight.ne(0)});
        graph.set_output(graph.node("GGML_OP_MUL_MAT", {weight, input}));
        return graph.finish();
    }

private:
    BuildContext m_context;
};
}  // namespace

ov::frontend::gguf::ArchitectureDefinition projector_architecture() {
    return {"example.projector", "example-projector", [](const ov::frontend::gguf::BuildContext& context) {
                return std::make_shared<Projector>(context);
            }};
}
ov::frontend::gguf::ProjectorDefinition vision_projector_component() {
    using namespace ov::frontend::gguf;
    return {
        "clip.vision.example-adapter",
        "clip",
        "vision",
        "example-adapter",
        [](GgufGraphContext& graph) {
            const auto weight = graph.tensors().require("projection.weight");
            const auto input = graph.add_input("vision.embeddings_input", ov::element::f32, {1, 1, -1, weight.ne(0)});
            return ProjectorResult{graph.node("GGML_OP_MUL_MAT", {weight, input}), {}, ProjectorLayout::Image};
        },
        {},
        {},
        [](const std::shared_ptr<ov::Model>& model) {
            auto result = model->get_results().front();
            auto output = std::make_shared<ov::op::v1::Add>(result->input_value(0),
                                                            ov::op::v0::Constant::create(ov::element::f32, {}, {7.f}));
            output->output(0).set_names({"last_hidden_state"});
            result->input(0).replace_source_output(output);
            return ModelAdapterResult{model, {{"custom_vision", model}}};
        }};
}
}  // namespace example
