// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "projector.hpp"

#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace example {
namespace {
using namespace ov::frontend::gguf;

// A small non-decoder family: project a variable number of embedding vectors. This source uses
// only installed SDK headers and is compiled unchanged in the plugin and catalog tests.
class Projector : public ModelBuilder {
public:
    explicit Projector(const BuildContext& context) : m_context(context) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext graph(m_context);
        const auto weight = graph.tensors().require("projection.weight");
        OPENVINO_ASSERT(weight.ne(0) > 0, "Projector requires a static input width");
        const auto input = graph.add_input("embeddings", ov::element::f32, {1, 1, -1, weight.ne(0)});
        graph.set_output(graph.mul_mat(weight, input));
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
}  // namespace example
