// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "attn_subgraph.hpp"

#include "../attention.hpp"
#include "../logging.hpp"
#include "../partitioning/partitioning.hpp"
#include "../partitioning/patterns/sdpa.hpp"

namespace ov {
namespace npuw {
namespace attn {
namespace {

ov::npuw::v1::subgraphs::RuntimeBehaviorFactory make_runtime_factory() {
    return [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
        class DynAttnBehavior final : public ov::npuw::v1::subgraphs::ISubgraphBehavior {
        public:
            void prologue(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                if (ctx.opaque_prologue) {
                    ctx.opaque_prologue();
                }
            }

            void run(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                ctx.legacy_infer();
            }
        };

        return std::make_unique<DynAttnBehavior>();
    };
}

void attach_runtime_behavior(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                             ov::npuw::v1::subgraphs::Context& compiled_context) {
    compiled_pipeline.registration.group = ov::npuw::patterns::attn::SDPA::group_name();
    compiled_pipeline.registration.name = ov::npuw::patterns::attn::SDPA::pattern_name();
    ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
    spec.registration = compiled_pipeline.registration;
    spec.context = compiled_context;
    spec.factory = make_runtime_factory();
    spec.handles_function_prologue = true;
    compiled_pipeline.runtime_behavior = std::move(spec);
}

}  // namespace

std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> register_patterns(
    ov::npuw::v1::subgraphs::PatternRegistry& registry) {
    std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> registrations;
    registrations.reserve(3);

    // Matcher-only registrations so the registry handles SDPA/SDPADecomposed isolation
    // (replaces the legacy HNDL_ATTN fallback in snapshot.cpp)
    registrations.emplace_back(registry.on<ov::npuw::patterns::attn::SDPA>().scoped());
    registrations.emplace_back(registry.on<ov::npuw::patterns::attn::SDPADecomposed>().scoped());

    // Behavior registration: fires for any function tagged "attn" (i.e. from either SDPA
    // or SDPADecomposed pattern).  The at_partition callback checks whether dynamic
    // attention was detected (f._attention set by Partitioner::attention()) and, if so,
    // stashes the compile-time descriptor in the pipeline context for later stages.
    ov::npuw::v1::subgraphs::PatternRegistration attn_behavior;
    attn_behavior.tag = ov::npuw::patterns::attn::SDPA::isolation_tag();
    attn_behavior.partition_stage =
        [](ov::npuw::Function& f, ov::npuw::v1::subgraphs::Context& ctx) {
            if (!f._attention.has_value()) {
                return;
            }
            ctx.put<ov::npuw::compiled::Attention>(
                ov::npuw::compiled::Attention(f._attention.value(), f._model));
        };
    attn_behavior.compile_stage =
        [](ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
           ov::npuw::v1::subgraphs::Context& compiled_context) {
            if (!compiled_context.contains<ov::npuw::compiled::Attention>()) {
                return;
            }
            attach_runtime_behavior(compiled_pipeline, compiled_context);
        };
    registrations.emplace_back(registry.add(std::move(attn_behavior)));

    return registrations;
}

}  // namespace attn
}  // namespace npuw
}  // namespace ov
