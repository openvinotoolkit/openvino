// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "attn_subgraph.hpp"

#include "../attention.hpp"
#include "../just_sync_infer_request.hpp"
#include "../logging.hpp"
#include "../partitioning/partitioning.hpp"
#include "../partitioning/patterns/sdpa.hpp"

namespace ov {
namespace npuw {
namespace attn {
namespace {

ov::npuw::JustInferRequest& get_request(ov::npuw::v1::subgraphs::InferContext& ctx) {
    auto* request = dynamic_cast<ov::npuw::JustInferRequest*>(&ctx.infer_request);
    OPENVINO_ASSERT(request != nullptr, "Expected JustInferRequest for attention runtime behavior");
    return *request;
}

BehaviorKind get_behavior_kind(const ov::npuw::v1::subgraphs::Context& ctx) {
    if (const auto* kind = ctx.get_if<BehaviorKind>()) {
        return *kind;
    }
    OPENVINO_THROW("Attention behavior context is missing BehaviorKind");
}

ov::npuw::v1::subgraphs::RuntimeBehaviorFactory make_runtime_factory() {
    return [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
        class AttnBehavior final : public ov::npuw::v1::subgraphs::ISubgraphBehavior {
        public:
            explicit AttnBehavior(BehaviorKind kind) : m_kind(kind) {}

            bool bind_function_input(ov::npuw::v1::subgraphs::InferContext& ctx,
                                     std::size_t input_idx,
                                     const ov::SoPtr<ov::ITensor>& tensor) override {
                auto& request = get_request(ctx);
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                    return request.behavior_bind_dynamic_input(ctx.real_subgraph_idx,
                                                               ctx.subgraph_idx,
                                                               input_idx,
                                                               tensor);
                case BehaviorKind::Pyramid:
                    return request.behavior_bind_pyramid_input(ctx.real_subgraph_idx,
                                                               ctx.subgraph_idx,
                                                               input_idx,
                                                               tensor);
                case BehaviorKind::HFA:
                    return request.behavior_bind_hfa_input(ctx.subgraph_idx, input_idx, tensor);
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

            bool bind_function_output(ov::npuw::v1::subgraphs::InferContext& ctx,
                                      std::size_t output_idx,
                                      const ov::SoPtr<ov::ITensor>& tensor) override {
                if (m_kind != BehaviorKind::HFA) {
                    return false;
                }
                auto& request = get_request(ctx);
                return request.behavior_bind_hfa_output(ctx.subgraph_idx, output_idx, tensor);
            }

            void prologue(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                if (ctx.opaque_prologue) {
                    ctx.opaque_prologue();
                }
                auto& request = get_request(ctx);
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                    request.behavior_prologue_dynamic(ctx.real_subgraph_idx, ctx.subgraph_idx);
                    return;
                case BehaviorKind::Pyramid:
                    request.behavior_prologue_pyramid(ctx.real_subgraph_idx, ctx.subgraph_idx);
                    return;
                case BehaviorKind::HFA:
                    return;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

            void run(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                auto& request = get_request(ctx);
                switch (m_kind) {
                case BehaviorKind::Dynamic:
                case BehaviorKind::Pyramid:
                    ctx.legacy_infer();
                    return;
                case BehaviorKind::HFA:
                    request.behavior_run_hfa(ctx.real_subgraph_idx, ctx.subgraph_idx);
                    return;
                }
                OPENVINO_THROW("Unsupported attention behavior kind");
            }

        private:
            BehaviorKind m_kind;
        };

        return std::make_unique<AttnBehavior>(get_behavior_kind(ctx));
    };
}

}  // namespace

void attach_runtime_behavior(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                             ov::npuw::v1::subgraphs::Context& compiled_context,
                             BehaviorKind kind) {
    compiled_pipeline.registration.group = ov::npuw::patterns::attn::SDPA::group_name();
    compiled_pipeline.registration.name = ov::npuw::patterns::attn::SDPA::pattern_name();
    compiled_context.put<BehaviorKind>(kind);
    ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
    spec.registration = compiled_pipeline.registration;
    spec.context = compiled_context;
    spec.factory = make_runtime_factory();
    spec.handles_function_prologue = true;
    compiled_pipeline.runtime_behavior = std::move(spec);
}

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
    attn_behavior.partition_stage = [](ov::npuw::Function& f, ov::npuw::v1::subgraphs::Context& ctx) {
        if (f._attention.has_value()) {
            ctx.put<ov::npuw::compiled::Attention>(ov::npuw::compiled::Attention(f._attention.value(), f._model));
            ctx.put<BehaviorKind>(BehaviorKind::Dynamic);
            return;
        }
        if (f._pyramid_attention.has_value()) {
            ctx.put<BehaviorKind>(BehaviorKind::Pyramid);
            return;
        }
        if (f._host_flash_attention.has_value()) {
            ctx.put<BehaviorKind>(BehaviorKind::HFA);
        }
    };
    attn_behavior.compile_stage = [](ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                                     ov::npuw::v1::subgraphs::Context& compiled_context) {
        const auto* kind = compiled_context.get_if<BehaviorKind>();
        if (kind == nullptr) {
            return;
        }
        attach_runtime_behavior(compiled_pipeline, compiled_context, *kind);
    };
    registrations.emplace_back(registry.add(std::move(attn_behavior)));

    return registrations;
}

}  // namespace attn
}  // namespace npuw
}  // namespace ov
