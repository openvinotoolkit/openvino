// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "attn_subgraph.hpp"

#include <sstream>

#include "../attention.hpp"
#include "../host_flash_attention.hpp"
#include "../just_sync_infer_request.hpp"
#include "../logging.hpp"
#include "../partitioning/partitioning.hpp"
#include "../partitioning/patterns/sdpa.hpp"
#include "../pyramid_attention.hpp"
#include "../serialization.hpp"

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

template <typename T>
T* get_compiled_state(ov::npuw::v1::subgraphs::Context& context) {
    auto* state = context.get_if<std::shared_ptr<T>>();
    return state == nullptr ? nullptr : state->get();
}

template <typename T>
const T* get_compiled_state(const ov::npuw::v1::subgraphs::Context& context) {
    auto* state = context.get_if<std::shared_ptr<T>>();
    return state == nullptr ? nullptr : state->get();
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

void put_compiled_dynamic(v1::subgraphs::Context& context, CompiledDynamicState state) {
    context.put<CompiledDynamicState>(std::move(state));
}

void put_compiled_pyramid(v1::subgraphs::Context& context, CompiledPyramidState state) {
    context.put<CompiledPyramidState>(std::move(state));
}

void put_compiled_hfa(v1::subgraphs::Context& context, CompiledHFAState state) {
    context.put<CompiledHFAState>(std::move(state));
}

ov::npuw::compiled::Attention* get_compiled_dynamic(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::Attention>(context);
}

const ov::npuw::compiled::Attention* get_compiled_dynamic(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::Attention>(context);
}

ov::npuw::compiled::PyramidAttention* get_compiled_pyramid(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::PyramidAttention>(context);
}

const ov::npuw::compiled::PyramidAttention* get_compiled_pyramid(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::PyramidAttention>(context);
}

ov::npuw::compiled::HostFlashAttention* get_compiled_hfa(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::HostFlashAttention>(context);
}

const ov::npuw::compiled::HostFlashAttention* get_compiled_hfa(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::HostFlashAttention>(context);
}

bool has_compiled_state(const v1::subgraphs::CompiledPipeline& pipeline) {
    return get_compiled_dynamic(pipeline.context) != nullptr || get_compiled_pyramid(pipeline.context) != nullptr ||
           get_compiled_hfa(pipeline.context) != nullptr;
}

void serialize_compiled_state(v1::subgraphs::Context& context,
                              ov::npuw::s11n::Stream& stream,
                              const ov::npuw::s11n::SubmodelDeserializeCtx* submodel_ctx) {
    std::optional<ov::npuw::compiled::Attention> dynamic;
    if (const auto* state = get_compiled_dynamic(context)) {
        dynamic = *state;
    }
    stream & dynamic;
    if (stream.input() && dynamic.has_value()) {
        put_compiled_dynamic(context, std::make_shared<ov::npuw::compiled::Attention>(dynamic.value()));
    }

    std::optional<ov::npuw::compiled::PyramidAttention> pyramid;
    if (const auto* state = get_compiled_pyramid(context)) {
        pyramid = *state;
    }
    stream & pyramid;
    if (stream.input() && pyramid.has_value()) {
        put_compiled_pyramid(context, std::make_shared<ov::npuw::compiled::PyramidAttention>(pyramid.value()));
    }

    auto* mutable_pyramid = get_compiled_pyramid(context);
    if (mutable_pyramid != nullptr) {
        size_t num_models = 0;
        if (stream.output()) {
            num_models = mutable_pyramid->_compiled_models.size();
        }
        stream & num_models;

        if (stream.output()) {
            for (size_t i = 0; i < num_models - 1; ++i) {
                std::stringstream ss;
                mutable_pyramid->_compiled_models[i]->export_model(ss);
                std::string model_str = ss.str();
                stream & model_str;
            }
        } else if (num_models > 0) {
            mutable_pyramid->_compiled_models.resize(num_models);
            NPUW_ASSERT(submodel_ctx != nullptr);
            for (size_t i = 0; i < num_models - 1; ++i) {
                std::string model_str;
                stream & model_str;
                std::stringstream ss(model_str);
                mutable_pyramid->_compiled_models[i] =
                    submodel_ctx->plugin->get_core()->import_model(ss,
                                                                   submodel_ctx->device,
                                                                   submodel_ctx->import_config);
            }
            if (submodel_ctx->compiled_model) {
                mutable_pyramid->_compiled_models[num_models - 1] = submodel_ctx->compiled_model;
                LOG_DEBUG("Reused compiled_model for the last pyramid attention model");
            }
        }
    }

    std::optional<ov::npuw::compiled::HostFlashAttention> hfa;
    if (const auto* state = get_compiled_hfa(context)) {
        hfa = *state;
    }
    stream & hfa;
    if (stream.input() && hfa.has_value()) {
        put_compiled_hfa(context, std::make_shared<ov::npuw::compiled::HostFlashAttention>(hfa.value()));
    }

    auto* mutable_hfa = get_compiled_hfa(context);
    if (mutable_hfa != nullptr) {
        bool has_compiled_model = false;
        if (stream.output()) {
            has_compiled_model = mutable_hfa->_compiled_tile_model != nullptr;
        }
        stream & has_compiled_model;
        if (has_compiled_model) {
            if (stream.output()) {
                std::stringstream ss;
                mutable_hfa->_compiled_tile_model->export_model(ss);
                std::string model_str = ss.str();
                stream & model_str;
            } else {
                NPUW_ASSERT(submodel_ctx != nullptr);
                std::string model_str;
                stream & model_str;
                std::stringstream ss(model_str);
                mutable_hfa->_compiled_tile_model = submodel_ctx->plugin->get_core()->import_model(ss,
                                                                                                    submodel_ctx->device,
                                                                                                    submodel_ctx->import_config);
                LOG_DEBUG("Imported compiled tile model for host flash attention");
            }
        }
        if (stream.input()) {
            NPUW_ASSERT(submodel_ctx != nullptr);
            mutable_hfa->_compiled_final_tile_model = submodel_ctx->compiled_model;
            LOG_DEBUG("Set compiled final tile model reference for host flash attention");
        }
    }
}

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
            put_compiled_dynamic(ctx,
                                 std::make_shared<ov::npuw::compiled::Attention>(f._attention.value(), f._model));
            ctx.put<BehaviorKind>(BehaviorKind::Dynamic);
            return;
        }
        if (f._pyramid_attention.has_value()) {
            put_compiled_pyramid(ctx,
                                 std::make_shared<ov::npuw::compiled::PyramidAttention>(f._pyramid_attention.value()));
            ctx.put<BehaviorKind>(BehaviorKind::Pyramid);
            return;
        }
        if (f._host_flash_attention.has_value()) {
            put_compiled_hfa(
                ctx,
                std::make_shared<ov::npuw::compiled::HostFlashAttention>(f._host_flash_attention.value()));
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
