// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_subgraph.hpp"

#include <sstream>

#include "../logging.hpp"
#include "../moe_transformations/moe_transformation.hpp"
#include "../partitioning/partitioning.hpp"
#include "../partitioning/patterns/moe.hpp"
#include "../serialization.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace npuw {
namespace moe {
namespace {

const char* behavior_name(const BehaviorRole role) {
    switch (role) {
    case BehaviorRole::EXPERTS:
        return ov::npuw::patterns::moe::GPTOSSExpert::pattern_name();
    case BehaviorRole::DOWNSTREAM:
        return "MoEDownstream";
    }
    OPENVINO_THROW("Unsupported MoE behavior role");
}

ov::npuw::v1::subgraphs::RuntimeBehaviorFactory make_runtime_factory() {
    return [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
        class MoEBehavior final : public ov::npuw::v1::subgraphs::ISubgraphBehavior {
        public:
            explicit MoEBehavior(const BehaviorRole role) : m_role(role) {}

            void prologue(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                if (ctx.opaque_prologue) {
                    ctx.opaque_prologue();
                }
            }

            void run(ov::npuw::v1::subgraphs::InferContext& ctx) override {
                if (m_role != BehaviorRole::EXPERTS) {
                    ctx.legacy_infer();
                    return;
                }

                OPENVINO_ASSERT(static_cast<bool>(ctx.opaque_run),
                                "Expected opaque run callback for MoE subgraph behavior");
                ctx.opaque_run();
            }

        private:
            BehaviorRole m_role = BehaviorRole::EXPERTS;
        };

        return std::make_unique<MoEBehavior>(ctx.get<BehaviorRole>());
    };
}

void attach_runtime_behavior(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                             ov::npuw::v1::subgraphs::Context& compiled_context,
                             const BehaviorRole role,
                             const bool handles_function_prologue) {
    compiled_context.put<BehaviorRole>(role);
    compiled_pipeline.registration.group = ov::npuw::patterns::moe::GPTOSSExpert::group_name();
    compiled_pipeline.registration.name = behavior_name(role);
    ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
    spec.registration = compiled_pipeline.registration;
    spec.context = compiled_context;
    spec.factory = make_runtime_factory();
    spec.handles_function_prologue = handles_function_prologue;
    compiled_pipeline.runtime_behavior = std::move(spec);
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

std::shared_ptr<ov::Model> find_router_model(ov::npuw::v1::subgraphs::Context& ctx) {
    auto* callbacks = ctx.get_if<ov::npuw::v1::subgraphs::PartitioningCallbacks>();
    if (callbacks == nullptr || !callbacks->find_tagged_model) {
        return nullptr;
    }
    return callbacks->find_tagged_model(ov::npuw::patterns::moe::ROUTER_TAG);
}

void transform_experts(ov::npuw::Function& function,
                       ov::npuw::v1::subgraphs::Context& ctx,
                       const std::size_t moe_chunk_size) {
    auto router_model = find_router_model(ctx);
    if (!router_model) {
        return;
    }
    auto experts = ov::npuw::function::MoEExperts::from(function._model, router_model, moe_chunk_size);
    if (experts.has_value()) {
        ctx.put<ov::npuw::function::MoEExperts>(std::move(experts.value()));
    }
}

void compile_transformed_expert_models(const CompiledExpertsState& runtime_experts,
                                       ov::npuw::v1::subgraphs::CompileContext& compile_ctx) {
    OPENVINO_ASSERT(runtime_experts != nullptr, "Expected compiled MoE experts state");
    LOG_INFO("Compiling MoE expert models...");
    LOG_BLOCK();
    for (const auto& entry : runtime_experts->_models_to_compile) {
        runtime_experts->set_compiled_model(
            entry.first,
            compile_ctx.compile_model(entry.second, "/moe_chunk_" + std::to_string(entry.first), compile_ctx.devices));
    }
    const auto& compiled_models = runtime_experts->_compiled_models;
    OPENVINO_ASSERT(!compiled_models.empty(), "Expected at least one compiled MoE expert model");
    compile_ctx.compiled_model = compiled_models.begin()->second;
}

// Configure the generic compile stage to compile all expert variants produced by transform_experts().
void configure_expert_compile(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                              ov::npuw::v1::subgraphs::Context& compiled_context) {
    auto* experts = compiled_context.get_if<ov::npuw::function::MoEExperts>();
    if (experts == nullptr) {
        return;
    }
    auto compiled_experts = std::make_shared<ov::npuw::compiled::MoEExperts>(*experts);
    const auto& models_to_compile = compiled_experts->_models_to_compile;
    OPENVINO_ASSERT(!models_to_compile.empty(), "Fatal: MoEExperts has no models to compile!");
    put_compiled_experts(compiled_context, compiled_experts);

    attach_runtime_behavior(compiled_pipeline, compiled_context, BehaviorRole::EXPERTS, true);
    compiled_pipeline.compile_executor = [compiled_experts](ov::npuw::v1::subgraphs::CompileContext& compile_ctx) {
        compile_transformed_expert_models(compiled_experts, compile_ctx);
    };
}

void transform_downstream(ov::npuw::Function& function, ov::npuw::v1::subgraphs::Context& ctx) {
    auto router_model = find_router_model(ctx);
    if (!router_model) {
        return;
    }
    auto downstream = ov::npuw::function::create_moe_downstream(function._model, router_model);
    if (downstream.has_value()) {
        ctx.put<ov::npuw::function::MoEDownstream>(std::move(downstream.value()));
    }
}

void compile_transformed_downstream_model(const ov::npuw::v1::subgraphs::CompileExecutor& previous_compile_executor,
                                          const CompiledDownstreamState& runtime_downstream,
                                          ov::npuw::v1::subgraphs::CompileContext& compile_ctx) {
    OPENVINO_ASSERT(runtime_downstream != nullptr, "Expected compiled MoE downstream state");
    compile_ctx.model = runtime_downstream->_model_to_compile;
    if (previous_compile_executor) {
        previous_compile_executor(compile_ctx);
    } else {
        compile_ctx.compiled_model = compile_ctx.compile_model(compile_ctx.model, "", compile_ctx.devices);
    }
    OPENVINO_ASSERT(compile_ctx.compiled_model, "Expected compiled model before wrapping MoE downstream");
    runtime_downstream->set_compiled_model(std::move(compile_ctx.compiled_model));
    compile_ctx.compiled_model = runtime_downstream->_compiled_model;
}

// Configure the generic compile stage to wrap the default compilation result into the transformed downstream model.
void configure_downstream_compile(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                                  ov::npuw::v1::subgraphs::Context& compiled_context) {
    auto* downstream = compiled_context.get_if<ov::npuw::function::MoEDownstream>();
    if (downstream == nullptr) {
        return;
    }
    auto compiled_downstream = std::make_shared<ov::npuw::compiled::MoEDownstream>(*downstream);
    put_compiled_downstream(compiled_context, compiled_downstream);

    attach_runtime_behavior(compiled_pipeline, compiled_context, BehaviorRole::DOWNSTREAM, true);
    const auto previous_compile_executor = compiled_pipeline.compile_executor;
    compiled_pipeline.compile_executor = [previous_compile_executor,
                                          compiled_downstream](ov::npuw::v1::subgraphs::CompileContext& compile_ctx) {
        compile_transformed_downstream_model(previous_compile_executor, compiled_downstream, compile_ctx);
    };
}

}  // namespace

void put_compiled_experts(v1::subgraphs::Context& context, CompiledExpertsState state) {
    context.put<CompiledExpertsState>(std::move(state));
}

void put_compiled_downstream(v1::subgraphs::Context& context, CompiledDownstreamState state) {
    context.put<CompiledDownstreamState>(std::move(state));
}

ov::npuw::compiled::MoEExperts* get_compiled_experts(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::MoEExperts>(context);
}

const ov::npuw::compiled::MoEExperts* get_compiled_experts(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::MoEExperts>(context);
}

ov::npuw::compiled::MoEDownstream* get_compiled_downstream(v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::MoEDownstream>(context);
}

const ov::npuw::compiled::MoEDownstream* get_compiled_downstream(const v1::subgraphs::Context& context) {
    return get_compiled_state<ov::npuw::compiled::MoEDownstream>(context);
}

bool has_compiled_experts(const v1::subgraphs::CompiledPipeline& pipeline) {
    return get_compiled_experts(pipeline.context) != nullptr;
}

bool has_compiled_downstream(const v1::subgraphs::CompiledPipeline& pipeline) {
    return get_compiled_downstream(pipeline.context) != nullptr;
}

bool has_compiled_state(const v1::subgraphs::CompiledPipeline& pipeline) {
    return has_compiled_experts(pipeline) || has_compiled_downstream(pipeline);
}

void serialize_compiled_state(v1::subgraphs::Context& context,
                              ov::npuw::s11n::Stream& stream,
                              const ov::npuw::s11n::SubmodelDeserializeCtx* submodel_ctx) {
    std::optional<ov::npuw::compiled::MoEExperts> moe_experts;
    if (const auto* experts = get_compiled_experts(context)) {
        moe_experts = *experts;
    }
    stream & moe_experts;
    if (!stream.output() && moe_experts.has_value()) {
        put_compiled_experts(context, std::make_shared<ov::npuw::compiled::MoEExperts>(moe_experts.value()));
    }
    auto* mutable_moe_experts = get_compiled_experts(context);
    if (mutable_moe_experts != nullptr) {
        size_t num_compiled_models = 0;
        if (stream.output()) {
            num_compiled_models = mutable_moe_experts->_compiled_models.size();
        }
        stream & num_compiled_models;

        if (stream.output()) {
            for (const auto& [chunk_size, compiled_model] : mutable_moe_experts->_compiled_models) {
                stream & chunk_size;
                bool has_model = compiled_model != nullptr;
                stream & has_model;
                if (has_model) {
                    std::stringstream ss;
                    compiled_model->export_model(ss);
                    std::string model_str = ss.str();
                    stream & model_str;
                }
            }
        } else {
            NPUW_ASSERT(submodel_ctx != nullptr);
            for (size_t i = 0; i < num_compiled_models; ++i) {
                size_t chunk_size = 0;
                stream & chunk_size;
                bool has_model = false;
                stream & has_model;
                if (has_model) {
                    std::string model_str;
                    stream & model_str;
                    std::stringstream ss(model_str);
                    mutable_moe_experts->_compiled_models[chunk_size] =
                        submodel_ctx->plugin->get_core()->import_model(ss,
                                                                       submodel_ctx->device,
                                                                       submodel_ctx->import_config);
                    LOG_DEBUG("Imported MoE compiled model for chunk_size=" << chunk_size);
                }
            }
            LOG_DEBUG("Deserialized " << mutable_moe_experts->_compiled_models.size() << " MoE expert models");
        }
    }

    std::optional<ov::npuw::compiled::MoEDownstream> moe_downstream;
    if (const auto* downstream = get_compiled_downstream(context)) {
        moe_downstream = *downstream;
    }
    stream & moe_downstream;
    if (!stream.output() && moe_downstream.has_value()) {
        put_compiled_downstream(context, std::make_shared<ov::npuw::compiled::MoEDownstream>(moe_downstream.value()));
    }
    auto* mutable_moe_downstream = get_compiled_downstream(context);
    if (mutable_moe_downstream != nullptr) {
        bool has_model = false;
        if (stream.output()) {
            has_model = mutable_moe_downstream->_compiled_model != nullptr;
        }
        stream & has_model;
        if (has_model) {
            if (stream.output()) {
                std::stringstream ss;
                mutable_moe_downstream->_compiled_model->export_model(ss);
                std::string model_str = ss.str();
                stream & model_str;
            } else {
                NPUW_ASSERT(submodel_ctx != nullptr);
                std::string model_str;
                stream & model_str;
                std::stringstream ss(model_str);
                mutable_moe_downstream->_compiled_model =
                    submodel_ctx->plugin->get_core()->import_model(ss,
                                                                   submodel_ctx->device,
                                                                   submodel_ctx->import_config);
                LOG_DEBUG("Imported MoE downstream compiled model");
            }
        }
    }
}

std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> register_patterns(
    ov::npuw::v1::subgraphs::PatternRegistry& registry,
    const std::size_t moe_chunk_size) {
    std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> registrations;
    registrations.reserve(3);

    registrations.emplace_back(registry.on<ov::npuw::patterns::moe::GPTOSSRouter>().scoped());

    registrations.emplace_back(
        registry.on<ov::npuw::patterns::moe::GPTOSSExpert>()
            .at_partition([moe_chunk_size](ov::npuw::Function& function, ov::npuw::v1::subgraphs::Context& ctx) {
                transform_experts(function, ctx, moe_chunk_size);
            })
            .at_compile([](ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                           ov::npuw::v1::subgraphs::Context& compiled_context) {
                configure_expert_compile(compiled_pipeline, compiled_context);
            })
            .scoped());

    ov::npuw::v1::subgraphs::PatternRegistration downstream_registration;
    downstream_registration.partition_stage = [](ov::npuw::Function& function, ov::npuw::v1::subgraphs::Context& ctx) {
        transform_downstream(function, ctx);
    };
    downstream_registration.compile_stage = [](ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                                               ov::npuw::v1::subgraphs::Context& compiled_context) {
        configure_downstream_compile(compiled_pipeline, compiled_context);
    };
    registrations.emplace_back(registry.add(std::move(downstream_registration)));

    return registrations;
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
