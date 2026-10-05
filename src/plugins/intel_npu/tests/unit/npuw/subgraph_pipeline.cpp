// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "v1/subgraph_pipeline.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "attn/attn_subgraph.hpp"
#include "host_flash_attention.hpp"
#include "moe/moe_executor.hpp"
#include "moe/moe_subgraph.hpp"
#include "moe_transformations/moe_transformation.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "partitioning/partitioning.hpp"
#include "partitioning/patterns/moe.hpp"
#include "partitioning/patterns/sdpa.hpp"
#include "pyramid_attention.hpp"

namespace {

struct TestPayload {
    int value = 0;
    std::string name;

    TestPayload() = default;
    TestPayload(int value, std::string name) : value(value), name(std::move(name)) {}
};

class NullPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        return {};
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        return {};
    }
};

// Stands in for a device-level compiled model: only its identity matters to these tests
class FakeCompiledModel final : public ov::ICompiledModel {
public:
    FakeCompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return nullptr;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return nullptr;
    }
};

ov::SoPtr<ov::ICompiledModel> make_fake_compiled_model() {
    static const auto plugin = std::make_shared<NullPlugin>();
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto result = std::make_shared<ov::op::v0::Result>(param);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    return {std::make_shared<FakeCompiledModel>(model, plugin), {}};
}

// Runs the pipeline's extra-compiled-models hook and returns what it visited, in order
std::vector<const ov::ICompiledModel*> visit_extra_compiled_models(
    const ov::npuw::v1::subgraphs::CompiledPipeline& pipeline) {
    std::vector<const ov::ICompiledModel*> visited;
    pipeline.for_each_extra_compiled_model(pipeline.context, [&](const ov::SoPtr<ov::ICompiledModel>& cm) {
        visited.push_back(cm._ptr.get());
    });
    return visited;
}

}  // namespace

TEST(SubgraphPipelineContextTest, StoresAndRetrievesTypedValues) {
    ov::npuw::v1::subgraphs::Context ctx;

    auto& label = ctx.put<std::string>("moe");
    auto& payload = ctx.emplace<TestPayload>(7, "router");

    EXPECT_EQ(label, "moe");
    EXPECT_EQ(payload.value, 7);
    EXPECT_EQ(payload.name, "router");
    EXPECT_TRUE(ctx.contains<std::string>());
    EXPECT_TRUE(ctx.contains<TestPayload>());
    EXPECT_EQ(ctx.size(), 2u);
    EXPECT_EQ(ctx.get<std::string>(), "moe");

    const auto* found = ctx.get_if<TestPayload>();
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->value, 7);
    EXPECT_EQ(found->name, "router");
}

TEST(SubgraphPipelineContextTest, MissingTypeReturnsNull) {
    ov::npuw::v1::subgraphs::Context ctx;

    EXPECT_FALSE(ctx.contains<int>());
    EXPECT_EQ(ctx.get_if<int>(), nullptr);
    EXPECT_TRUE(ctx.empty());
}

TEST(SubgraphPipelineFunctionTest, FunctionTagSeedsPipelineRegistrationPattern) {
    ov::npuw::Function function;

    function.settag("GPTOSSExpert");

    EXPECT_EQ(function.gettag(), "GPTOSSExpert");
    ASSERT_EQ(function._pipeline.registration.patterns.size(), 1u);
    EXPECT_EQ(function._pipeline.registration.patterns.front(), "GPTOSSExpert");
}

TEST(SubgraphPipelineBehaviorTest, FactoryCreatesBehaviorObject) {
    auto behavior = ov::npuw::v1::subgraphs::make_direct_behavior();

    EXPECT_NE(behavior, nullptr);
}

TEST(SubgraphPipelineBehaviorTest, RealPatternRegistrationBuildsRuntimeBehaviorForTaggedSubgraph) {
    ov::npuw::Subgraph subgraph;
    subgraph.settag(ov::npuw::patterns::attn::SDPA::isolation_tag());
    ov::npuw::v1::subgraphs::PatternRegistry registry;

    auto scoped_registration =
        registry.on<ov::npuw::patterns::attn::SDPA>()
            .at_compile([](ov::npuw::v1::subgraphs::CompiledPipeline&, ov::npuw::v1::subgraphs::Context& ctx) {
                ctx.put<std::string>("marker");
            })
            .at_runtime(
                [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
                    EXPECT_EQ(ctx.get<std::string>(), "marker");
                    return ov::npuw::v1::subgraphs::make_direct_behavior();
                })
            .scoped();

    registry.apply(subgraph);

    ASSERT_TRUE(static_cast<bool>(subgraph._pipeline.compile_stage));
    ov::npuw::v1::subgraphs::CompiledPipeline compiled_pipeline;
    compiled_pipeline.registration = subgraph._pipeline.registration;
    compiled_pipeline.context = subgraph._pipeline.context;
    subgraph._pipeline.compile_stage(compiled_pipeline, compiled_pipeline.context);

    ASSERT_TRUE(compiled_pipeline.runtime_behavior.has_value());
    EXPECT_EQ(compiled_pipeline.registration.name, ov::npuw::patterns::attn::SDPA::pattern_name());
    EXPECT_EQ(compiled_pipeline.runtime_behavior->registration.name, ov::npuw::patterns::attn::SDPA::pattern_name());
    auto behavior = compiled_pipeline.runtime_behavior->factory(compiled_pipeline.runtime_behavior->context);
    EXPECT_NE(behavior, nullptr);
}

TEST(SubgraphPipelineBehaviorTest, MoERegistrationBuildsDeferredPartitionPipelineForExpertFunction) {
    ov::npuw::Function function;
    function.settag(ov::npuw::patterns::moe::GPTOSSExpert::isolation_tag());

    ov::npuw::v1::subgraphs::PatternRegistry registry;
    auto registrations = ov::npuw::moe::register_patterns(registry, 0u);
    registry.apply(function);

    ASSERT_TRUE(static_cast<bool>(function._pipeline.partition_stage));
    ASSERT_TRUE(static_cast<bool>(function._pipeline.compile_stage));
    EXPECT_EQ(function._pipeline.registration.group, ov::npuw::patterns::moe::GPTOSSExpert::group_name());
    EXPECT_EQ(function._pipeline.registration.name, ov::npuw::patterns::moe::GPTOSSExpert::pattern_name());
    EXPECT_NE(std::find(function._pipeline.registration.patterns.begin(),
                        function._pipeline.registration.patterns.end(),
                        ov::npuw::patterns::moe::GPTOSSExpert::pattern_name()),
              function._pipeline.registration.patterns.end());
}

// Verify that attn::register_patterns() chains partition_stage and compile_stage on any
// Function whose isolation tag matches SDPA::isolation_tag() ("attn").
TEST(SubgraphPipelineBehaviorTest, AttnRegistrationSetsPartitionAndCompileStagesForAttnTaggedFunction) {
    ov::npuw::Function function;
    function.settag(ov::npuw::patterns::attn::SDPA::isolation_tag());

    ov::npuw::v1::subgraphs::PatternRegistry registry;
    auto registrations = ov::npuw::attn::register_patterns(registry);
    registry.apply(function);

    ASSERT_TRUE(static_cast<bool>(function._pipeline.partition_stage));
    ASSERT_TRUE(static_cast<bool>(function._pipeline.compile_stage));
    // The registration must carry the SDPA pattern name so the compile loop can identify it.
    EXPECT_NE(std::find(function._pipeline.registration.patterns.begin(),
                        function._pipeline.registration.patterns.end(),
                        ov::npuw::patterns::attn::SDPA::isolation_tag()),
              function._pipeline.registration.patterns.end());
}

// Verify that the compile_stage does NOT attach a runtime behavior when the partition_stage
// was never given a compiled::Attention context entry — i.e. when f._attention is not set
// (NPUW_ATTN=STATIC, or the model has no dynamic dims in the attention function).
TEST(SubgraphPipelineBehaviorTest, AttnCompileStageSkipsRuntimeBehaviorWhenAttentionNotDynamic) {
    ov::npuw::Function function;
    function.settag(ov::npuw::patterns::attn::SDPA::isolation_tag());

    ov::npuw::v1::subgraphs::PatternRegistry registry;
    auto registrations = ov::npuw::attn::register_patterns(registry);
    registry.apply(function);

    ASSERT_TRUE(static_cast<bool>(function._pipeline.partition_stage));
    ASSERT_TRUE(static_cast<bool>(function._pipeline.compile_stage));

    // Run partition_stage WITHOUT setting f._attention — simulates NPUW_ATTN=STATIC or a
    // model where function::Attention::from() found no dynamic dims.
    function._pipeline.partition_stage(function, function._pipeline.context);
    EXPECT_FALSE(function._pipeline.context.contains<ov::npuw::compiled::Attention>())
        << "partition_stage must not put compiled::Attention in context when f._attention is unset";

    // Run compile_stage: with no compiled::Attention in context, no runtime behavior should appear.
    ov::npuw::v1::subgraphs::CompiledPipeline compiled;
    compiled.registration = function._pipeline.registration;
    compiled.context = function._pipeline.context;
    function._pipeline.compile_stage(compiled, compiled.context);

    EXPECT_FALSE(compiled.runtime_behavior.has_value())
        << "compile_stage must not attach DynAttnBehavior when compiled::Attention is absent";
}

TEST(SubgraphPipelineBehaviorTest, AttnCompileStageAttachesPyramidBehaviorWhenHintIsPresent) {
    ov::npuw::Function function;
    function.settag(ov::npuw::patterns::attn::SDPA::isolation_tag());

    ov::npuw::v1::subgraphs::PatternRegistry registry;
    auto registrations = ov::npuw::attn::register_patterns(registry);
    registry.apply(function);

    function._pipeline.context.put<ov::npuw::attn::BehaviorKind>(ov::npuw::attn::BehaviorKind::Pyramid);

    ov::npuw::v1::subgraphs::CompiledPipeline compiled;
    compiled.registration = function._pipeline.registration;
    compiled.context = function._pipeline.context;
    function._pipeline.compile_stage(compiled, compiled.context);

    ASSERT_TRUE(compiled.runtime_behavior.has_value());
    auto behavior = compiled.runtime_behavior->factory(compiled.runtime_behavior->context);
    EXPECT_NE(behavior, nullptr);
}

TEST(SubgraphPipelineBehaviorTest, AttnCompileStageAttachesHFABehaviorWhenHintIsPresent) {
    ov::npuw::Function function;
    function.settag(ov::npuw::patterns::attn::SDPA::isolation_tag());

    ov::npuw::v1::subgraphs::PatternRegistry registry;
    auto registrations = ov::npuw::attn::register_patterns(registry);
    registry.apply(function);

    function._pipeline.context.put<ov::npuw::attn::BehaviorKind>(ov::npuw::attn::BehaviorKind::HFA);

    ov::npuw::v1::subgraphs::CompiledPipeline compiled;
    compiled.registration = function._pipeline.registration;
    compiled.context = function._pipeline.context;
    function._pipeline.compile_stage(compiled, compiled.context);

    ASSERT_TRUE(compiled.runtime_behavior.has_value());
    auto behavior = compiled.runtime_behavior->factory(compiled.runtime_behavior->context);
    EXPECT_NE(behavior, nullptr);
}

// The extra-compiled-models hook is set by attach_runtime_behavior(), which runs both at
// compile time (before the extra models are compiled) and on import (after the state is
// restored). In every test below the models are filled in only after attaching, so the
// hook must look them up when it is called rather than capture them when it is set.

TEST(SubgraphPipelineExtraCompiledModelsTest, AttnPyramidVisitsAllPyramidModels) {
    ov::npuw::v1::subgraphs::CompiledPipeline pipeline;
    auto pyramid = std::make_shared<ov::npuw::compiled::PyramidAttentionContiguous>();
    ov::npuw::attn::put_compiled_pyramid(pipeline.context, pyramid);
    ov::npuw::attn::attach_runtime_behavior(pipeline, pipeline.context, ov::npuw::attn::BehaviorKind::Pyramid);
    ASSERT_TRUE(static_cast<bool>(pipeline.for_each_extra_compiled_model));

    const auto level_0 = make_fake_compiled_model();
    const auto level_1 = make_fake_compiled_model();
    const auto level_2 = make_fake_compiled_model();
    pyramid->_compiled_models = {level_0, level_1, level_2};

    EXPECT_EQ(visit_extra_compiled_models(pipeline),
              (std::vector<const ov::ICompiledModel*>{level_0._ptr.get(), level_1._ptr.get(), level_2._ptr.get()}));
}

TEST(SubgraphPipelineExtraCompiledModelsTest, AttnHfaVisitsRegularAndFinalTiles) {
    ov::npuw::v1::subgraphs::CompiledPipeline pipeline;
    auto hfa = std::make_shared<ov::npuw::compiled::HostFlashAttention>();
    ov::npuw::attn::put_compiled_hfa(pipeline.context, hfa);
    ov::npuw::attn::attach_runtime_behavior(pipeline, pipeline.context, ov::npuw::attn::BehaviorKind::HFA);
    ASSERT_TRUE(static_cast<bool>(pipeline.for_each_extra_compiled_model));

    const auto tile = make_fake_compiled_model();
    const auto final_tile = make_fake_compiled_model();
    hfa->set_compiled_tile_model(tile);
    hfa->set_compiled_final_tile_model(final_tile);

    EXPECT_EQ(visit_extra_compiled_models(pipeline),
              (std::vector<const ov::ICompiledModel*>{tile._ptr.get(), final_tile._ptr.get()}));
}

TEST(SubgraphPipelineExtraCompiledModelsTest, AttnDynamicVisitsNothing) {
    ov::npuw::v1::subgraphs::CompiledPipeline pipeline;
    ov::npuw::attn::attach_runtime_behavior(pipeline, pipeline.context, ov::npuw::attn::BehaviorKind::Dynamic);
    ASSERT_TRUE(static_cast<bool>(pipeline.for_each_extra_compiled_model));

    // Dynamic attention runs on the subgraph's own compiled model only
    EXPECT_TRUE(visit_extra_compiled_models(pipeline).empty());
}

TEST(SubgraphPipelineExtraCompiledModelsTest, MoeExpertsVisitsEveryChunkModel) {
    ov::npuw::v1::subgraphs::CompiledPipeline pipeline;
    auto experts = std::make_shared<ov::npuw::compiled::MoEExperts>();
    ov::npuw::moe::put_compiled_experts(pipeline.context, experts);
    ov::npuw::moe::attach_runtime_behavior(pipeline, pipeline.context, ov::npuw::moe::BehaviorRole::EXPERTS, true);
    ASSERT_TRUE(static_cast<bool>(pipeline.for_each_extra_compiled_model));

    const auto chunk_64 = make_fake_compiled_model();
    const auto chunk_128 = make_fake_compiled_model();
    experts->_compiled_models[64] = chunk_64;
    experts->_compiled_models[128] = chunk_128;

    EXPECT_EQ(visit_extra_compiled_models(pipeline),
              (std::vector<const ov::ICompiledModel*>{chunk_64._ptr.get(), chunk_128._ptr.get()}));
}

TEST(SubgraphPipelineExtraCompiledModelsTest, MoeDownstreamVisitsItsModel) {
    ov::npuw::v1::subgraphs::CompiledPipeline pipeline;
    auto downstream = std::make_shared<ov::npuw::compiled::MoEDownstream>();
    ov::npuw::moe::put_compiled_downstream(pipeline.context, downstream);
    ov::npuw::moe::attach_runtime_behavior(pipeline, pipeline.context, ov::npuw::moe::BehaviorRole::DOWNSTREAM, true);
    ASSERT_TRUE(static_cast<bool>(pipeline.for_each_extra_compiled_model));

    const auto downstream_model = make_fake_compiled_model();
    downstream->_compiled_model = downstream_model;

    EXPECT_EQ(visit_extra_compiled_models(pipeline),
              (std::vector<const ov::ICompiledModel*>{downstream_model._ptr.get()}));
}
