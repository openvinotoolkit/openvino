// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "v1/subgraph_pipeline.hpp"

#include <gtest/gtest.h>

#include <string>

#include "attn/attn_subgraph.hpp"
#include "moe/moe_executor.hpp"
#include "partitioning/partitioning.hpp"
#include "partitioning/patterns/moe.hpp"
#include "partitioning/patterns/sdpa.hpp"

namespace {

struct TestPayload {
    int value = 0;
    std::string name;

    TestPayload() = default;
    TestPayload(int value, std::string name) : value(value), name(std::move(name)) {}
};

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
