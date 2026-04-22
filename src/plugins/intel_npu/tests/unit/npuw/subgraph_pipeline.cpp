// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "partitioning/partitioning.hpp"
#include "partitioning/patterns/sdpa.hpp"
#include "v1/subgraph_pipeline.hpp"

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

    auto scoped_registration = registry.on<ov::npuw::patterns::attn::SDPA>()
                                   .at_compile([](ov::npuw::v1::subgraphs::CompiledPipeline&,
                                                  ov::npuw::v1::subgraphs::Context& ctx) {
                                       ctx.put<std::string>("marker");
                                   })
                                   .at_runtime([](const ov::npuw::v1::subgraphs::Context& ctx)
                                                   -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
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
    EXPECT_EQ(compiled_pipeline.runtime_behavior->registration.name,
              ov::npuw::patterns::attn::SDPA::pattern_name());
    auto behavior = compiled_pipeline.runtime_behavior->factory(compiled_pipeline.runtime_behavior->context);
    EXPECT_NE(behavior, nullptr);
}
