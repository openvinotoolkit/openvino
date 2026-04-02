// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "partitioning/online/compiler.hpp"
#include "partitioning/online/group.hpp"
#include "partitioning/online/snapshot.hpp"

using ov::test::npuw::LLMConfig;
using ov::test::npuw::ModelBuilder;

namespace {

::intel_npu::Config createConfigWithKeepBlockSize(std::size_t size) {
    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    auto cfg = ::intel_npu::Config(opt_desc);
    ::intel_npu::registerNPUWOptions(*opt_desc);
    std::map<std::string, std::string> cfg_map = {{"NPUW_ONLINE_KEEP_BLOCK_SIZE", std::to_string(size)}};
    cfg.update(cfg_map);
    return cfg;
}

bool isEqualEns(ov::npuw::Ensemble& ens1, ov::npuw::Ensemble& ens2);
bool isEqualEns(ov::npuw::Ensemble& ens1, ov::npuw::Ensemble& ens2) {
    if (ens1.groups.size() != ens2.groups.size()) {
        return false;
    }

    for (auto& g : ens1.groups) {
        std::sort(g.input_layers.begin(), g.input_layers.end());
        std::sort(g.output_layers.begin(), g.output_layers.end());
        std::sort(g.all_layers.begin(), g.all_layers.end());
    }

    for (auto& g : ens2.groups) {
        std::sort(g.input_layers.begin(), g.input_layers.end());
        std::sort(g.output_layers.begin(), g.output_layers.end());
        std::sort(g.all_layers.begin(), g.all_layers.end());
    }

    std::sort(ens1.groups.begin(), ens1.groups.end(), [](const ov::npuw::Group& g1, const ov::npuw::Group& g2) {
        return g1.all_layers.front() < g2.all_layers.front();
    });

    std::sort(ens2.groups.begin(), ens2.groups.end(), [](const ov::npuw::Group& g1, const ov::npuw::Group& g2) {
        return g1.all_layers.front() < g2.all_layers.front();
    });

    for (size_t i = 0; i < ens1.groups.size(); ++i) {
        const auto& g1 = ens1.groups.at(i);
        const auto& g2 = ens2.groups.at(i);

        if (g1.avoid_list != g2.avoid_list || g1.input_layers != g2.input_layers ||
            g1.output_layers != g2.output_layers || g1.all_layers != g2.all_layers) {
            return false;
        }

        // Can't compare them directly since they are random, but dont't affect the structure
        if ((g1.repeated_id.empty() && !g2.repeated_id.empty()) ||
            (!g1.repeated_id.empty() && g2.repeated_id.empty())) {
            return false;
        }
    }

    if (ens1.repeated.size() != ens2.repeated.size()) {
        return false;
    }

    auto get_sorted_rep = [](const std::map<std::string, ov::npuw::RepeatedBlock>& rep) {
        std::vector<std::vector<std::set<std::string>>> sorted_rep;

        std::transform(rep.begin(), rep.end(), std::back_inserter(sorted_rep), [](const auto& v) {
            return v.second.matches;
        });

        for (auto& g : sorted_rep) {
            std::sort(g.begin(), g.end(), [](const auto& a, const auto& b) {
                return *a.begin() < *b.begin();
            });
        }

        std::sort(sorted_rep.begin(), sorted_rep.end(), [](const auto& a, const auto& b) {
            return *a.front().begin() < *b.front().begin();
        });

        return sorted_rep;
    };

    if (get_sorted_rep(ens1.repeated) != get_sorted_rep(ens2.repeated)) {
        return false;
    }

    return true;
}

class IsRegularResultCaseParametrized : public ::testing::TestWithParam<std::tuple<std::vector<std::size_t>, bool>> {};
class IsRegularParameterCaseParametrized : public ::testing::TestWithParam<std::tuple<std::vector<std::size_t>, bool>> {
};
class IsRegularCrossGroupConsumerCaseParametrized
    : public ::testing::TestWithParam<std::tuple<std::vector<std::size_t>, bool>> {};

};  // namespace

TEST(OnlinePartitioningTest, Partitioning_IsTheSame_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto cfg = createConfigWithKeepBlockSize(9);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    for (size_t i = 0; i < 100; ++i) {
        auto ens_again = ov::npuw::online::buildPartitioning(model, cfg);
        EXPECT_TRUE(isEqualEns(ens, ens_again));
    }
}

TEST(OnlinePartitioningTest, Partitioning_IsTheSame_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto cfg = createConfigWithKeepBlockSize(9);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    for (size_t i = 0; i < 100; ++i) {
        auto ens_again = ov::npuw::online::buildPartitioning(model, cfg);
        EXPECT_TRUE(isEqualEns(ens, ens_again));
    }
}

TEST(OnlinePartitioningTest, Partitioning_SingleGroup_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->singleGroup();
    EXPECT_EQ(snap->graphSize(), 1);
}

TEST(OnlinePartitioningTest, Partitioning_SingleGroup_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->singleGroup();
    EXPECT_EQ(snap->graphSize(), 1);
}

TEST(OnlinePartitioningTest, Partitioning_buildGraph_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();
    auto g = snap->getGraph();
    for (const auto& nh : g->sorted()) {
        ov::npuw::online::Group::GPtr group = g->meta(nh).get<ov::npuw::online::Group::GPtr>();
        EXPECT_EQ(group->size(), 1);
    }
    EXPECT_EQ(snap->getNodeToGroupMap()->size(), snap->graphSize());
}

TEST(OnlinePartitioningTest, Partitioning_buildGraph_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();
    auto g = snap->getGraph();
    for (const auto& nh : g->sorted()) {
        ov::npuw::online::Group::GPtr group = g->meta(nh).get<ov::npuw::online::Group::GPtr>();
        EXPECT_EQ(group->size(), 1);
    }
    EXPECT_EQ(snap->getNodeToGroupMap()->size(), snap->graphSize());
}

TEST(OnlinePartitioningTest, Partitioning_earlyAvoids_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    ov::npuw::online::PassContext ctx;
    ctx.avoids = {{ov::npuw::online::PatternType::OP, "Gather", "mydevice"},
                  {ov::npuw::online::PatternType::OP, "MatMul", "mydevice"}};
    snap->setCtx(ctx);
    snap->buildGraph();
    snap->earlyAvoids();
    auto g = snap->getGraph();
    size_t count = 0;
    for (const auto& nh : g->sorted()) {
        ov::npuw::online::Group::GPtr group = g->meta(nh).get<ov::npuw::online::Group::GPtr>();
        EXPECT_EQ(group->size(), 1);
        if (group->avoidedTargets().size() == 1 && *(group->avoidedTargets().begin()) == "mydevice") {
            ++count;
        }
    }
    EXPECT_EQ(count, 2);
}

TEST(OnlinePartitioningTest, Partitioning_earlyAvoids_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    ov::npuw::online::PassContext ctx;
    ctx.avoids = {{ov::npuw::online::PatternType::OP, "Gather", "mydevice"},
                  {ov::npuw::online::PatternType::OP, "MatMul", "mydevice"}};
    snap->setCtx(ctx);
    snap->buildGraph();
    snap->earlyAvoids();
    auto g = snap->getGraph();
    size_t count = 0;
    for (const auto& nh : g->sorted()) {
        ov::npuw::online::Group::GPtr group = g->meta(nh).get<ov::npuw::online::Group::GPtr>();
        EXPECT_EQ(group->size(), 1);
        if (group->avoidedTargets().size() == 1 && *(group->avoidedTargets().begin()) == "mydevice") {
            ++count;
        }
    }
    EXPECT_EQ(count, 20);
}

TEST(OnlinePartitioningTest, Partitioning_collectLHF_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {10, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->collectLHF();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_collectLHF_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {82, 82};
    size_t iter = 0;

    snap->repeat([&] {
        snap->collectLHF();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseRemnants_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {10, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseRemnants();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseRemnants_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {75, 38, 19, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseRemnants();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseRemnantsExtended_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {10, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseRemnantsExtended_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {10, 10};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseInputs_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {15, 14, 14};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseInputs();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_fuseInputs_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes = {148, 138, 138};
    size_t iter = 0;

    snap->repeat([&] {
        snap->fuseInputs();
        EXPECT_LT(iter, sizes.size());
        EXPECT_EQ(snap->graphSize(), sizes[iter++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_Just_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes_lhf = {10, 10};
    size_t iter_lhf = 0;

    std::vector<std::size_t> sizes_fr = {10, 10};
    size_t iter_fr = 0;

    snap->repeat([&] {
        snap->collectLHF();
        EXPECT_LT(iter_lhf, sizes_lhf.size());
        EXPECT_EQ(snap->graphSize(), sizes_lhf[iter_lhf++]);
    });
    snap->repeat([&] {
        snap->fuseRemnants();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_Just_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes_lhf = {82, 82};
    size_t iter_lhf = 0;

    std::vector<std::size_t> sizes_fr = {41, 21, 11, 10, 10};
    size_t iter_fr = 0;

    snap->repeat([&] {
        snap->collectLHF();
        EXPECT_LT(iter_lhf, sizes_lhf.size());
        EXPECT_EQ(snap->graphSize(), sizes_lhf[iter_lhf++]);
    });
    snap->repeat([&] {
        snap->fuseRemnants();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_RepeatedBlocks_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes_fr = {10, 10};
    size_t iter_fr = 0;

    snap->earlyAvoids();
    snap->earlyRegroup();
    snap->repeatedBlocks();
    EXPECT_EQ(snap->graphSize(), 17);

    auto matches = snap->getMatches();
    EXPECT_EQ(matches.size(), 0);

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_RepeatedBlocks_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    std::vector<std::size_t> sizes_fr = {12, 12};
    size_t iter_fr = 0;

    snap->earlyAvoids();
    snap->earlyRegroup();
    snap->repeatedBlocks();
    EXPECT_EQ(snap->graphSize(), 18);

    auto matches = snap->getMatches();
    EXPECT_EQ(matches.size(), 1);

    for (const auto& m : matches) {
        EXPECT_EQ(m.second.size(), 17);
        for (const auto& layers : m.second) {
            EXPECT_EQ(layers.size(), 10);
        }
    }

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_Compute_SmallModel) {
    ModelBuilder mb;
    auto model = mb.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);

    std::vector<std::size_t> sizes_fr = {10, 10};
    size_t iter_fr = 0;

    ov::npuw::online::PassContext ctx;
    ctx.isolates = {{ov::npuw::online::PatternType::OP, "Transpose", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "ScatterUpdate", "test_compute"}};
    ctx.nofolds = {"test_compute"};
    snap->setCtx(ctx);

    snap->buildGraph();
    snap->earlyAvoids();
    snap->earlyRegroup();
    snap->repeatedBlocks();
    EXPECT_EQ(snap->graphSize(), 17);

    auto matches = snap->getMatches();
    EXPECT_EQ(matches.size(), 0);

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Compiler_Compute_RepeatedModel) {
    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);

    std::vector<std::size_t> sizes_fr = {10, 10};
    size_t iter_fr = 0;

    ov::npuw::online::PassContext ctx;
    ctx.isolates = {{ov::npuw::online::PatternType::OP, "Gather", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "ScatterUpdate", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "ShapeOf", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "Divide", "test_compute"},
                    {ov::npuw::online::PatternType::OP, "Floor", "test_compute"}};
    ctx.nofolds = {"test_compute"};
    snap->setCtx(ctx);

    snap->buildGraph();
    snap->earlyAvoids();
    snap->earlyRegroup();
    snap->repeatedBlocks();
    EXPECT_EQ(snap->graphSize(), 38);

    // FIXME: create a config in which there will be repeated blocks
    auto matches = snap->getMatches();
    EXPECT_EQ(matches.size(), 0);

    snap->repeat([&] {
        snap->fuseRemnantsExtended();
        EXPECT_LT(iter_fr, sizes_fr.size());
        EXPECT_EQ(snap->graphSize(), sizes_fr[iter_fr++]);
    });
}

TEST(OnlinePartitioningTest, Partitioning_Avoids_Pipeline_None) {
    std::shared_ptr<ov::op::v0::Parameter> input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    input->set_friendly_name("input");

    auto n1 = std::make_shared<ov::op::v1::Add>(input, input);
    n1->set_friendly_name("n1");
    auto n2 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{1});
    n2->set_friendly_name("n2");
    auto n3 = std::make_shared<ov::op::v1::Divide>(n1, n2, true);
    n3->set_friendly_name("n3");
    auto n4 = std::make_shared<ov::op::v0::Sin>(n1);
    n4->set_friendly_name("n4");
    auto n5 = std::make_shared<ov::op::v0::Cos>(n1);
    n5->set_friendly_name("n5");
    auto n6 = std::make_shared<ov::op::v0::Sin>(n3);
    n6->set_friendly_name("n6");
    auto n7 = std::make_shared<ov::op::v0::Cos>(n3);
    n7->set_friendly_name("n7");
    auto n8 = std::make_shared<ov::op::v0::Concat>(std::vector<std::shared_ptr<ov::Node>>{n1, n4, n5, n6, n7}, -1);
    n8->set_friendly_name("n8");

    auto result = std::make_shared<ov::op::v0::Result>(n8);
    result->set_friendly_name("res");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    auto cfg = ::intel_npu::Config(opt_desc);
    ::intel_npu::registerNPUWOptions(*opt_desc);
    std::map<std::string, std::string> cfg_map = {{"NPUW_ONLINE_AVOID", "Op:Sin/NPU,Op:Cos/NPU"},
                                                  {"NPUW_ONLINE_PIPELINE", "NONE"}};
    cfg.update(cfg_map);

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(ens.groups.size(), 3);
}

TEST(OnlinePartitioningTest, IsRegularIOCaseWhenNoRB) {
    bool expected_result = false;

    ModelBuilder mb;
    std::vector<std::shared_ptr<ov::Model>> models = {mb.get_model_with_one_op(),
                                                      mb.get_model_without_repeated_blocks()};

    for (auto model : models) {
        auto cfg = createConfigWithKeepBlockSize(9);
        auto ens = ov::npuw::online::buildPartitioning(model, cfg);

        EXPECT_EQ(ens.repeated.size(), 0);  // sanity check that we don't have repeated blocks
        EXPECT_EQ(ens.irregular_io, expected_result);
    }
}

TEST(OnlinePartitioningTest, IsRegularResultCaseMultipleOutputs) {
    ModelBuilder mb;
    std::vector<std::pair<std::shared_ptr<ov::Model>, bool>> model_expected = {
        {mb.get_model_with_multi_output_repeating_blocks(10, /*irregular_io=*/true), /*irregular_io=*/true},
        {mb.get_model_with_multi_output_repeating_blocks(10, /*irregular_io=*/false),
         /*irregular_io=*/false}};

    for (auto [model, expected_result] : model_expected) {
        auto cfg = createConfigWithKeepBlockSize(3);
        auto ens = ov::npuw::online::buildPartitioning(model, cfg);

        EXPECT_EQ(ens.repeated.size(), 1);  // sanity check that we have repeated blocks
        EXPECT_EQ(ens.irregular_io, expected_result);
    }
}

TEST_P(IsRegularResultCaseParametrized, CheckForDifferentResultConfigs) {
    auto [block_indices, expected_result] = GetParam();

    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks_and_results(10, block_indices);

    auto cfg = createConfigWithKeepBlockSize(9);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(ens.repeated.size(), 1);  // sanity check that we have repeated blocks
    EXPECT_EQ(ens.irregular_io, expected_result);
}

INSTANTIATE_TEST_SUITE_P(OnlinePartitioningTest,
                         IsRegularResultCaseParametrized,
                         ::testing::Values(
                             // All blocks have an ov::Result consumer
                             std::make_tuple(std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                             /*irregular_io=*/false),
                             // Some blocks have an ov::Result consumer
                             std::make_tuple(std::vector<std::size_t>{2, 5, 8}, /*irregular_io=*/true),
                             // Only last block has an additional ov::Result consumer
                             std::make_tuple(std::vector<std::size_t>{9}, /*irregular_io=*/true),
                             // No blocks have an additional ov::Result consumers
                             std::make_tuple(std::vector<std::size_t>{}, /*irregular_io=*/false)));

TEST_P(IsRegularParameterCaseParametrized, CheckForDifferentParameterConfigs) {
    auto [block_indices, expected_result] = GetParam();

    ModelBuilder mb;
    auto model = mb.get_model_with_repeated_blocks_and_parameters(10, block_indices);
    auto cfg = createConfigWithKeepBlockSize(4);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_EQ(ens.repeated.size(), 1);  // sanity check that we have repeated blocks
    EXPECT_EQ(ens.irregular_io, expected_result);
}

INSTANTIATE_TEST_SUITE_P(OnlinePartitioningTest,
                         IsRegularParameterCaseParametrized,
                         ::testing::Values(
                             // All blocks have an ov::Parameter producer
                             std::make_tuple(std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                             /*irregular_io=*/false),
                             // Some blocks have an ov::Parameter producer
                             std::make_tuple(std::vector<std::size_t>{1, 2, 3}, /*irregular_io=*/true),
                             // Only one block has an additional ov::Parameter producer
                             std::make_tuple(std::vector<std::size_t>{5}, /*irregular_io=*/true),
                             // No blocks have an additional ov::Parameter producers
                             std::make_tuple(std::vector<std::size_t>{}, /*irregular_io=*/false)));

// Regression test for the isRegularParameterCase iteration-order bug.
//
// In a prefill-style model built with use_inputs_embeds=true, the first transformer
// layer's residual Add reads inputs_embeds (ov::Parameter) as its ONLY external
// producer -- the other input (o_proj/MatMul) is internal to the same partitioned
// group. Because isOp(Parameter)=false, updateInputLayers() excludes that Add from
// group 0's getInputs(). The old code used gset.begin() (hash-order-dependent) as
// the sole reference group: if it landed on group 0, the mismatch between layer 0's
// producers mask [true, false] and layers 1+'s mask [false, false] was invisible,
// so irregular_io was incorrectly set to false and F16IC was wrongly enabled,
// leading to a 26!=27 sanityCheck assertion failure at runtime.
//
// The fix iterates ALL groups in gset so that at least one sibling group (layers 1+)
// always exposes its boundary Add in getInputs(), detects the mask mismatch, and
// correctly sets irregular_io=true regardless of hash order.
TEST(OnlinePartitioningTest, IsRegularParameterCase_PrefillModel_InputsEmbeds) {
    LLMConfig config;
    config.num_layers = 4;
    config.hidden_size = 64;
    config.use_inputs_embeds = true;  // layer 0's residual Add reads inputs_embeds (ov::Parameter)
    config.use_kv_cache = true;

    ModelBuilder mb;
    auto model = mb.build_llm(config);

    // Partitioning requires a static-shape stateless model, matching the real
    // production path in LLMCompiledModel before getPartitioning() is called.
    ov::pass::StatefulToStateless().run_on_model(model);
    // StatefulToStateless removes Assign nodes from model sinks but does NOT sever
    // their incoming edges (Concat → Assign).  clone() rebuilds the model from
    // get_ordered_ops(), which no longer includes the dangling Assigns, producing a
    // clean graph.  This mirrors LLMCompiledModel's kvcache_model->clone() → prefill_model.
    model = model->clone();

    // Resolve dynamic dimensions using the same name-aware logic as the production
    // LLMCompiledModel::reshape_to_static():
    //   - inputs_embeds   -> [1, seq_len, hidden_size]   (hidden_size stays static)
    //   - attention_mask  -> [1, kvcache_size]            (covers full past + current)
    //   - position_ids    -> [1, seq_len]
    //   - past KV tensors -> [1, kv_heads, past_kv_len, head_dim]
    //                        (axes 1 and 3 are already static in the model builder)
    const int64_t seq_len = 8;
    const int64_t past_kv_len = 8;
    const int64_t kvcache_size = seq_len + past_kv_len;
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto& pshape = input.get_partial_shape();
        ov::PartialShape new_shape;
        if (name.find("inputs_embeds") != std::string::npos) {
            new_shape = {1, seq_len, pshape[2]};  // hidden_size is static
        } else if (name.find("attention_mask") != std::string::npos) {
            new_shape = {1, kvcache_size};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shape = {1, seq_len};
        } else {
            // Past KV tensors: [batch, kv_heads, past_kv_len, head_dim]
            // (axes 1 and 3 are already static; set batch=1 and past_kv_len)
            new_shape = pshape;
            new_shape[0] = 1;
            new_shape[2] = past_kv_len;
        }
        new_shapes[name] = new_shape;
    }
    model->reshape(new_shapes);

    auto cfg = createConfigWithKeepBlockSize(4);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(ens.repeated.size(), 1u);  // sanity: transformer layers must form a repeated block
    // Layer 0 reads inputs_embeds (ov::Parameter); layers 1+ read from prior computation.
    // isRegularParameterCase must detect this structural asymmetry and set irregular_io=true.
    EXPECT_TRUE(ens.irregular_io);
}

TEST_P(IsRegularCrossGroupConsumerCaseParametrized, CheckForDifferentCrossGroupConsumerConfigs) {
    auto [head_block_indices, expected_result] = GetParam();

    ModelBuilder mb;
    // Uses the KV-sharing block structure (Add→Relu→Multiply→Relu) where both Relu
    // nodes share an identical metadesc.  When a head block's interior Relu gains an
    // external consumer, output_ometa remains {"Relu"} because the boundary Relu
    // already contributes that metadesc.  All 10 blocks stay in one repeated family,
    // so isRegularCrossGroupConsumerCase is the first check that can detect the
    // per-bank connectivity asymmetry.
    auto model = mb.get_model_with_kv_sharing_repeated_blocks(10, head_block_indices);

    // keepBlockSize=4 matches the 4-op block structure (Add→Relu→Multiply→Relu)
    auto cfg = createConfigWithKeepBlockSize(4);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(ens.repeated.size(), 1u);  // sanity check that we have repeated blocks
    EXPECT_EQ(ens.irregular_io, expected_result);
}

// isRegularCrossGroupConsumerCase checks that within each operation bank of a repeated
// block family, the cross-group (non-Result) consumer pattern is symmetric across all
// instances.  F16IC inserts a Convert on every cross-group edge; if some instances
// have an external consumer for a given output port and others do not, the Convert
// count differs, breaking the operation-bank invariant.
//
// The key model property required to trigger this check:
//   - All blocks must remain in ONE repeated-block family (same MetaInterconnectIO).
//   - Some blocks' interior node must have an extra external consumer.
// get_model_with_kv_sharing_repeated_blocks() satisfies both conditions because the
// interior Relu and the boundary Relu share the same metadesc ("Relu f32{1,1,8}").
// Adding the interior Relu as an extra output_layer leaves output_ometa unchanged.
INSTANTIATE_TEST_SUITE_P(OnlinePartitioningTest,
                         IsRegularCrossGroupConsumerCaseParametrized,
                         ::testing::Values(
                             // All blocks have a cross-group consumer: symmetric, F16IC safe
                             std::make_tuple(std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                             /*irregular_io=*/false),
                             // Some blocks have a cross-group consumer: interior Relu bank is asymmetric
                             // (some mask=[true], others mask=[false]) → isRegularCrossGroupConsumerCase
                             // returns false → irregular_io=true
                             std::make_tuple(std::vector<std::size_t>{2, 5, 8}, /*irregular_io=*/true),
                             // Only one block has a cross-group consumer: same asymmetry detected
                             std::make_tuple(std::vector<std::size_t>{9}, /*irregular_io=*/true),
                             // No blocks have a cross-group consumer: symmetric, F16IC safe
                             std::make_tuple(std::vector<std::size_t>{}, /*irregular_io=*/false)));

// Regression test for isRegularCrossGroupConsumerCase, introduced to handle the
// Gemma4 KV-sharing pattern.
//
// In Gemma4, ALL transformer layers belong to one repeated-block family.  However,
// only the "head" layers additionally forward their interior Multiply_1 output to
// k/v_proj (a different partition group), while non-head layers keep Multiply_1
// purely internal.  Because Multiply_1's metadesc is already present in output_ometa
// (via another boundary node of the same op type), gaining an extra output_layer for
// Multiply_1 leaves output_ometa unchanged, so the family is NOT split by the scanner.
// isRegularCrossGroupConsumerCase then detects the per-bank inconsistency:
//   - non-head interior node bank → mask=[false] (no external consumer)
//   - head interior node bank     → mask=[true]  (external MatMul consumer)
// → returns false → irregular_io=true, disabling F16IC for this model.
//
// get_model_with_kv_sharing_repeated_blocks() reproduces this same condition by using
// a block structure (Add→Relu(interior)→Multiply→Relu(boundary)) where both Relu
// nodes share an identical metadesc.  Block 5's interior Relu gains an external
// MatMul consumer but output_ometa stays {"Relu"}, so all 10 blocks remain in one
// family and the function's return-false path is exercised.
TEST(OnlinePartitioningTest, IsRegularCrossGroupConsumerCase_Gemma4KVSharingPattern) {
    ModelBuilder mb;
    auto model = mb.get_model_with_kv_sharing_repeated_blocks(10, {5});

    auto cfg = createConfigWithKeepBlockSize(4);
    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    EXPECT_GE(ens.repeated.size(), 1u);
    // isRegularCrossGroupConsumerCase detects the interior-Relu bank asymmetry
    // (block 5 has external consumer, others do not) → irregular_io=true.
    EXPECT_TRUE(ens.irregular_io);
}

TEST(OnlinePartitioningTest, SlidingWindowAllLayers_ModelBuilds) {
    auto model = ov::test::npuw::build_sliding_window_test_model();
    ASSERT_NE(model, nullptr);

    bool has_attention_mask = false;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == "attention_mask")
            has_attention_mask = true;
    }
    EXPECT_TRUE(has_attention_mask);
}

TEST(OnlinePartitioningTest, SlidingWindowAlternating_ModelBuilds) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, true);
    ASSERT_NE(model, nullptr);
}

TEST(OnlinePartitioningTest, TokenTypeIds_HasCorrectInputs) {
    auto model = ov::test::npuw::build_token_type_ids_test_model();
    ASSERT_NE(model, nullptr);

    bool has_token_type_ids = false;
    bool has_inputs_embeds = false;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == "token_type_ids")
            has_token_type_ids = true;
        if (param->get_friendly_name() == "inputs_embeds")
            has_inputs_embeds = true;
    }
    EXPECT_TRUE(has_token_type_ids);
    EXPECT_TRUE(has_inputs_embeds);
}

// Verify token_type_ids mask modifier handles seq != total_seq (KV-cache scenario).
// Mirrors production: stateful→stateless, reshape to static shapes, validate+clone.
TEST(OnlinePartitioningTest, TokenTypeIds_StaticReshape_SeqNeTotalSeq) {
    auto model = ov::test::npuw::build_token_type_ids_test_model();
    ASSERT_NE(model, nullptr);

    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    // seq_len=1 (generate), total_seq=17 (past + current) — tests the offset slicing
    const int64_t seq_len = 1;
    const int64_t past_kv_len = 16;
    const int64_t total_seq = seq_len + past_kv_len;
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto& pshape = input.get_partial_shape();
        ov::PartialShape new_shape;
        if (name.find("inputs_embeds") != std::string::npos) {
            new_shape = {1, seq_len, pshape[2]};
        } else if (name.find("attention_mask") != std::string::npos ||
                   name.find("token_type_ids") != std::string::npos) {
            new_shape = {1, total_seq};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shape = {1, seq_len};
        } else if (name.find("beam_idx") != std::string::npos) {
            new_shape = {1};
        } else {
            new_shape = pshape;
            new_shape[0] = 1;
            new_shape[2] = past_kv_len;
        }
        new_shapes[name] = new_shape;
    }

    // Must not throw — validates that the token_type_ids mask modifier
    // correctly handles the offset slice when seq != total_seq.
    EXPECT_NO_THROW(model->reshape(new_shapes));
}
