// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/openvino.hpp"
#include "partitioning/online/compiler.hpp"
#include "partitioning/online/group.hpp"
#include "partitioning/online/snapshot.hpp"
#include "model_generator/model_generator.hpp"

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

TEST(OnlinePartitioningTest, Partitioning_IsTheSame_SmallModel) {
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    auto cfg = ::intel_npu::Config(opt_desc);
    ::intel_npu::registerNPUWOptions(*opt_desc);
    std::map<std::string, std::string> cfg_map = {{"NPUW_ONLINE_KEEP_BLOCK_SIZE", "9"}};
    cfg.update(cfg_map);

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    for (size_t i = 0; i < 100; ++i) {
        auto ens_again = ov::npuw::online::buildPartitioning(model, cfg);
        EXPECT_TRUE(isEqualEns(ens, ens_again));
    }
}

TEST(OnlinePartitioningTest, Partitioning_IsTheSame_RepeatedModel) {
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    auto cfg = ::intel_npu::Config(opt_desc);
    ::intel_npu::registerNPUWOptions(*opt_desc);
    std::map<std::string, std::string> cfg_map = {{"NPUW_ONLINE_KEEP_BLOCK_SIZE", "9"}};
    cfg.update(cfg_map);

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    for (size_t i = 0; i < 100; ++i) {
        auto ens_again = ov::npuw::online::buildPartitioning(model, cfg);
        EXPECT_TRUE(isEqualEns(ens, ens_again));
    }
}

TEST(OnlinePartitioningTest, Partitioning_SingleGroup_SmallModel) {
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->singleGroup();
    EXPECT_EQ(snap->graphSize(), 1);
}

TEST(OnlinePartitioningTest, Partitioning_SingleGroup_RepeatedModel) {
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->singleGroup();
    EXPECT_EQ(snap->graphSize(), 1);
}

TEST(OnlinePartitioningTest, Partitioning_buildGraph_SmallModel) {
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

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
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

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
