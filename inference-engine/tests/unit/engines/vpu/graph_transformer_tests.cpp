// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <atomic>

#include <vpu/utils/io.hpp>

void VPU_GraphTransformerTest::SetUp() {
    ASSERT_NO_FATAL_FAILURE(TestsCommon::SetUp());

    _log = std::make_shared<vpu::Logger>(
        "Test",
        vpu::LogLevel::Debug,
        vpu::consoleOutput());

    stageBuilder = std::make_shared<vpu::StageBuilder>();
    frontEnd = std::make_shared<vpu::FrontEnd>(stageBuilder);
    backEnd = std::make_shared<vpu::BackEnd>();
    passManager = std::make_shared<vpu::PassManager>(stageBuilder, backEnd);
}

void VPU_GraphTransformerTest::TearDown() {
    for (const auto& model : _models) {
        backEnd->dumpModel(model);
    }

    vpu::CompileEnv::free();

    TestsCommon::TearDown();
}

void VPU_GraphTransformerTest::InitCompileEnv() {
    vpu::CompileEnv::init(platform, config, _log);
}

namespace {

std::atomic<int> g_counter(0);

}

vpu::Model::Ptr VPU_GraphTransformerTest::CreateModel() {
    const auto& env = vpu::CompileEnv::get();

    auto unitTest = testing::UnitTest::GetInstance();
    IE_ASSERT(unitTest != nullptr);
    auto curTestInfo = unitTest->current_test_info();
    IE_ASSERT(curTestInfo != nullptr);

    auto model = std::make_shared<vpu::Model>(
        vpu::formatString("%s/%s", curTestInfo->test_case_name(), curTestInfo->name()));
    model->attrs().set<int>("index", g_counter.fetch_add(1));
    model->attrs().set<vpu::Resources>("resources", env.resources);

    _models.push_back(model);

    return model;
}
