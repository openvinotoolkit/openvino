// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include <gtest/gtest.h>
#include <tests_common.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/model/model.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/frontend/stage_builder.hpp>
#include <vpu/pass_manager.hpp>
#include <vpu/backend/backend.hpp>

class VPU_GraphTransformerTest : public TestsCommon {
public:
    vpu::Platform platform = vpu::Platform::MYRIAD_X;
    vpu::CompilationConfig config;

    vpu::StageBuilder::Ptr stageBuilder;
    vpu::FrontEnd::Ptr frontEnd;
    vpu::PassManager::Ptr passManager;
    vpu::BackEnd::Ptr backEnd;

    void SetUp() override;
    void TearDown() override;

    void InitCompileEnv();

    vpu::Model::Ptr CreateModel();

private:
    vpu::Logger::Ptr _log;
    std::list<vpu::Model::Ptr> _models;
};

template <class Cont, class Cond>
bool contains(const Cont& cont, const Cond& cond) {
    for (const auto& val : cont) {
        if (cond(val)) {
            return true;
        }
    }
    return false;
}
