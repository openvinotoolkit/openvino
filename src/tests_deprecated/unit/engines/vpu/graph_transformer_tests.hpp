// Copyright (C) 2018-2022 Intel Corporation
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
#include <vpu/stage_builder.hpp>
#include <vpu/middleend/pass_manager.hpp>
#include <vpu/backend/backend.hpp>

#include <unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp>

namespace vpu {

template <class Cont, class Cond>
bool contains(const Cont& cont, const Cond& cond) {
    for (const auto& val : cont) {
        if (cond(val)) {
            return true;
        }
    }
    return false;
}

template <typename Value>
using InOutPortMap = std::unordered_map<int, Value>;

class TestStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override;

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override;

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override;

    void finalizeDataLayoutImpl() override;

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override;

    void serializeParamsImpl(BlobSerializer&) const override;

    void serializeDataImpl(BlobSerializer&) const override;
};

enum class InputType {
    Original,
    PrevStageOutput
};

struct InputInfo final {
    InputType type = InputType::Original;
    int originalInputInd = -1;
    int prevStageInd = -1;
    int prevStageOutputInd = -1;

    static InputInfo fromNetwork(int ind = 0) {
        InputInfo info;
        info.type = InputType::Original;
        info.originalInputInd = ind;
        return info;
    }

    static InputInfo fromPrevStage(int ind) {
        InputInfo info;
        info.type = InputType::PrevStageOutput;
        info.prevStageInd = ind;
        info.prevStageOutputInd = 0;
        return info;
    }

    InputInfo& output(int ind) {
        assert(type == InputType::PrevStageOutput);
        prevStageOutputInd = ind;
        return *this;
    }
};

enum class OutputType {
    Original,
    Intermediate
};

struct OutputInfo final {
    OutputType type = OutputType::Original;
    int originalOutputInd = -1;

    static OutputInfo fromNetwork(int ind = 0) {
        OutputInfo info;
        info.type = OutputType::Original;
        info.originalOutputInd = ind;
        return info;
    }

    static OutputInfo intermediate() {
        OutputInfo info;
        info.type = OutputType::Intermediate;
        return info;
    }
};

class TestModel final {
public:
    TestModel() = default;
    TestModel(const Model& model, const DataDesc& dataDesc);

    const Model& getBaseModel() const { return _model; }
    const DataVector& getInputs() const { return _inputs; }
    const DataVector& getOutputs() const { return _outputs; }
    const StageVector& getStages() const { return _stages; }

    void createInputs(int numInputs);
    void createOutputs(int numOutputs);

    int addStage(
            std::initializer_list<InputInfo> curInputInfos,
            std::initializer_list<OutputInfo> curOutputInfos);

    void setStageDataOrderInfo(
            int stageInd,
            const InOutPortMap<DimsOrder>& inputInfo,
            const InOutPortMap<DimsOrder>& outputInfo);
    void setStageStridesInfo(
            int stageInd,
            const InOutPortMap<StridesRequirement>& inputInfo,
            const InOutPortMap<StridesRequirement>& outputInfo);
    void setStageBatchInfo(
            int stageInd,
            const InOutPortMap<BatchSupport>& inputInfo);

private:
    Model _model;
    DataDesc _dataDesc;

    DataVector _inputs;
    DataVector _outputs;
    StageVector _stages;
};

template <class StageRange>
void CheckStageTestInds(const StageRange& stageRange, std::initializer_list<int> expectedInds) {
    auto stageVector = toVector(stageRange);

    ASSERT_EQ(expectedInds.size(), stageVector.size());

    size_t stageInd = 0;
    for (auto expectedInd : expectedInds) {
        ASSERT_EQ(expectedInd, stageVector[stageInd]->attrs().template get<int>("test_ind"));
        ++stageInd;
    }
}

template <class StageRange>
void CheckStageTestInds(const StageRange& stageRange, std::initializer_list<int> expectedInds, const std::function<void(const Stage&)>& extraCheck) {
    auto stageVector = toVector(stageRange);

    ASSERT_EQ(expectedInds.size(), stageVector.size());

    size_t stageInd = 0;
    for (auto expectedInd : expectedInds) {
        ASSERT_EQ(expectedInd, stageVector[stageInd]->attrs().template get<int>("test_ind"));
        ++stageInd;

        ASSERT_NO_FATAL_FAILURE(extraCheck(stageVector[stageInd]));
    }
}

PluginConfiguration createConfiguration();

class GraphTransformerTest : public TestsCommon {
public:
    PluginConfiguration config;

    StageBuilder::Ptr stageBuilder;
    FrontEnd::Ptr frontEnd;
    PassManager::Ptr passManager;
    BackEnd::Ptr backEnd;

    bool compileEnvInitialized = false;

    void SetUp() override;
    void TearDown() override;

    void InitCompileEnv();

    Model CreateModel();

    TestModel CreateTestModel(const DataDesc& dataDesc);

private:
    std::shared_ptr<MockICore> _mockCore = std::make_shared<MockICore>();
    Logger::Ptr _log;
    std::list<ModelPtr> _models;
};

}
