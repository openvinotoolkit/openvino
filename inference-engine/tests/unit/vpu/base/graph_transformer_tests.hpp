// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include <gtest/gtest.h>

#include <vpu/compile_env.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/model/model.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/middleend/pass_manager.hpp>
#include <vpu/backend/backend.hpp>
#include <vpu/utils/ie_helpers.hpp>

#include <unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp>

namespace vpu {

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
    PrevStageOutput,
    Intermediate,
    Constant
};

struct InputInfo final {
    InputType type = InputType::Original;
    int originalInputInd = -1;
    int prevStageInd = -1;
    int prevStageOutputInd = -1;
    DataDesc desc = DataDesc();

    static InputInfo fromNetwork(int ind = 0);

    static InputInfo fromPrevStage(int ind, int outputInd = 0);
    static InputInfo constant(const DataDesc& desc);

    InputInfo& output(int ind);
};

enum class OutputType {
    Original,
    Intermediate
};

struct OutputInfo final {
    OutputType type = OutputType::Original;
    int originalOutputInd = -1;
    DataDesc desc = DataDesc();
    MemoryType memReq = MemoryType::DDR;

    static OutputInfo fromNetwork(int ind = 0);

    static OutputInfo intermediate(const DataDesc& desc = DataDesc());
    static OutputInfo intermediate(MemoryType memReq = MemoryType::DDR);
};

class TestModel final {
public:
    TestModel() = default;
    TestModel(const Model& model);

    const Model& getBaseModel() const;
    const DataVector& getInputs() const;
    const DataVector& getOutputs() const;
    const StageVector& getStages() const;

    void createInputs(std::vector<DataDesc> inputDescs = {});
    void createOutputs(std::vector<DataDesc> outputDescs = {});

    Stage addStage(const std::vector<InputInfo>& curInputInfos, const std::vector<OutputInfo>& curOutputInfos);

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

    DataVector _inputs;
    DataVector _outputs;
    StageVector _stages;
};

template <class StageRange>
void checkStageTestInds(const StageRange& stageRange, std::initializer_list<int> expectedInds);

template <class StageRange>
void checkStageTestInds(const StageRange& stageRange, std::initializer_list<int> expectedInds, const std::function<void(const Stage&)>& extraCheck);

bool checkExecutionOrder(const Model& model, const std::vector<int>& execOrder);

class GraphTransformerTest : public ::testing::Test {
public:
    Platform platform = Platform::MYRIAD_X;
    CompilationConfig config;

    StageBuilder::Ptr stageBuilder;
    FrontEnd::Ptr frontEnd;
    PassManager::Ptr passManager;
    BackEnd::Ptr backEnd;

    bool compileEnvInitialized = false;

    void SetUp() override;
    void TearDown() override;

    void InitCompileEnv();

    Model CreateModel();

    TestModel CreateTestModel();

private:
    MockICore  _mockCore;
    Logger::Ptr _log;
    std::list<ModelPtr> _models;
};

} // namespace vpu
