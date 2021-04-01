// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <atomic>
#include <iomanip>

#include <vpu/utils/io.hpp>

namespace vpu {

StagePtr TestStage::cloneImpl() const {
    return std::make_shared<TestStage>(*this);
}

namespace {

template <typename Value>
void setInOutPortInfo(
        const Stage& stage,
        const std::string& attrBaseName,
        StageDataInfo<Value>& info) {
    auto inAttrName = formatString("test_input_%s_info", attrBaseName);
    auto outAttrName = formatString("test_output_%s_info", attrBaseName);

    if (stage->attrs().has(inAttrName)) {
        const auto& inputInfo = stage->attrs().get<InOutPortMap<Value>>(inAttrName);

        for (const auto& p : inputInfo) {
            info.setInput(stage->inputEdge(p.first), p.second);
        }
    }

    if (stage->attrs().has(outAttrName)) {
        const auto& outputInfo = stage->attrs().get<InOutPortMap<Value>>(outAttrName);

        for (const auto& p : outputInfo) {
            info.setOutput(stage->outputEdge(p.first), p.second);
        }
    }
}

}

void TestStage::propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) {
    setInOutPortInfo(this, "DataOrder", orderInfo);
}

void TestStage::getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) {
    setInOutPortInfo(this, "Strides", stridesInfo);
}

void TestStage::finalizeDataLayoutImpl() {
}

void TestStage::getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) {
    setInOutPortInfo(this, "Batch", batchInfo);

    if (attrs().has("test_input_Batch_info")) {
        for (const auto& outEdge : outputEdges()) {
            batchInfo.setOutput(outEdge, BatchSupport::Split);
        }
    }
}

void TestStage::serializeParamsImpl(BlobSerializer&) const {
}

void TestStage::serializeDataImpl(BlobSerializer&) const {
}

TestModel::TestModel(const Model& model, const DataDesc& dataDesc) :
        _model(model), _dataDesc(dataDesc) {
}

void TestModel::createInputs(int numInputs) {
    _model->attrs().set<int>("numInputs", numInputs);
    _inputs.resize(numInputs);

    for (int i = 0; i < numInputs; ++i) {
        _inputs[i] = _model->addInputData(formatString("Input %d", i), _dataDesc);
    }
}

void TestModel::createOutputs(int numOutputs) {
    _model->attrs().set<int>("numOutputs", numOutputs);
    _outputs.resize(numOutputs);

    for (int i = 0; i < numOutputs; ++i) {
        _outputs[i] = _model->addOutputData(formatString("Output %d", i), _dataDesc);
    }
}

int TestModel::addStage(
        std::initializer_list<InputInfo> curInputInfos,
        std::initializer_list<OutputInfo> curOutputInfos) {
    DataVector curInputs;
    for (const auto& info : curInputInfos) {
        if (info.type == InputType::Original) {
            curInputs.push_back(_inputs.at(info.originalInputInd));
        } else {
            curInputs.push_back(_stages.at(info.prevStageInd)->output(info.prevStageOutputInd));
        }
    }

    DataVector curOutputs;
    for (const auto& info : curOutputInfos) {
        if (info.type == OutputType::Original) {
            curOutputs.push_back(_outputs.at(info.originalOutputInd));
        } else {
            curOutputs.push_back(_model->addNewData(formatString("Data %d / %d", _stages.size(), curOutputs.size()), _dataDesc));
        }
    }

    auto stage = _model->addNewStage<TestStage>(
        formatString("Stage %m%m%d", std::setw(2), std::setfill('0'), _stages.size()),
        StageType::None,
        nullptr,
        curInputs,
        curOutputs);
    stage->attrs().set<int>("test_ind", _stages.size());

    _stages.push_back(stage);

    return _stages.size() - 1;
}
void TestModel::setStageDataOrderInfo(
        int stageInd,
        const InOutPortMap<DimsOrder>& inputInfo,
        const InOutPortMap<DimsOrder>& outputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_DataOrder_info", inputInfo);
    }
    if (!outputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_DataOrder_info", outputInfo);
    }
}

void TestModel::setStageStridesInfo(
        int stageInd,
        const InOutPortMap<StridesRequirement>& inputInfo,
        const InOutPortMap<StridesRequirement>& outputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Strides_info", inputInfo);
    }
    if (!outputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Strides_info", outputInfo);
    }
}

void TestModel::setStageBatchInfo(
        int stageInd,
        const InOutPortMap<BatchSupport>& inputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Batch_info", inputInfo);
    }
}

void GraphTransformerTest::SetUp() {
    ASSERT_NO_FATAL_FAILURE(TestsCommon::SetUp());

    _log = std::make_shared<Logger>(
        "Test",
        LogLevel::Debug,
        consoleOutput());

    stageBuilder = std::make_shared<StageBuilder>();
    frontEnd = std::make_shared<FrontEnd>(stageBuilder, &_mockCore);
    backEnd = std::make_shared<BackEnd>();
    passManager = std::make_shared<PassManager>(stageBuilder, backEnd);
}

void GraphTransformerTest::TearDown() {
    for (const auto& model : _models) {
        backEnd->dumpModel(model);
    }

    if (compileEnvInitialized) {
        CompileEnv::free();
    }

    TestsCommon::TearDown();
}

void GraphTransformerTest::InitCompileEnv() {
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_FILE_NAME")) {
        config.dumpInternalGraphFileName = envVar;
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_DIRECTORY")) {
        config.dumpInternalGraphDirectory = envVar;
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_ALL_PASSES")) {
        config.dumpAllPasses = std::stoi(envVar) != 0;
    }

    CompileEnv::init(platform, config, _log);
    compileEnvInitialized = true;
}

namespace {

std::atomic<int> g_counter(0);

}

Model GraphTransformerTest::CreateModel() {
    const auto& env = CompileEnv::get();

    auto unitTest = testing::UnitTest::GetInstance();
    IE_ASSERT(unitTest != nullptr);
    auto curTestInfo = unitTest->current_test_info();
    IE_ASSERT(curTestInfo != nullptr);

    auto model = std::make_shared<ModelObj>(
        formatString("%s/%s", curTestInfo->test_case_name(), curTestInfo->name()));
    model->attrs().set<int>("index", g_counter.fetch_add(1));
    model->attrs().set<Resources>("resources", env.resources);

    _models.push_back(model);

    return model;
}

TestModel GraphTransformerTest::CreateTestModel(const DataDesc& dataDesc) {
    return TestModel(CreateModel(), dataDesc);
}

}
