// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <vpu/utils/io.hpp>

#include <atomic>
#include <iomanip>

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

} // namespace

InputInfo InputInfo::fromNetwork(int ind) {
    InputInfo info;
    info.type = InputType::Original;
    info.originalInputInd = ind;
    return info;
}

InputInfo InputInfo::fromPrevStage(int ind) {
    InputInfo info;
    info.type = InputType::PrevStageOutput;
    info.prevStageInd = ind;
    info.prevStageOutputInd = 0;
    return info;
}

InputInfo& InputInfo::output(int ind) {
    assert(type == InputType::PrevStageOutput);
    prevStageOutputInd = ind;
    return *this;
}

OutputInfo OutputInfo::fromNetwork(int ind) {
    OutputInfo info;
    info.type = OutputType::Original;
    info.originalOutputInd = ind;
    return info;
}

OutputInfo OutputInfo::intermediate(const DataDesc& desc) {
    OutputInfo info;
    info.type = OutputType::Intermediate;
    info.desc = desc;
    return info;
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

TestModel::TestModel(const Model& model) : _model(model) {}

const Model& TestModel::getBaseModel() const {
    return _model;
}

const DataVector& TestModel::getInputs() const {
    return _inputs;
}

const DataVector& TestModel::getOutputs() const {
    return _outputs;
}

const StageVector& TestModel::getStages() const {
    return _stages;
}

void TestModel::createInputs(std::vector<DataDesc> inputDescs) {
    const auto numInputs = inputDescs.size();

    _model->attrs().set<int>("numInputs", numInputs);
    _inputs.resize(numInputs);

    for (int i = 0; i < numInputs; ++i) {
        _inputs[i] = _model->addInputData(formatString("Input %d", i), inputDescs[i]);
    }
}

void TestModel::createOutputs(std::vector<DataDesc> outputDescs) {
    const auto numOutputs = outputDescs.size();

    _model->attrs().set<int>("numOutputs", numOutputs);
    _outputs.resize(numOutputs);

    for (int i = 0; i < numOutputs; ++i) {
        _outputs[i] = _model->addOutputData(formatString("Output %d", i), outputDescs[i]);
    }
}

Stage TestModel::addStage(
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
            curOutputs.push_back(_model->addNewData(formatString("Data %d / %d", _stages.size(), curOutputs.size()), info.desc));
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

    return stage;
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

template <class StageRange>
void checkStageTestInds(const StageRange& stageRange, std::initializer_list<int> expectedInds) {
    auto stageVector = toVector(stageRange);

    ASSERT_EQ(expectedInds.size(), stageVector.size());

    size_t stageInd = 0;
    for (auto expectedInd : expectedInds) {
        ASSERT_EQ(expectedInd, stageVector[stageInd]->attrs().template get<int>("test_ind"));
        ++stageInd;
    }
}

template <class StageRange>
void checkStageTestInds(const StageRange& stageRange, std::initializer_list<int> expectedInds, const std::function<void(const Stage&)>& extraCheck) {
    auto stageVector = toVector(stageRange);

    ASSERT_EQ(expectedInds.size(), stageVector.size());

    size_t stageInd = 0;
    for (auto expectedInd : expectedInds) {
        ASSERT_EQ(expectedInd, stageVector[stageInd]->attrs().template get<int>("test_ind"));
        ++stageInd;

        ASSERT_NO_FATAL_FAILURE(extraCheck(stageVector[stageInd]));
    }
}

bool checkExecutionOrder(const Model& model, const std::vector<int>& execOrder) {
    auto it = execOrder.begin();

    for (const auto& stage : model->getStages()) {
        if (it == execOrder.end()) {
            return true;
        }

        if (stage->id() == *it) {
            ++it;
        }
    }

    return it == execOrder.end();
}

void GraphTransformerTest::SetUp() {
    _log = std::make_shared<Logger>(
            "Test",
            LogLevel::Debug,
            consoleOutput());

    stageBuilder = std::make_shared<StageBuilder>();
    frontEnd = std::make_shared<FrontEnd>(stageBuilder);
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

TestModel GraphTransformerTest::CreateTestModel() {
    return TestModel(CreateModel());
}

} // namespace vpu
