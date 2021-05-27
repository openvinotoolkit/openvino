// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#include "ngraph/opsets/opset7.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/memory.hpp"

namespace LayerTestsDefinitions {

    std::string MemoryTest::getTestCaseName(const testing::TestParamInfo<MemoryTestParams> &obj) {
        int64_t iteration_count;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape;
        std::string targetDevice;
        std::tie(iteration_count, inputShape, netPrecision, targetDevice) = obj.param;

        std::ostringstream result;
        result << "iteration_count=" << iteration_count << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        result << ")";
        return result.str();
    }

    void MemoryTest::SetUp() {
        using namespace ngraph;
        InferenceEngine::SizeVector inputShape;
        std::tie(iteration_count, inputShape, netPrecision, targetDevice) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto param = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto variable = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
        auto read_value = std::make_shared<opset7::ReadValue>(param.at(0), variable);
        auto add = std::make_shared<opset7::Add>(read_value, param.at(0));
        auto assign = std::make_shared<opset7::Assign>(add, variable);
        auto res = std::make_shared<opset7::Result>(add);
        function = std::make_shared<Function>(ResultVector{res}, SinkVector{assign}, param, "TestMemory");

        auto hostTensor = std::make_shared<ngraph::HostTensor>(ngPrc, inputShape);
        auto variable_context = std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
        auto variable_value = std::make_shared<VariableValue>(hostTensor);
        variable_context->get().set_variable_value(function->get_variable_by_id("v0"), variable_value);
        eval_context["VariableContext"] = variable_context;
    }


    void MemoryTest::Run() {
        using namespace LayerTestsUtils;
        auto crashHandler = [](int errCode) {
            auto &s = Summary::getInstance();
            s.saveReport();
            std::cout << "Unexpected application crash!" << std::endl;
            std::abort();
        };
        signal(SIGSEGV, crashHandler);

        auto &s = LayerTestsUtils::Summary::getInstance();
        s.setDeviceName(targetDevice);

        if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
            s.updateOPsStats(function, PassRate::Statuses::SKIPPED);
            GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
        } else {
            s.updateOPsStats(function, PassRate::Statuses::CRASHED);
        }

        try {
            LoadNetwork();
            GenerateInputs();
            for (int64_t i = 0; i < iteration_count; ++i) {
                Infer();
                Validate();
            }
            s.updateOPsStats(function, PassRate::Statuses::PASSED);
        }
        catch (const std::runtime_error &re) {
            s.updateOPsStats(function, PassRate::Statuses::FAILED);
            GTEST_FATAL_FAILURE_(re.what());
        } catch (const std::exception &ex) {
            s.updateOPsStats(function, PassRate::Statuses::FAILED);
            GTEST_FATAL_FAILURE_(ex.what());
        } catch (...) {
            s.updateOPsStats(function, PassRate::Statuses::FAILED);
            GTEST_FATAL_FAILURE_("Unknown failure occurred.");
        }
    }

    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> MemoryTest::CalculateRefs() {
        using namespace ngraph;
        function->validate_nodes_and_infer_types();

        auto referenceInputs = std::vector<std::vector<uint8_t>>(inputs.size());
        auto refInputsTypes = std::vector<ngraph::element::Type>(inputs.size());
        HostTensorVector inputTensors;
        for (auto & input : inputs) {
            const auto &dataSize = input->byteSize();
            const auto &tensorDesc = input->getTensorDesc();

            auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
            IE_ASSERT(memory);
            const auto lockedMemory = memory->wmap();
            const auto buffer = lockedMemory.as<const std::uint8_t *>();

            auto hostTensor = std::make_shared<ngraph::HostTensor>(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(tensorDesc.getPrecision()),
                                                                   tensorDesc.getDims());
            hostTensor->write(buffer, dataSize);
            inputTensors.push_back(hostTensor);
        }

        const auto &outInfo = executableNetwork.GetOutputsInfo();
        HostTensorVector outputTensors(outInfo.size(), std::make_shared<ngraph::HostTensor>());
        function->evaluate(outputTensors, inputTensors, eval_context);

        std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> outputs(outInfo.size());
        for (size_t idx = 0; idx < outInfo.size(); ++idx) {
            outputs[idx].first = outputTensors[idx]->get_element_type();
            outputs[idx].second.resize(outputTensors[idx]->get_size_in_bytes());
            outputTensors[idx]->read(outputs[idx].second.data(), outputTensors[idx]->get_size_in_bytes());
        }
        return outputs;
    }

}  // namespace LayerTestsDefinitions

