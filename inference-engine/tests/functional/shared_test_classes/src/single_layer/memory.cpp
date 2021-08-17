// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#include <ie_transformations.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/serialize.hpp>
#include <functional_test_utils/core_config.hpp>
#include "ngraph/opsets/opset7.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph/pass/low_latency.hpp"
#include "shared_test_classes/single_layer/memory.hpp"

using namespace ngraph;
using namespace opset7;

namespace LayerTestsDefinitions {

    std::string MemoryTest::getTestCaseName(const testing::TestParamInfo<MemoryTestParams> &obj) {
        int64_t iteration_count;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape;
        std::string targetDevice;
        ngraph::helpers::MemoryTransformation transformation;
        std::tie(transformation, iteration_count, inputShape, netPrecision, targetDevice) = obj.param;

        std::ostringstream result;
        result << "transformation=" << transformation << "_";
        result << "iteration_count=" << iteration_count << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        result << ")";
        return result.str();
    }

    void MemoryTest::SetUp() {
        std::tie(transformation, iteration_count, inputShape, netPrecision, targetDevice) = this->GetParam();
        ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        if (transformation == ngraph::helpers::MemoryTransformation::NONE) {
            CreateCommonFunc();
        } else {
            CreateTIFunc();
            ApplyLowLatency();
        }

        auto hostTensor = std::make_shared<HostTensor>(ngPrc, inputShape);
        auto variable_context = std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
        auto variable_value = std::make_shared<VariableValue>(hostTensor);
        variable_context->get().set_variable_value(function->get_variable_by_id("v0"), variable_value);
        eval_context["VariableContext"] = variable_context;
    }


    void MemoryTest::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
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
            if (transformation != ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API) {
                LoadNetwork();
            } else {
                CoreConfiguration(this);
                ConfigureNetwork();
                executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
            }
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

    std::vector<std::pair<element::Type, std::vector<std::uint8_t>>> MemoryTest::CalculateRefs() {
        using namespace ngraph;
        function->validate_nodes_and_infer_types();

        auto referenceInputs = std::vector<std::vector<uint8_t>>(inputs.size());
        auto refInputsTypes = std::vector<element::Type>(inputs.size());
        HostTensorVector inputTensors;
        for (auto & input : inputs) {
            const auto &dataSize = input->byteSize();
            const auto &tensorDesc = input->getTensorDesc();

            auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
            IE_ASSERT(memory);
            const auto lockedMemory = memory->wmap();
            const auto buffer = lockedMemory.as<const std::uint8_t *>();

            auto hostTensor = std::make_shared<HostTensor>(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(tensorDesc.getPrecision()),
                                                                   tensorDesc.getDims());
            hostTensor->write(buffer, dataSize);
            inputTensors.push_back(hostTensor);
        }

        // evaluate method is not implemented for TI op.
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::UnrollTensorIterator>();
        manager.run_passes(function);

        const auto &outInfo = executableNetwork.GetOutputsInfo();
        HostTensorVector outputTensors(outInfo.size());
        for (auto& outTensor : outputTensors) {
            outTensor = std::make_shared<HostTensor>();
        }
        function->evaluate(outputTensors, inputTensors, eval_context);

        std::vector<std::pair<element::Type, std::vector<std::uint8_t>>> outputs(outInfo.size());
        for (size_t idx = 0; idx < outInfo.size(); ++idx) {
            outputs[idx].first = outputTensors[idx]->get_element_type();
            outputs[idx].second.resize(outputTensors[idx]->get_size_in_bytes());
            outputTensors[idx]->read(outputs[idx].second.data(), outputTensors[idx]->get_size_in_bytes());
        }
        return outputs;
    }

    void MemoryTest::CreateTIFunc() {
        auto param = builder::makeParams(ngPrc, {inputShape}).at(0);
        std::vector<std::vector<size_t>> shape = {{static_cast<size_t>(iteration_count), 1}};
        auto iter_count = builder::makeParams(ngPrc, shape).at(0);

        // Body
        auto X = builder::makeParams(ngPrc, {inputShape}).at(0);
        auto Y = builder::makeParams(ngPrc, {inputShape}).at(0);
        auto Iter = builder::makeParams(ngPrc, {Shape{1, 1}}).at(0);
        auto add = std::make_shared<Add>(X, Y);
        auto res = std::make_shared<Result>(add);
        auto Iter_res = std::make_shared<Result>(Iter);
        auto body = std::make_shared<Function>(OutputVector{res, Iter_res}, ParameterVector {X, Y, Iter});

        // TI construction
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_merged_input(X, param, res);
        tensor_iterator->set_invariant_input(Y, param);
        tensor_iterator->set_sliced_input(Iter, iter_count, 0, 1, 1, -1, 0);

        auto output = tensor_iterator->get_iter_value(res, -1);
        auto output_iter = tensor_iterator->get_concatenated_slices(Iter_res, 0, 1, 1, -1, 0);
        function = std::make_shared<Function>(OutputVector{output, output_iter},
                                              ParameterVector{param, iter_count},
                                              "PureTI");
    }

    void MemoryTest::CreateCommonFunc() {
        auto param = builder::makeParams(ngPrc, {inputShape});
        auto variable = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
        auto read_value = std::make_shared<ReadValue>(param.at(0), variable);
        auto add = std::make_shared<Add>(read_value, param.at(0));
        auto assign = std::make_shared<Assign>(add, variable);
        auto res = std::make_shared<Result>(add);
        function = std::make_shared<Function>(ResultVector{res}, SinkVector{assign}, param, "TestMemory");
    }

    void MemoryTest::ApplyLowLatency() {
       if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2) {
           function->validate_nodes_and_infer_types();
           pass::Manager manager;
           manager.register_pass<pass::LowLatency2>();
           manager.run_passes(function);
       } else if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT) {
           function->validate_nodes_and_infer_types();
           pass::Manager manager;
           manager.register_pass<pass::LowLatency2>(false);
           manager.run_passes(function);
        } else if (transformation == ngraph::helpers::MemoryTransformation::LOW_LATENCY_V2_REGULAR_API) {
            cnnNetwork = InferenceEngine::CNNNetwork{function};
           InferenceEngine::lowLatency2(cnnNetwork, iteration_count);
        }
    }

}  // namespace LayerTestsDefinitions

