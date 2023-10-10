// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/memory.hpp"

#include <signal.h>

#include <functional_test_utils/core_config.hpp>
#include <ie_transformations.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>

#include "ngraph/opsets/opset7.hpp"
#include "ngraph/pass/low_latency.hpp"
#include "openvino/op/util/variable_context.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace opset7;

namespace LayerTestsDefinitions {

std::string MemoryTest::getTestCaseName(const testing::TestParamInfo<MemoryTestParams>& obj) {
    int64_t iteration_count;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    ngraph::helpers::MemoryTransformation transformation;
    std::tie(transformation, iteration_count, inputShape, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "transformation=" << transformation << "_";
    result << "iteration_count=" << iteration_count << "_";
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
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
    auto variable_context = ov::op::util::VariableContext();
    auto variable_value = std::make_shared<VariableValue>(hostTensor);
    variable_context.set_variable_value(function->get_variable_by_id("v0"), variable_value);
    eval_context["VariableContext"] = variable_context;
}

void MemoryTest::Run() {
    functionRefs = ngraph::clone_function(*function);
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    using namespace LayerTestsUtils;
    auto crashHandler = [](int errCode) {
        auto& s = ov::test::utils::OpSummary::getInstance();
        s.saveReport();
        std::cout << "Unexpected application crash!" << std::endl;
        std::abort();
    };
    signal(SIGSEGV, crashHandler);

    auto& s = ov::test::utils::OpSummary::getInstance();
    s.setDeviceName(targetDevice);
    if (ov::test::utils::current_test_is_disabled()) {
        s.updateOPsStats(function, ov::test::utils::PassRate::Statuses::SKIPPED);
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
    } else {
        s.updateOPsStats(function, ov::test::utils::PassRate::Statuses::CRASHED);
    }

    try {
        CoreConfiguration(this);
        ConfigureNetwork();
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
        inferRequest = executableNetwork.CreateInferRequest();
        GenerateInputs();
        for (int64_t i = 0; i < iteration_count; ++i) {
            Infer();
            Validate();
        }
        s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::PASSED);
    } catch (const std::runtime_error& re) {
        s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_(re.what());
    } catch (const std::exception& ex) {
        s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_(ex.what());
    } catch (...) {
        s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_("Unknown failure occurred.");
    }
}

void MemoryTest::Infer() {
    ConfigureInferRequest();
    inferRequest.Infer();
}

std::vector<std::pair<element::Type, std::vector<std::uint8_t>>> MemoryTest::CalculateRefs() {
    using namespace ngraph;
    function->validate_nodes_and_infer_types();

    auto referenceInputs = std::vector<std::vector<uint8_t>>(inputs.size());
    auto refInputsTypes = std::vector<element::Type>(inputs.size());
    HostTensorVector inputTensors;
    for (auto& input : inputs) {
        const auto& dataSize = input->byteSize();
        const auto& tensorDesc = input->getTensorDesc();

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto buffer = lockedMemory.as<const std::uint8_t*>();

        auto hostTensor =
            std::make_shared<HostTensor>(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(tensorDesc.getPrecision()),
                                         tensorDesc.getDims());
        hostTensor->write(buffer, dataSize);
        inputTensors.push_back(hostTensor);
    }

    // evaluate method is not implemented for TI op.
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::UnrollTensorIterator>();
    manager.run_passes(function);

    const auto& outInfo = executableNetwork.GetOutputsInfo();
    HostTensorVector outputTensors(outInfo.size());
    for (auto& outTensor : outputTensors) {
        outTensor = std::make_shared<HostTensor>();
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    function->evaluate(outputTensors, inputTensors, eval_context);
    OPENVINO_SUPPRESS_DEPRECATED_END

    std::vector<std::pair<element::Type, std::vector<std::uint8_t>>> outputs(outInfo.size());
    for (size_t idx = 0; idx < outInfo.size(); ++idx) {
        outputs[idx].first = outputTensors[idx]->get_element_type();
        outputs[idx].second.resize(outputTensors[idx]->get_size_in_bytes());
        outputTensors[idx]->read(outputs[idx].second.data(), outputTensors[idx]->get_size_in_bytes());
    }
    return outputs;
}

void MemoryTest::CreateTIFunc() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape));
    std::vector<std::vector<size_t>> shape = {{static_cast<size_t>(iteration_count), 1}};
    auto iter_count =
        std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{static_cast<size_t>(iteration_count), 1});

    // Body
    auto X = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape));
    auto Y = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape));
    auto Iter = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 1});
    auto add = std::make_shared<Add>(X, Y);
    auto res = std::make_shared<Result>(add);
    auto Iter_res = std::make_shared<Result>(Iter);
    auto body = std::make_shared<Function>(OutputVector{res, Iter_res}, ParameterVector{X, Y, Iter});

    // TI construction
    auto tensor_iterator = std::make_shared<TensorIterator>();
    tensor_iterator->set_body(body);

    tensor_iterator->set_merged_input(X, param, res);
    tensor_iterator->set_invariant_input(Y, param);
    tensor_iterator->set_sliced_input(Iter, iter_count, 0, 1, 1, -1, 0);

    auto output = tensor_iterator->get_iter_value(res, -1);
    auto output_iter = tensor_iterator->get_concatenated_slices(Iter_res, 0, 1, 1, -1, 0);
    function =
        std::make_shared<Function>(OutputVector{output, output_iter}, ParameterVector{param, iter_count}, "PureTI");
}

void MemoryTest::CreateCommonFunc() {
    ov::ParameterVector param{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    const auto variable_info = targetDevice == ov::test::utils::DEVICE_GPU
                                   ? VariableInfo{Shape{inputShape}, ngPrc, "v0"}
                                   : VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"};
    auto variable = std::make_shared<Variable>(variable_info);
    auto read_value = CreateReadValueOp(param.at(0), variable);
    auto add = std::make_shared<Add>(read_value, param.at(0));
    auto assign = CreateAssignOp(add, variable);
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
    }
}

}  // namespace LayerTestsDefinitions

