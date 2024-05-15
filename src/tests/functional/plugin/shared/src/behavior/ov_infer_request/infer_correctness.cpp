// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <iostream>
#include <thread>

#include "behavior/ov_infer_request/infer_consistency.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/avg_pool.hpp"
#include "common_test_utils/data_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace behavior {
std::shared_ptr<ov::Model> GetDefaultGraph() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 224, 224});
    size_t spatialDims = 2;
    std::vector<ptrdiff_t> padBegin(spatialDims, 0), padEnd(spatialDims, 0);
    ov::Shape strides(spatialDims, 1);
    auto weights = ov::test::utils::make_constant(ov::element::f32, {64, 3, 7, 7});
    auto conv1 = std::make_shared<ov::op::v1::Convolution>(input, weights, strides,
            padBegin, padEnd, strides);
    auto gamma = ov::test::utils::make_constant(ov::element::f32, {64});
    auto beta = ov::test::utils::make_constant(ov::element::f32, {64});
    auto mean = ov::test::utils::make_constant(ov::element::f32, {64});
    auto variance = ov::test::utils::make_constant(ov::element::f32, {64});
    auto batchNorm1 = std::make_shared<ov::op::v0::BatchNormInference>(conv1, gamma,
            beta, mean, variance, 1e-5);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(batchNorm1);
    auto pool = std::make_shared<ov::op::v1::AvgPool>(relu1, strides, ov::Shape{1, 1},
            ov::Shape{1, 1}, ov::Shape{4, 4}, true);
    return std::make_shared<ov::Model>(ov::OutputVector{pool},
            ov::ParameterVector{input},
            "autoSampleGraph");
}

void OVInferConsistencyTest::SetUp() {
    std::tie(_inferReqNumPerModel, _inferNumPerInfer,
        _deviceConfigs) = WithParamInterface::GetParam();
    auto function = GetDefaultGraph();
    // prepare model and inferRequests
    for (auto&& item : _deviceConfigs) {
        ModelContext modelContext;
        modelContext._model = core->compile_model(function, item.first, item.second);
        if (_inferReqNumPerModel == 0) {
            try {
                _inferReqNumPerModel =  modelContext._model.get_property(ov::optimal_number_of_infer_requests);
            } catch (...) {
                throw("cannot deduce infer request number");
            }
        }
        for (auto i = 0; i < _inferReqNumPerModel; i++) {
            InferContext inferContext;
            inferContext._inferRequest = modelContext._model.create_infer_request();
            FillInput(inferContext, i + 1);
            modelContext._inferContexts.push_back(std::move(inferContext));
        }
        _modelContexts.push_back(std::move(modelContext));
    }
    ASSERT_GT(_modelContexts.size(), 1);
    // prepare expect inferRequst results which will be compared
    std::vector<std::thread> threads;
    for (auto i = 1; i < _modelContexts.size(); i++) {
        threads.push_back(std::thread{[i, this]() {
            for (auto& inferContext : _modelContexts[i]._inferContexts) {
                inferContext._inferRequest.start_async();
                inferContext._inferRequest.wait();
                inferContext._outputs = GetAllOutputs(_modelContexts[i]._model,
                        inferContext._inferRequest);
            }
        }});
    }
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void OVInferConsistencyTest::TearDown() {
    _modelContexts.clear();
    _deviceConfigs.clear();
}
std::string OVInferConsistencyTest::getTestCaseName(const
    testing::TestParamInfo<ParamType>& obj) {
    unsigned int inferReqNumPerModel; //inferRequst nums per model
    unsigned int inferNumPerInfer; //infer nums wil do per  inferRequest
    std::vector<std::pair<std::string, ov::AnyMap>> deviceConfigs; // devicesConfigs
    std::tie(inferReqNumPerModel, inferNumPerInfer, deviceConfigs) = obj.param;
    std::ostringstream result;
    for (auto&& item : deviceConfigs) {
        result << "device_" << item.first << "_";
        for (auto&& configs : item.second) {
            result << "key_" << configs.first << "_";
            result << "value_" << configs.second.as<std::string>() << "_";
        }
    }
    result << "inferReqNumPerModel_" << inferReqNumPerModel << "_";
    result << "inferNumPerInfer_" << inferNumPerInfer;
    return result.str();
}

bool OVInferConsistencyTest::IsEqual(std::vector<ov::Tensor>& a,
    std::vector<ov::Tensor>& b) {
    if (a.size() == 0 || a.size() != b.size()) {
        return false;
    }
    bool isEqual = true;
    for (size_t j = 0; j < a.size(); j++) {
        if (a[j].get_size() != b[j].get_size()) {
            return false;
        }
        try {
            // if not equal will throw exception
            ov::test::utils::compare_raw_data(
                a[j].data<float>(), b[j].data<float>(), a[j].get_size(), 1e-2f);
        } catch (...) {
            isEqual = false;
            break;
        }
    }
    return isEqual;
}


std::vector<ov::Tensor>
OVInferConsistencyTest::GetAllOutputs(ov::CompiledModel& model,
    ov::InferRequest& inferRequest) {
    std::vector<ov::Tensor> outputs;
    for (auto i = 0; i < model.outputs().size(); i++) {
        outputs.push_back(inferRequest.get_output_tensor(i));
    }
    return outputs;
}

void OVInferConsistencyTest::InferCheck(bool isSync) {
    ASSERT_GT(_modelContexts.size(), 1);
    std::vector<std::thread> threads;
    for (auto i = 0; i < _modelContexts[0]._inferContexts.size(); i++) {
        threads.push_back(std::thread{[i, isSync, this]() {
            for (auto num = 0; num < _inferNumPerInfer; num++) {
                auto inferContext = _modelContexts[0]._inferContexts[i];
                if (isSync) {
                    inferContext._inferRequest.infer();
                } else {
                    inferContext._inferRequest.start_async();
                    inferContext._inferRequest.wait();
                }
                auto actualoutputs = GetAllOutputs(_modelContexts[0]._model,
                        inferContext._inferRequest);
                bool isSame = false;
                // compare with devices, the result is same with one of them, break loop
                for (auto y = 1; y < _modelContexts.size(); y++) {
                    auto expectOutPuts = _modelContexts[y]._inferContexts[i]._outputs;
                    isSame = isSame || IsEqual(expectOutPuts, actualoutputs);
                    if (isSame) {
                        break;
                    }
                }
                EXPECT_TRUE(isSame);
            }
        }});
    }
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// different index will create different input, index shoud > 0
void OVInferConsistencyTest::FillInput(InferContext& inferContext, int index) {
    ASSERT_GT(index, 0);
    inferContext._inputs.clear();
    auto input_tensor =
        inferContext._inferRequest.get_input_tensor(0);
    auto data = input_tensor.data<float>();
    ov::test::utils::fill_data(data, 1 * 3 * 224 * 224, index);
    inferContext._inputs.push_back(input_tensor);
}

TEST_P(OVInferConsistencyTest, Infer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferCheck(true);
}

TEST_P(OVInferConsistencyTest, AsyncInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    InferCheck(false);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
