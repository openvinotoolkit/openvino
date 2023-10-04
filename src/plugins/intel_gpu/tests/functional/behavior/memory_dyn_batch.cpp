// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset8.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "openvino/runtime/core.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <cpp/ie_cnn_network.h>
#include <ie_plugin_config.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/common_utils.hpp"

#include <vector>

#include <gtest/gtest.h>

using namespace ngraph;
using namespace opset8;
using namespace ov::test;


using MemoryDynamicBatchParams = std::tuple<
    ov::PartialShape,                           // Partial shape for network initialization
    ov::Shape,                                  // Actual shape to be passed to inference request
    int,                                        // Iterations number
    std::string>;                               // Device name

class MemoryDynamicBatch : public ::testing::Test,
    public ::testing::WithParamInterface<MemoryDynamicBatchParams> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<MemoryDynamicBatchParams> obj) {
        ov::PartialShape inputPartialShape;
        ov::Shape inputShape;
        int iterationsNum;
        std::string targetDevice;
        std::tie(inputPartialShape, inputShape, iterationsNum, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS=";
        result << ov::test::utils::partialShape2str({ inputPartialShape }) << "_";
        result << "TS=";
        result << ov::test::utils::partialShape2str({inputShape});
        result << ")_";
        result << "iterationsCount=" << iterationsNum << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void SetUp() override {
        std::tie(inputPartialShape_, inputShape_, iterationsNum_, deviceName_) = GetParam();
        model_ = buildModel(precision_, inputPartialShape_);
        core_ = ov::test::utils::PluginCache::get().core();
    }

    static std::shared_ptr<ov::Model> buildModel(ElementType precision, const ov::PartialShape& shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(precision, shape);
        const VariableInfo variable_info { shape, precision, "v0" };
        auto variable = std::make_shared<Variable>(variable_info);
        auto read_value = std::make_shared<ReadValue>(param, variable);
        auto add = std::make_shared<Add>(read_value, param);
        auto assign = std::make_shared<Assign>(add, variable);
        auto res = std::make_shared<Result>(add);
        return std::make_shared<ov::Model>(ResultVector { res }, SinkVector { assign }, ov::ParameterVector{param},
            "MemoryDynamicBatchTest");
    }

    static std::vector<int> generateInput(const ov::Shape& shape) {
        auto len = ov::shape_size(shape);
        std::vector<int> result {};
        result.reserve(len);
        for (size_t i = 0; i < len; i++)
            result.push_back(static_cast<int>(i));
        return result;
    }

    static std::vector<int> calculateReference(const std::vector<int>& input, int iterations) {
        std::vector<int> reference {};
        reference.reserve(input.size());
        std::transform(input.begin(), input.end(), std::back_inserter(reference), [iterations](const int &i) {
            return i * iterations;
        });
        return reference;
    }


protected:
    ov::PartialShape inputPartialShape_;
    ov::Shape inputShape_;
    int iterationsNum_;
    std::string deviceName_;
    std::shared_ptr<ov::Model> model_;
    std::shared_ptr<ov::Core> core_;
    std::vector<int> input_;
    ElementType precision_ { ElementType::i32 };
};

TEST_P(MemoryDynamicBatch, MultipleInferencesOnTheSameInferRequest) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto compiledModel = core_->compile_model(model_, ov::test::utils::DEVICE_GPU, { });
    auto inferRequest = compiledModel.create_infer_request();
    input_ = generateInput(inputShape_);
    ov::Tensor inputTensor = ov::Tensor(precision_, inputShape_, input_.data());
    inferRequest.set_input_tensor(inputTensor);
    for (int i = 0; i < iterationsNum_; i++)
        inferRequest.infer();
    auto output = inferRequest.get_output_tensor(0);
    std::vector<int> reference = calculateReference(input_, iterationsNum_);
    std::vector<int> actual(output.data<int>(), output.data<int>() + output.get_size());
    for (auto actualIt = actual.begin(), referenceIt = reference.begin(); actualIt < actual.end();
        actualIt++, referenceIt++)
        EXPECT_EQ(*actualIt, *referenceIt);
}

TEST_P(MemoryDynamicBatch, ResetVariableState) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto compiledModel = core_->compile_model(model_, ov::test::utils::DEVICE_GPU, { });
    auto inferRequest = compiledModel.create_infer_request();
    input_ = generateInput(inputShape_);
    ov::Tensor inputTensor = ov::Tensor(precision_, inputShape_, input_.data());
    inferRequest.set_input_tensor(inputTensor);
    inferRequest.infer();
    inferRequest.query_state().front().reset();
    inferRequest.infer();
    auto output = inferRequest.get_output_tensor(0);
    std::vector<int> reference = calculateReference(input_, 1);
    std::vector<int> actual(output.data<int>(), output.data<int>() + output.get_size());
    for (auto actualIt = actual.begin(), referenceIt = reference.begin(); actualIt < actual.end();
        actualIt++, referenceIt++)
        EXPECT_EQ(*actualIt, *referenceIt);
}

TEST_P(MemoryDynamicBatch, GetVariableState) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto compiledModel = core_->compile_model(model_, ov::test::utils::DEVICE_GPU, { });
    auto inferRequest = compiledModel.create_infer_request();
    input_ = generateInput(inputShape_);
    ov::Tensor inputTensor = ov::Tensor(precision_, inputShape_, input_.data());
    inferRequest.set_input_tensor(inputTensor);
    for (int i = 0; i < iterationsNum_; i++)
        inferRequest.infer();
    auto blob = inferRequest.query_state().front().get_state();
    std::vector<int> reference = calculateReference(input_, iterationsNum_);
    std::vector<int> actual(blob.data<int>(), blob.data<int>() + blob.get_size());
    for (auto actualIt = actual.begin(), referenceIt = reference.begin(); actualIt < actual.end();
        actualIt++, referenceIt++)
        EXPECT_EQ(*actualIt, *referenceIt);
}

TEST_P(MemoryDynamicBatch, SetVariableState) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto compiledModel = core_->compile_model(model_, ov::test::utils::DEVICE_GPU, { });
    auto inferRequest = compiledModel.create_infer_request();
    input_ = generateInput(inputShape_);
    ov::Tensor inputTensor = ov::Tensor(precision_, inputShape_, input_.data());
    inferRequest.set_input_tensor(inputTensor);
    ov::Tensor state = ov::Tensor(precision_, inputShape_, input_.data());
    inferRequest.query_state().front().set_state(state);
    for (int i = 0; i < iterationsNum_; i++)
        inferRequest.infer();
    auto output = inferRequest.get_output_tensor(0);
    std::vector<int> reference = calculateReference(input_, iterationsNum_ + 1);
    std::vector<int> actual(output.data<int>(), output.data<int>() + output.get_size());
    for (auto actualIt = actual.begin(), referenceIt = reference.begin(); actualIt < actual.end();
        actualIt++, referenceIt++)
        EXPECT_EQ(*actualIt, *referenceIt);
}

ov::PartialShape networkPartialShape { {1, 19}, 4, 20, 20 };
std::vector<ov::Shape> inputShapes { { 7, 4, 20, 20 }, { 19, 4, 20, 20 } };
std::vector<int> iterationsNum { 3, 7 };

INSTANTIATE_TEST_SUITE_P(smoke_MemoryDynamicBatch, MemoryDynamicBatch,
                         ::testing::Combine(
                             ::testing::Values(networkPartialShape),
                             ::testing::ValuesIn(inputShapes),
                             ::testing::ValuesIn(iterationsNum),
                             ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         MemoryDynamicBatch::getTestCaseName);
