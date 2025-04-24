// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/op/add.hpp"

namespace {
using MemoryDynamicBatchParams = std::tuple<
    ov::PartialShape,                           // Partial shape for network initialization
    ov::Shape,                                  // Actual shape to be passed to inference request
    int,                                        // Iterations number
    std::string>;                               // Device name

class MemoryDynamicBatch : public ::testing::Test,
                           public ::testing::WithParamInterface<MemoryDynamicBatchParams> {
public:
    static std::string get_test_case_name(::testing::TestParamInfo<MemoryDynamicBatchParams> obj) {
        ov::PartialShape input_phape;
        ov::Shape input_shape;
        int iterations_num;
        std::string target_device;
        std::tie(input_phape, input_shape, iterations_num, target_device) = obj.param;

        std::ostringstream result;
        result << "IS=";
        result << ov::test::utils::partialShape2str({ input_phape }) << "_";
        result << "TS=";
        result << ov::test::utils::partialShape2str({input_shape});
        result << ")_";
        result << "iterations_num=" << iterations_num << "_";
        result << "target_device=" << target_device;
        return result.str();
    }

    void SetUp() override {
        ov::PartialShape input_pshape;
        std::string device_name;
        std::tie(input_pshape, input_shape, iterations_num, device_name) = GetParam();
        std::shared_ptr<ov::Model> model = build_model(element_type, input_pshape);
        std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();

        compiled_model = core->compile_model(model, device_name, { });
        infer_request = compiled_model.create_infer_request();
    }

    static std::shared_ptr<ov::Model> build_model(ov::element::Type type, const ov::PartialShape& shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        const ov::op::util::VariableInfo variable_info { shape, type, "v0" };
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(param, variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, param);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add);
        return std::make_shared<ov::Model>(ov::ResultVector { res }, ov::SinkVector { assign }, ov::ParameterVector{param}, "MemoryDynamicBatchTest");
    }

    static std::vector<int> generate_inputs(const ov::Shape& shape) {
        auto len = ov::shape_size(shape);
        std::vector<int> result {};
        result.reserve(len);
        for (size_t i = 0; i < len; i++)
            result.push_back(static_cast<int>(i));
        return result;
    }

    static std::vector<int> calculate_reference(const std::vector<int>& input, int iterations) {
        std::vector<int> reference {};
        reference.reserve(input.size());
        std::transform(input.begin(), input.end(), std::back_inserter(reference), [iterations](const int &i) {
            return i * iterations;
        });
        return reference;
    }

protected:
    ov::Shape input_shape;
    int iterations_num;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::vector<int> input_data;
    ov::element::Type element_type { ov::element::i32 };
};

TEST_P(MemoryDynamicBatch, MultipleInferencesOnTheSameInfer_request) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    input_data = generate_inputs(input_shape);
    ov::Tensor input_tensor = ov::Tensor(element_type, input_shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    for (int i = 0; i < iterations_num; i++)
        infer_request.infer();
    auto output = infer_request.get_output_tensor(0);
    std::vector<int> reference = calculate_reference(input_data, iterations_num + 1);
    std::vector<int> actual(output.data<int>(), output.data<int>() + output.get_size());
    for (auto actualIt = actual.begin(), referenceIt = reference.begin(); actualIt < actual.end();
        actualIt++, referenceIt++)
        ASSERT_EQ(*actualIt, *referenceIt);
}

TEST_P(MemoryDynamicBatch, ResetVariableState) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    input_data = generate_inputs(input_shape);
    ov::Tensor input_tensor = ov::Tensor(element_type, input_shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    infer_request.query_state().front().reset();
    infer_request.infer();
    auto output = infer_request.get_output_tensor(0);
    std::vector<int> reference = calculate_reference(input_data, 2);
    std::vector<int> actual(output.data<int>(), output.data<int>() + output.get_size());
    for (auto actualIt = actual.begin(), referenceIt = reference.begin(); actualIt < actual.end();
        actualIt++, referenceIt++)
        ASSERT_EQ(*actualIt, *referenceIt);
}

TEST_P(MemoryDynamicBatch, GetVariableState) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    input_data = generate_inputs(input_shape);
    ov::Tensor input_tensor = ov::Tensor(element_type, input_shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    for (int i = 0; i < iterations_num; i++)
        infer_request.infer();
    auto blob = infer_request.query_state().front().get_state();
    std::vector<int> reference = calculate_reference(input_data, iterations_num + 1);
    std::vector<int> actual(blob.data<int>(), blob.data<int>() + blob.get_size());
    for (auto actualIt = actual.begin(), referenceIt = reference.begin(); actualIt < actual.end();
        actualIt++, referenceIt++)
        ASSERT_EQ(*actualIt, *referenceIt);
}

TEST_P(MemoryDynamicBatch, SetVariableState) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    input_data = generate_inputs(input_shape);
    ov::Tensor input_tensor = ov::Tensor(element_type, input_shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    ov::Tensor state = ov::Tensor(element_type, input_shape, input_data.data());
    infer_request.query_state().front().set_state(state);
    for (int i = 0; i < iterations_num; i++)
        infer_request.infer();
    auto output = infer_request.get_output_tensor(0);
    std::vector<int> reference = calculate_reference(input_data, iterations_num + 1);
    std::vector<int> actual(output.data<int>(), output.data<int>() + output.get_size());
    for (auto actualIt = actual.begin(), referenceIt = reference.begin(); actualIt < actual.end();
        actualIt++, referenceIt++)
        ASSERT_EQ(*actualIt, *referenceIt);
}

static ov::PartialShape model_pshape { {1, 19}, 4, 20, 20 };
static std::vector<ov::Shape> input_shapes { { 7, 4, 20, 20 }, { 19, 4, 20, 20 } };
static std::vector<int> iterations_num { 3, 7 };

INSTANTIATE_TEST_SUITE_P(smoke_MemoryDynamicBatch, MemoryDynamicBatch,
                         ::testing::Combine(
                             ::testing::Values(model_pshape),
                             ::testing::ValuesIn(input_shapes),
                             ::testing::ValuesIn(iterations_num),
                             ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         MemoryDynamicBatch::get_test_case_name);
} // namespace
