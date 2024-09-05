// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset9.hpp"
#include "utils/cpu_test_utils.hpp"
#include "snippets/op/subgraph.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/graph_comparator.hpp"


using namespace CPUTestUtils;
using namespace ov::opset9;

namespace ov {
namespace test {

class SubgraphSnippetSerializationTest : public ::testing::Test, public CPUTestsBase {};

TEST_F(SubgraphSnippetSerializationTest, smoke_SerializeSubgraph) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto model = ([] () -> std::shared_ptr<ov::Model> {
        auto shape = ov::Shape({2, 2});
        auto input0 = std::make_shared<Parameter>(ov::element::f32, shape);
        auto input1 = std::make_shared<Parameter>(ov::element::f32, shape);
        auto ininput0 = std::make_shared<Parameter>(ov::element::f32, shape);
        auto ininput1 = std::make_shared<Parameter>(ov::element::f32, shape);
        auto add = std::make_shared<Add>(ininput0, ininput1);
        auto subgraph_body = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{ininput0, ininput1});
        auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(ov::NodeVector{input0, input1}, subgraph_body.get()->clone());
        return std::make_shared<ov::Model>(ov::NodeVector{subgraph}, ov::ParameterVector{input0, input1});
    })();
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    std::stringstream stream;
    compiled_model.export_model(stream);
    ov::CompiledModel imported_compiled_model = core.import_model(stream, "CPU");
    float data[] = {1.f, 1.f, 1.f, 1.f};
    ov::Tensor input_data1{ov::element::f32, ov::Shape({2, 2}), data};
    ov::Tensor input_data2{ov::element::f32, ov::Shape({2, 2}), data};
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(0, input_data1);
    infer_request.set_input_tensor(1, input_data2);
    infer_request.infer();
    auto out = infer_request.get_output_tensor(0);
    float* out_p = static_cast<float*>(out.data(ov::element::Type_t::f32));
    auto out_val = std::vector<float>(out_p, out_p + out.get_size());
    ov::InferRequest imported_infer_request = imported_compiled_model.create_infer_request();
    imported_infer_request.set_input_tensor(0, input_data1);
    imported_infer_request.set_input_tensor(1, input_data2);
    imported_infer_request.infer();
    auto imported_out = imported_infer_request.get_output_tensor(0);
    float* imported_out_p = static_cast<float*>(imported_out.data(ov::element::Type_t::f32));
    auto imported_out_val = std::vector<float>(imported_out_p, imported_out_p + imported_out.get_size());
    ASSERT_EQ(out_val, imported_out_val);

    auto compiled_model_runtime = compiled_model.get_runtime_model()->clone();
    auto imported_compiled_model_runtime = imported_compiled_model.get_runtime_model()->clone();
    const auto fc = FunctionsComparator::with_default()
                                .enable(FunctionsComparator::CONST_VALUES)
                                .enable(FunctionsComparator::ATTRIBUTES);
    const auto results = fc.compare(compiled_model_runtime, imported_compiled_model_runtime);

    ASSERT_TRUE(results.valid) << results.message;
}

TEST_F(SubgraphSnippetSerializationTest, smoke_SerializeSubgraphWithScalarConst) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto model = ([] () -> std::shared_ptr<ov::Model> {
        auto shape = ov::Shape({1});
        auto input = std::make_shared<Parameter>(ov::element::f32, shape);
        auto internal_input = std::make_shared<Parameter>(ov::element::f32, shape);
        auto constant = std::make_shared<Constant>(ov::element::f32, shape, 2);
        auto internal_constant = std::make_shared<Constant>(ov::element::f32, shape, 2);
        auto add = std::make_shared<Add>(input, constant);
        auto internal_add = std::make_shared<Add>(internal_input, internal_constant);
        auto subgraph_body = std::make_shared<ov::Model>(ov::NodeVector{internal_add}, ov::ParameterVector{internal_input});
        auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(ov::NodeVector{add}, subgraph_body.get()->clone());
        return std::make_shared<ov::Model>(ov::NodeVector{subgraph}, ov::ParameterVector{input});
    })();
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    std::stringstream stream;
    compiled_model.export_model(stream);
    float data[] = {1.f};
    ov::Tensor input_data1{ov::element::f32, ov::Shape({1}), data};
    ov::CompiledModel imported_compiled_model = core.import_model(stream, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(0, input_data1);
    infer_request.infer();
    auto out = infer_request.get_output_tensor(0);
    float* out_p = static_cast<float*>(out.data(ov::element::Type_t::f32));
    auto out_val = std::vector<float>(out_p, out_p + out.get_size());
    ov::InferRequest imported_infer_request = imported_compiled_model.create_infer_request();
    imported_infer_request.set_input_tensor(0, input_data1);
    imported_infer_request.infer();
    auto imported_out = imported_infer_request.get_output_tensor(0);
    float* imported_out_p = static_cast<float*>(imported_out.data(ov::element::Type_t::f32));
    auto imported_out_val = std::vector<float>(imported_out_p, imported_out_p + imported_out.get_size());
    ASSERT_EQ(out_val, imported_out_val);

    auto compiled_model_runtime = compiled_model.get_runtime_model()->clone();
    auto imported_compiled_model_runtime = imported_compiled_model.get_runtime_model()->clone();
    const auto fc = FunctionsComparator::with_default()
                                .enable(FunctionsComparator::CONST_VALUES)
                                .enable(FunctionsComparator::ATTRIBUTES);
    const auto results = fc.compare(compiled_model_runtime, imported_compiled_model_runtime);

    ASSERT_TRUE(results.valid) << results.message;
}

TEST_F(SubgraphSnippetSerializationTest, smoke_SerializeSubgraphWithResultAs1stOutput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto precision = ov::element::f32;
    auto shape = ov::Shape{1, 3, 16, 16};

    auto model = [&] () -> std::shared_ptr<ov::Model> {
        auto input1 = std::make_shared<Parameter>(precision, shape);
        auto input2 = std::make_shared<Parameter>(precision, shape);
        auto sinh1 = std::make_shared<Sinh>(input1);
        auto sinh2 = std::make_shared<Sinh>(input2);

        auto relu = std::make_shared<Relu>(sinh2);
        auto sinh_out = std::make_shared<Sinh>(relu);
        auto result1 = std::make_shared<Result>(sinh_out);

        auto add = std::make_shared<Add>(sinh1, relu);
        auto result2 = std::make_shared<Result>(add);

        ov::ParameterVector params{input1, input2};
        ov::ResultVector results{result1, result2};
        return std::make_shared<ov::Model>(results, params);
    }();
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    std::stringstream stream;
    compiled_model.export_model(stream);
    ov::CompiledModel imported_compiled_model = core.import_model(stream, "CPU");

    auto compiled_model_runtime = compiled_model.get_runtime_model()->clone();
    auto imported_compiled_model_runtime = imported_compiled_model.get_runtime_model()->clone();
    const auto fc = FunctionsComparator::with_default()
                                .enable(FunctionsComparator::CONST_VALUES)
                                .enable(FunctionsComparator::ATTRIBUTES);
    const auto results = fc.compare(compiled_model_runtime, imported_compiled_model_runtime);

    ASSERT_TRUE(results.valid) << results.message;
}
}  // namespace test
}  // namespace ov
