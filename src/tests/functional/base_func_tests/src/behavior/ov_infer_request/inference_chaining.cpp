// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVInferenceChaining::getTestCaseName(const testing::TestParamInfo<InferRequestParams>& obj) {
    return OVInferRequestTests::getTestCaseName(obj);
}

std::shared_ptr<ov::Model> OVInferenceChaining::getFirstStaticFunction(const ov::PartialShape &shape) {
    ov::ParameterVector params;
    for (auto&& sp : {shape, shape, shape}) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, sp));
    }
    params[0]->get_output_tensor(0).set_names({"input_tensor_0"});
    params[0]->set_friendly_name("param_0");
    params[1]->get_output_tensor(0).set_names({"input_tensor_1"});
    params[1]->set_friendly_name("param_1");
    params[2]->get_output_tensor(0).set_names({"input_tensor_2"});
    params[2]->set_friendly_name("param_2");
    auto eltwise = ov::test::utils::make_eltwise(params[0], params[1], ov::test::utils::EltwiseTypes::ADD);
    auto eltwise2 = ov::test::utils::make_eltwise(eltwise, params[2], ov::test::utils::EltwiseTypes::ADD);
    eltwise2->get_output_tensor(0).set_names({"result_tensor_0"});
    eltwise2->set_friendly_name("result_0");

    return std::make_shared<ov::Model>(eltwise2, ov::ParameterVector(params));
}

std::shared_ptr<ov::Model> OVInferenceChaining::getSecondStaticFunction(const ov::PartialShape &shape) {
    ov::ParameterVector params;
    for (auto&& sp : {shape, shape}) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, sp));
    }
    params[0]->get_output_tensor(0).set_names({"input_tensor_0"});
    params[0]->set_friendly_name("param_0");
    params[1]->get_output_tensor(0).set_names({"input_tensor_1"});
    params[1]->set_friendly_name("param_1");
    auto eltwise = ov::test::utils::make_eltwise(params[0], params[1], ov::test::utils::EltwiseTypes::MULTIPLY);
    eltwise->get_output_tensor(0).set_names({"result_tensor_0"});
    eltwise->set_friendly_name("result_0");

    return std::make_shared<ov::Model>(eltwise, ov::ParameterVector(params));
}

std::shared_ptr<ov::Model> OVInferenceChaining::getThirdStaticFunction(const ov::PartialShape &shape) {
    ov::ParameterVector params;
    for (auto&& sp : {shape, shape, shape, shape}) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, sp));
    }
    params[0]->get_output_tensor(0).set_names({"input_tensor_0"});
    params[0]->set_friendly_name("param_0");
    params[1]->get_output_tensor(0).set_names({"input_tensor_1"});
    params[1]->set_friendly_name("param_1");
    params[2]->get_output_tensor(0).set_names({"input_tensor_2"});
    params[2]->set_friendly_name("param_2");
    params[3]->get_output_tensor(0).set_names({"input_tensor_3"});
    params[3]->set_friendly_name("param_3");
    auto eltwise = ov::test::utils::make_eltwise(params[0], params[1], ov::test::utils::EltwiseTypes::ADD);
    auto eltwise2 = ov::test::utils::make_eltwise(eltwise, params[2], ov::test::utils::EltwiseTypes::ADD);
    auto eltwise3 = ov::test::utils::make_eltwise(eltwise2, params[3], ov::test::utils::EltwiseTypes::MULTIPLY);
    eltwise3->get_output_tensor(0).set_names({"result_tensor_0"});
    eltwise3->set_friendly_name("result_0");

    return std::make_shared<ov::Model>(eltwise3, ov::ParameterVector(params));
}

void OVInferenceChaining::Run() {
    ov::CompiledModel execNet0, execNet1, execNet2;
    OV_ASSERT_NO_THROW(execNet0 = core->compile_model(function0, target_device, configuration));
    OV_ASSERT_NO_THROW(execNet1 = core->compile_model(function1, target_device, configuration));
    OV_ASSERT_NO_THROW(execNet2 = core->compile_model(function2, target_device, configuration));

    ov::InferRequest r0, r1, r2;
    OV_ASSERT_NO_THROW(r0 = execNet0.create_infer_request());
    OV_ASSERT_NO_THROW(r1 = execNet1.create_infer_request());
    OV_ASSERT_NO_THROW(r2 = execNet2.create_infer_request());

    // perform inference chaining
    if (outputToInput) {
        OV_ASSERT_NO_THROW(r1.set_tensor("input_tensor_0", r0.get_tensor("result_tensor_0")));
    } else {
        OV_ASSERT_NO_THROW(r0.set_tensor("result_tensor_0", r1.get_tensor("input_tensor_0")));
    }

    // create input tensors
    ov::Tensor t0 = tensor(std::vector<float>{1.0f, 2.0f, 3.0f});
    ov::Tensor t1 = tensor(std::vector<float>{4.0f, 5.0f, 6.0f});
    ov::Tensor t2 = tensor(std::vector<float>{7.0f, 8.0f, 9.0f});
    ov::Tensor t3 = tensor(std::vector<float>{2.0f, 3.0f, 2.0f});

    OV_ASSERT_NO_THROW(r0.set_tensor("input_tensor_0", t0));
    OV_ASSERT_NO_THROW(r0.set_tensor("input_tensor_1", t1));
    OV_ASSERT_NO_THROW(r0.set_tensor("input_tensor_2", t2));
    OV_ASSERT_NO_THROW(r1.set_tensor("input_tensor_1", t3));

    OV_ASSERT_NO_THROW(r2.set_tensor("input_tensor_0", t0));
    OV_ASSERT_NO_THROW(r2.set_tensor("input_tensor_1", t1));
    OV_ASSERT_NO_THROW(r2.set_tensor("input_tensor_2", t2));
    OV_ASSERT_NO_THROW(r2.set_tensor("input_tensor_3", t3));

    OV_ASSERT_NO_THROW(r0.infer());
    OV_ASSERT_NO_THROW(r1.infer());
    OV_ASSERT_NO_THROW(r2.infer());

    // check results
    std::vector<float> reference1 = {12.0f, 15.0f, 18.0f};
    std::vector<float> reference2 = {24.0f, 45.0f, 36.0f};

    auto rti = r0.get_tensor("result_tensor_0");
    auto rt0 = r1.get_tensor("result_tensor_0");
    auto rt1 = r2.get_tensor("result_tensor_0");

    for (size_t i = 0; i < reference1.size(); ++i) {
        EXPECT_EQ(reference1[i], rti.data<float>()[i]);
        EXPECT_EQ(reference2[i], rt0.data<float>()[i]);
        EXPECT_EQ(reference2[i], rt1.data<float>()[i]);
    }
}

// DEPRECATED VERSION
TEST_P(OVInferenceChaining, StaticOutputToStaticInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    function0 = getFirstStaticFunction();
    function1 = getSecondStaticFunction();
    function2 = getThirdStaticFunction();

    Run();
}

TEST_P(OVInferenceChainingStatic, StaticOutputToStaticInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    function0 = getFirstStaticFunction();
    function1 = getSecondStaticFunction();
    function2 = getThirdStaticFunction();

    Run();
}

TEST_P(OVInferenceChaining, StaticOutputToDynamicInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const auto dynamic = ov::PartialShape::dynamic(ov::Rank(1));
    function0 = getFirstStaticFunction();
    function1 = getSecondStaticFunction(dynamic);
    function2 = getThirdStaticFunction(dynamic);

    Run();
}

TEST_P(OVInferenceChaining, DynamicOutputToDynamicInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const auto dynamic = ov::PartialShape::dynamic();
    function0 = getFirstStaticFunction(dynamic);
    function1 = getSecondStaticFunction(dynamic);
    function2 = getThirdStaticFunction(dynamic);

    Run();
}

TEST_P(OVInferenceChaining, DynamicInputToDynamicOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    this->outputToInput = false;

    const auto dynamic = ov::PartialShape::dynamic();
    function0 = getFirstStaticFunction(dynamic);
    function1 = getSecondStaticFunction(dynamic);
    function2 = getThirdStaticFunction(dynamic);

    Run();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
