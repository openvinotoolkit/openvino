// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/core/model.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "behavior/ov_infer_request/iteration_chaining.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/concat.hpp"

namespace ov {
namespace test {
namespace behavior {
std::string OVIterationChaining::getTestCaseName(const testing::TestParamInfo<InferRequestParams>& obj) {
    return OVInferRequestTests::getTestCaseName(obj);
}

std::shared_ptr<ov::Model> OVIterationChaining::getIterativeFunction() {
    const ov::PartialShape pshape{-1, 16};
    auto params = std::make_shared<ov::op::v0::Parameter>(element::Type_t::f32, pshape);
    params->get_output_tensor(0).set_names({"input_tensor_0"});
    params->set_friendly_name("param_0");
    auto concat_const = ov::test::utils::make_constant(element::Type_t::f32, {1, 16});
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{params, concat_const}, 0 /*axis*/);
    auto eltwise_const = ov::test::utils::make_constant(element::Type_t::f32, {1, 16});
    auto eltwise = ov::test::utils::make_eltwise(concat, eltwise_const, ov::test::utils::EltwiseTypes::ADD);
    concat->get_output_tensor(0).set_names({"result_tensor_0"});
    concat->set_friendly_name("result_0");
    eltwise->get_output_tensor(0).set_names({"result_tensor_1"});
    eltwise->set_friendly_name("result_1");

    return std::make_shared<ov::Model>(ov::NodeVector{concat, eltwise}, ov::ParameterVector{params});
}

void OVIterationChaining::SetUp() {
    std::tie(target_device, configuration) = this->GetParam();
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
    function = getIterativeFunction();
    ov::AnyMap params;
    for (auto&& v : configuration) {
        params.emplace(v.first, v.second);
    }
    execNet = core->compile_model(function, target_device, params);

    try {
        req = execNet.create_infer_request();
    } catch (const std::exception& ex) {
        FAIL() << "Can't Create Infer Requiest in SetUp \nException [" << ex.what() << "]"
               << std::endl;
    }
}

void OVIterationChaining::TearDown() {
    req = {};
    OVInferRequestTests::TearDown();
}

bool OVIterationChaining::checkOutput(const ov::Tensor& in, const ov::Tensor& actual) {
    bool result = true;
    auto net = core->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
    ov::InferRequest req;
    req = net.create_infer_request();
    auto tensor = req.get_tensor(function->inputs().back().get_any_name());
    tensor.set_shape(in.get_shape());
    for (int i = 0; i < in.get_size(); i++) {
        tensor.data<float>()[i] = in.data<float>()[i];
    }
    req.infer();
    for (int i = 0; i < actual.get_size(); i++) {
        if (fabs(req.get_output_tensor(0).data<float>()[i] - actual.data<float>()[i]) > std::numeric_limits<float>::epsilon())
            return false;
    }
    return result;
}

void OVIterationChaining::Run() {
    // perform iteration chaining by iteratively
    // setting input tensor to be output tensor of last inference, and
    // beginnign with an empty tensor
    ov::Tensor t0(element::Type_t::f32, {0, 16});

    OV_ASSERT_NO_THROW(req.set_tensor("input_tensor_0", t0));
    for (size_t i = 0; i < 10; i++) {
        OV_ASSERT_NO_THROW(req.infer());
        ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor_0"), req.get_tensor("result_tensor_0")));

        const auto t1 = req.get_tensor("result_tensor_0");
        OV_ASSERT_NO_THROW(req.set_tensor("input_tensor_0", t1));
    }
    ASSERT_TRUE(checkOutput(req.get_tensor("input_tensor_0"), req.get_tensor("result_tensor_0")));
}

TEST_P(OVIterationChaining, Simple) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
