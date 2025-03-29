// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/opsets/opset8.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "behavior/ov_infer_request/inference.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestInferenceTests::getTestCaseName(
        const testing::TestParamInfo<OVInferRequestInferenceTestsParams>& obj) {
    return std::get<0>(obj.param).m_test_name + "_targetDevice=" + std::get<1>(obj.param);
}

void OVInferRequestInferenceTests::SetUp() {
    m_param = std::get<0>(GetParam());
    target_device = std::get<1>(GetParam());
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
}

std::shared_ptr<Model> OVInferRequestInferenceTests::create_n_inputs(size_t n,
                                                                     element::Type type,
                                                                     const PartialShape& shape) {
    ResultVector res;
    ParameterVector params;
    for (size_t i = 0; i < n; i++) {
        auto index_str = std::to_string(i);
        auto data1 = std::make_shared<ov::op::v0::Parameter>(type, shape);
        data1->set_friendly_name("input" + index_str);
        data1->get_output_tensor(0).set_names({"tensor_input" + index_str});
        auto constant = opset8::Constant::create(type, {1}, {1});
        auto op1 = std::make_shared<ov::op::v1::Add>(data1, constant);
        op1->set_friendly_name("Add" + index_str);
        auto res1 = std::make_shared<ov::op::v0::Result>(op1);
        res1->set_friendly_name("Result" + index_str);
        res1->get_output_tensor(0).set_names({"tensor_output" + index_str});
        params.push_back(data1);
        res.push_back(res1);
    }
    return std::make_shared<Model>(res, params);
}

TEST_P(OVInferRequestInferenceTests, Inference_ROI_Tensor) {
    auto shape_size = ov::shape_size(m_param.m_shape);
    auto model = OVInferRequestInferenceTests::create_n_inputs(1, element::f32, m_param.m_shape);
    auto execNet = ie->compile_model(model, target_device);
    // Create InferRequest
    ov::InferRequest req;
    req = execNet.create_infer_request();
    const std::string tensor_name = "tensor_input0";
    req.set_tensor(tensor_name, m_param.m_input_tensor);
    req.infer();
    auto actual_out_tensor = req.get_tensor("tensor_output0");
    auto out_ptr = actual_out_tensor.data<float>();
    for (size_t i = 0; i < shape_size; ++i) {
        EXPECT_EQ(out_ptr[i], m_param.m_expected[i]) << "Expected="
                                                     << m_param.m_expected[i]
                                                     << ", actual="
                                                     << out_ptr[i]
                                                     << " for "
                                                     << i;
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
