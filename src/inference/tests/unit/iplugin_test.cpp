// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iplugin.hpp"

#include <gmock/gmock-spec-builders.h>
#include <gtest/gtest.h>

#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icompiled_model.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iplugin.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_isync_infer_request.hpp"

using namespace ::testing;
using namespace std;

class IPluginTest : public ::testing::Test {
private:
    std::shared_ptr<ov::Model> create_model() {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("Param");
        param->output(0).set_names({"param", "name1", "name2", "name3"});

        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        relu->set_friendly_name("ReLU");
        relu->output(0).set_names({"relu"});

        return std::make_shared<ov::Model>(ov::OutputVector{relu->output(0)}, ov::ParameterVector{param});
    }

protected:
    shared_ptr<ov::IPlugin> plugin;
    shared_ptr<ov::MockIPlugin> mock_plugin_impl;
    shared_ptr<ov::MockICompiledModel> mock_compiled_model;
    shared_ptr<ov::MockISyncInferRequest> mock_infer_request;
    std::shared_ptr<const ov::Model> model = create_model();
    std::string pluginId;

    void TearDown() override {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_plugin_impl.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_compiled_model.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_infer_request.get()));
    }

    void SetUp() override {
        pluginId = "TEST";
        mock_plugin_impl.reset(new ov::MockIPlugin());
        mock_plugin_impl->set_device_name(pluginId);
        plugin = std::static_pointer_cast<ov::IPlugin>(mock_plugin_impl);
        mock_compiled_model = make_shared<ov::MockICompiledModel>(model, plugin);
        ON_CALL(*mock_compiled_model.get(), inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
        ON_CALL(*mock_compiled_model.get(), outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
        mock_infer_request = make_shared<ov::MockISyncInferRequest>(mock_compiled_model);
    }

    void getInferRequestWithMockImplInside(std::shared_ptr<ov::IAsyncInferRequest>& request) {
        std::shared_ptr<ov::ICompiledModel> compiled_model;
        EXPECT_CALL(*mock_plugin_impl.get(), compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .WillOnce(Return(mock_compiled_model));
        EXPECT_CALL(*mock_compiled_model.get(), create_sync_infer_request()).WillOnce(Return(mock_infer_request));
        ON_CALL(*mock_compiled_model.get(), create_infer_request()).WillByDefault([&]() {
            return mock_compiled_model->create_infer_request_default();
        });
        compiled_model = plugin->compile_model(model, {});
        ASSERT_NE(nullptr, compiled_model);
        request = compiled_model->create_infer_request();
        ASSERT_NE(nullptr, request);
    }
};

MATCHER_P(blob_in_map_pointer_is_same, ref_blob, "") {
    return reinterpret_cast<float*>(arg.begin()->second->buffer()) == reinterpret_cast<float*>(ref_blob->buffer());
}

TEST_F(IPluginTest, SetTensorWithIncorrectPortNames) {
    ov::SoPtr<ov::ITensor> tensor = ov::make_tensor(ov::element::f32, {1, 3, 2, 2});
    auto updated_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    updated_param->set_friendly_name("Param");

    updated_param->output(0).set_names({"new_name"});
    EXPECT_THROW(mock_infer_request->set_tensor(updated_param->output(0), tensor), ov::Exception);

    updated_param->output(0).set_names({"param", "new_name"});
    EXPECT_THROW(mock_infer_request->set_tensor(updated_param->output(0), tensor), ov::Exception);

    updated_param->output(0).set_names({"new_name", "name2"});
    EXPECT_THROW(mock_infer_request->set_tensor(updated_param->output(0), tensor), ov::Exception);
}

TEST_F(IPluginTest, SetTensorWithCorrectPortNames) {
    ov::SoPtr<ov::ITensor> tensor = ov::make_tensor(ov::element::f32, {1, 3, 2, 2});
    auto updated_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    updated_param->set_friendly_name("Param");

    updated_param->output(0).set_names({"param"});
    EXPECT_NO_THROW(mock_infer_request->set_tensor(updated_param->output(0), tensor));

    updated_param->output(0).set_names({"name1", "param"});
    EXPECT_NO_THROW(mock_infer_request->set_tensor(updated_param->output(0), tensor));

    updated_param->output(0).set_names({"name1", "name2"});
    EXPECT_NO_THROW(mock_infer_request->set_tensor(updated_param->output(0), tensor));

    updated_param->output(0).set_names({"param", "name1", "name2"});
    EXPECT_NO_THROW(mock_infer_request->set_tensor(updated_param->output(0), tensor));

    updated_param->output(0).set_names({"param", "name1", "name2", "name3"});
    EXPECT_NO_THROW(mock_infer_request->set_tensor(updated_param->output(0), tensor));
}

TEST_F(IPluginTest, failToSetTensorWithIncorrectPort) {
    auto incorrect_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2});
    ov::SoPtr<ov::ITensor> tensor = ov::make_tensor(ov::element::f32, {1, 1, 1, 1});
    std::string refError = "Cannot find tensor for port";
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    getInferRequestWithMockImplInside(infer_request);
    try {
        infer_request->set_tensor(incorrect_param->output(0), tensor);
    } catch (ov::Exception& ex) {
        ASSERT_TRUE(std::string{ex.what()}.find(refError) != std::string::npos)
            << "\tExpected: " << refError << "\n\tActual: " << ex.what();
    }
}

TEST_F(IPluginTest, failToSetEmptyITensor) {
    ov::SoPtr<ov::ITensor> tensor;
    std::string refError = "Failed to set tensor. ";
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    getInferRequestWithMockImplInside(infer_request);
    try {
        infer_request->set_tensor(model->input(0), tensor);
    } catch (ov::Exception& ex) {
        ASSERT_TRUE(std::string{ex.what()}.find(refError) != std::string::npos)
            << "\tExpected: " << refError << "\n\tActual: " << ex.what();
    }
}

TEST_F(IPluginTest, SetTensorWithCorrectPort) {
    ov::SoPtr<ov::ITensor> tensor = ov::make_tensor(ov::element::f32, {1, 3, 2, 2});
    std::shared_ptr<ov::IAsyncInferRequest> infer_request;
    getInferRequestWithMockImplInside(infer_request);
    EXPECT_NO_THROW(infer_request->set_tensor(model->input(0), tensor));
}
