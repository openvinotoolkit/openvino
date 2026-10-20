// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/isync_infer_request.hpp"

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icompiled_model.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iplugin.hpp"

namespace ov::test {

using ::testing::Return, ::testing::ReturnRefOfCopy;

namespace {

class TestSyncInferRequest : public ov::ISyncInferRequest {
public:
    using ov::ISyncInferRequest::ISyncInferRequest;

    void infer() override {}
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

    void set_tensors_impl(const ov::Output<const ov::Node> port,
                          const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override {
        for (const auto& input : get_inputs()) {
            if (input == port) {
                m_batched_tensors[input.get_tensor_ptr()] = tensors;
                return;
            }
        }
        OPENVINO_THROW("Cannot find input tensors for port ", port);
    }

    void run_convert_batched_tensors() {
        convert_batched_tensors();
    }
};

class TrackingRemoteContext : public ov::IRemoteContext {
public:
    const std::string& get_device_name() const override {
        return m_device_name;
    }

    const ov::AnyMap& get_property() const override {
        return m_properties;
    }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type&, const ov::Shape&, const ov::AnyMap&) override {
        return {};
    }

    ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override {
        ++m_host_tensor_creation_count;
        return {ov::make_tensor(type, shape), nullptr};
    }

    size_t host_tensor_creation_count() const {
        return m_host_tensor_creation_count;
    }

private:
    const std::string m_device_name{"TEST"};
    const ov::AnyMap m_properties;
    size_t m_host_tensor_creation_count = 0;
};

std::shared_ptr<const ov::Model> create_string_input_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{2});
    param->set_friendly_name("input0");
    param->output(0).set_names({"tensor_input0"});
    param->set_layout("N");
    auto result = std::make_shared<ov::op::v0::Result>(param);
    result->set_friendly_name("Result0");
    result->output(0).set_names({"tensor_output0"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

}  // namespace

class ISyncInferRequestStringBatchTest : public ::testing::Test {
protected:
    std::shared_ptr<const ov::Model> model = create_string_input_model();
    std::shared_ptr<ov::MockIPlugin> mock_plugin_impl;
    std::shared_ptr<ov::IPlugin> plugin;
    std::shared_ptr<ov::MockICompiledModel> mock_compiled_model;
    std::shared_ptr<TestSyncInferRequest> request;

    void SetUp() override {
        mock_plugin_impl.reset(new ov::MockIPlugin());
        mock_plugin_impl->set_device_name("TEST");
        plugin = std::static_pointer_cast<ov::IPlugin>(mock_plugin_impl);
        mock_compiled_model = std::make_shared<ov::MockICompiledModel>(model, plugin);
        ON_CALL(*mock_compiled_model, inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
        ON_CALL(*mock_compiled_model, outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
        ON_CALL(*mock_plugin_impl, get_default_context(testing::_))
            .WillByDefault(Return(ov::SoPtr<ov::IRemoteContext>()));
        request = std::make_shared<TestSyncInferRequest>(mock_compiled_model);
    }

    void expect_batched_strings_copied_independently(const std::string& value_prefix) {
        std::vector<ov::SoPtr<ov::ITensor>> items;
        for (int i = 0; i < 2; ++i) {
            auto raw_tensor = ov::make_tensor(ov::element::string, ov::Shape{1});
            raw_tensor->data<std::string>()[0] = value_prefix + std::to_string(i);
            items.push_back(ov::SoPtr<ov::ITensor>(raw_tensor));
        }

        OV_ASSERT_NO_THROW(request->set_tensors(model->input(0), items));
        OV_ASSERT_NO_THROW(request->run_convert_batched_tensors());

        auto merged = request->get_tensor(model->input(0));
        ASSERT_TRUE(merged);
        auto* merged_strings = merged->data<std::string>();
        EXPECT_EQ(merged_strings[0], value_prefix + "0");
        EXPECT_EQ(merged_strings[1], value_prefix + "1");

        merged_strings[0] = "mutated";
        EXPECT_EQ(items[0]->data<std::string>()[0], value_prefix + "0");

        merged = {};
        items.clear();
    }
};

TEST_F(ISyncInferRequestStringBatchTest, ConvertBatchedLongStringTensorsCopiesElementsIndependently) {
    // exceeds SSO threshold on mainstream libstdc++/libc++.
    expect_batched_strings_copied_independently(std::string(64, 'A'));
}

TEST_F(ISyncInferRequestStringBatchTest, ConvertBatchedShortStringTensorsCopiesElementsIndependently) {
    // within SSO threshold on mainstream libstdc++/libc++.
    expect_batched_strings_copied_independently("ab");
}

TEST_F(ISyncInferRequestStringBatchTest, ConvertBatchedStringTensorsUseInitializedRemoteHostStorage) {
    auto remote_context = std::make_shared<TrackingRemoteContext>();
    ON_CALL(*mock_plugin_impl, get_default_context(testing::_))
        .WillByDefault(Return(ov::SoPtr<ov::IRemoteContext>(remote_context)));

    expect_batched_strings_copied_independently(std::string(64, 'A'));

    EXPECT_EQ(remote_context->host_tensor_creation_count(), 1);
}

TEST_F(ISyncInferRequestStringBatchTest, ConvertBatchedNumericTensorsStillUsesFastPath) {
    std::vector<ov::SoPtr<ov::ITensor>> items;
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    param->set_layout("N");
    param->output(0).set_names({"tensor_input0"});
    auto result = std::make_shared<ov::op::v0::Result>(param);
    std::shared_ptr<const ov::Model> numeric_model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    auto numeric_compiled_model = std::make_shared<ov::MockICompiledModel>(numeric_model, plugin);
    auto remote_context = std::make_shared<TrackingRemoteContext>();
    ON_CALL(*numeric_compiled_model, inputs()).WillByDefault(ReturnRefOfCopy(numeric_model->inputs()));
    ON_CALL(*numeric_compiled_model, outputs()).WillByDefault(ReturnRefOfCopy(numeric_model->outputs()));
    ON_CALL(*mock_plugin_impl, get_default_context(testing::_))
        .WillByDefault(Return(ov::SoPtr<ov::IRemoteContext>(remote_context)));
    auto numeric_request = std::make_shared<TestSyncInferRequest>(numeric_compiled_model);

    for (float v : {1.0f, 2.0f}) {
        auto raw_tensor = ov::make_tensor(ov::element::f32, ov::Shape{1});
        raw_tensor->data<float>()[0] = v;
        items.push_back(ov::SoPtr<ov::ITensor>(raw_tensor));
    }

    OV_ASSERT_NO_THROW(numeric_request->set_tensors(numeric_model->input(0), items));
    OV_ASSERT_NO_THROW(numeric_request->run_convert_batched_tensors());

    auto merged = numeric_request->get_tensor(numeric_model->input(0));
    ASSERT_TRUE(merged);
    auto* merged_data = merged->data<float>();
    EXPECT_EQ(merged_data[0], 1.0f);
    EXPECT_EQ(merged_data[1], 2.0f);
    EXPECT_EQ(remote_context->host_tensor_creation_count(), 1);
}

}  // namespace ov::test
