// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/compiled_model.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <ostream>
#include <stdexcept>
#include <unit_test_utils/mocks/openvino/runtime/mock_iasync_infer_request.hpp>
#include <unit_test_utils/mocks/openvino/runtime/mock_ivariable_state.hpp>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/variable_state.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icompiled_model.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iplugin.hpp"

using namespace ::testing;

namespace {

struct CompiledModel_Impl {
    typedef std::shared_ptr<ov::ICompiledModel> ov::CompiledModel::*type;
    friend type get(CompiledModel_Impl);
};

template <typename Tag, typename Tag::type M>
struct Rob {
    friend typename Tag::type get(Tag) {
        return M;
    }
};

template struct Rob<CompiledModel_Impl, &ov::CompiledModel::_impl>;

}  // namespace

class CompiledModelTests : public ::testing::Test {
private:
    std::shared_ptr<ov::Model> create_model() {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
        param->set_friendly_name("Param");
        param->output(0).set_names({"param"});

        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        relu->set_friendly_name("ReLU");
        relu->output(0).set_names({"relu"});

        return std::make_shared<ov::Model>(ov::OutputVector{relu->output(0)}, ov::ParameterVector{param});
    }

protected:
    std::shared_ptr<ov::MockICompiledModel> mock_compiled_model;
    ov::CompiledModel compiled_model;
    std::shared_ptr<ov::IPlugin> plugin;
    std::shared_ptr<const ov::Model> model;

    void TearDown() override {
        mock_compiled_model.reset();
        compiled_model = {};
        plugin = {};
    }

    void SetUp() override {
        model = create_model();
        auto mock_plugin = std::make_shared<ov::MockIPlugin>();
        plugin = mock_plugin;
        mock_compiled_model = std::make_shared<ov::MockICompiledModel>(model, plugin);
        compiled_model.*get(CompiledModel_Impl()) = mock_compiled_model;
    }
};

TEST_F(CompiledModelTests, GetOutputsThrowsIfReturnErr) {
    EXPECT_CALL(*mock_compiled_model.get(), outputs()).Times(1).WillOnce(Throw(std::runtime_error{""}));

    ASSERT_THROW(compiled_model.outputs(), std::runtime_error);
}

TEST_F(CompiledModelTests, GetOutputs) {
    std::vector<ov::Output<const ov::Node>> data;
    EXPECT_CALL(*mock_compiled_model.get(), outputs()).Times(1).WillOnce(ReturnRefOfCopy(model->outputs()));
    OV_ASSERT_NO_THROW(data = compiled_model.outputs());
    ASSERT_EQ(data, model->outputs());
}

TEST_F(CompiledModelTests, GetInputsThrowsIfReturnErr) {
    EXPECT_CALL(*mock_compiled_model.get(), inputs()).Times(1).WillOnce(Throw(std::runtime_error{""}));

    ASSERT_THROW(compiled_model.inputs(), std::runtime_error);
}

TEST_F(CompiledModelTests, GetInputs) {
    EXPECT_CALL(*mock_compiled_model.get(), inputs()).Times(1).WillOnce(ReturnRefOfCopy(model->inputs()));

    std::vector<ov::Output<const ov::Node>> info;
    OV_ASSERT_NO_THROW(info = compiled_model.inputs());
    ASSERT_EQ(info, model->inputs());
}

class CompiledModelWithIInferReqTests : public CompiledModelTests {
protected:
    std::shared_ptr<ov::MockIAsyncInferRequest> mock_infer_request;

    void SetUp() override {
        CompiledModelTests::SetUp();
        mock_infer_request = std::make_shared<ov::MockIAsyncInferRequest>();
    }
};

TEST_F(CompiledModelWithIInferReqTests, CanCreateInferRequest) {
    EXPECT_CALL(*mock_compiled_model.get(), create_infer_request()).WillOnce(Return(mock_infer_request));
    ov::InferRequest actualInferReq;
    OV_ASSERT_NO_THROW(actualInferReq = compiled_model.create_infer_request());
}

TEST_F(CompiledModelWithIInferReqTests, CreateInferRequestThrowsIfReturnNotOK) {
    EXPECT_CALL(*mock_compiled_model.get(), create_infer_request()).WillOnce(Throw(std::runtime_error{""}));
    ASSERT_THROW(compiled_model.create_infer_request(), std::runtime_error);
}

TEST_F(CompiledModelWithIInferReqTests, QueryStateThrowsIfReturnErr) {
    EXPECT_CALL(*mock_compiled_model.get(), create_infer_request()).WillOnce(Return(mock_infer_request));
    ov::InferRequest actualInferReq;
    OV_ASSERT_NO_THROW(actualInferReq = compiled_model.create_infer_request());
    EXPECT_CALL(*mock_infer_request.get(), query_state()).Times(1).WillOnce(Throw(std::runtime_error{""}));
    EXPECT_THROW(actualInferReq.query_state(), std::runtime_error);
}

TEST_F(CompiledModelWithIInferReqTests, QueryState) {
    EXPECT_CALL(*mock_compiled_model.get(), create_infer_request()).WillOnce(Return(mock_infer_request));
    ov::InferRequest actualInferReq;
    OV_ASSERT_NO_THROW(actualInferReq = compiled_model.create_infer_request());
    ov::SoPtr<ov::IVariableState> state = std::make_shared<ov::MockIVariableState>();
    EXPECT_CALL(*mock_infer_request.get(), query_state())
        .Times(1)
        .WillOnce(Return(std::vector<ov::SoPtr<ov::IVariableState>>(1, state)));
    std::vector<ov::VariableState> MemState_v;
    MemState_v = actualInferReq.query_state();
    EXPECT_EQ(MemState_v.size(), 1);
}

class CompiledModelBaseTests : public ::testing::Test {
protected:
    std::shared_ptr<ov::MockICompiledModel> mock_compiled_model;
    ov::CompiledModel compiled_model;
    std::shared_ptr<ov::IPlugin> plugin;

    void SetUp() override {
        auto mock_plugin = std::make_shared<ov::MockIPlugin>();
        plugin = mock_plugin;
        mock_compiled_model = std::make_shared<ov::MockICompiledModel>(nullptr, plugin);
        compiled_model.*get(CompiledModel_Impl()) = mock_compiled_model;
    }
};

// CreateInferRequest
TEST_F(CompiledModelBaseTests, canForwardCreateInferRequest) {
    auto inferReqInternal = std::make_shared<ov::MockIAsyncInferRequest>();
    EXPECT_CALL(*mock_compiled_model.get(), create_infer_request()).Times(1).WillRepeatedly(Return(inferReqInternal));
    OV_ASSERT_NO_THROW(compiled_model.create_infer_request());
}

TEST_F(CompiledModelBaseTests, canReportErrorInCreateInferRequest) {
    EXPECT_CALL(*mock_compiled_model.get(), create_infer_request()).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(compiled_model.create_infer_request(), std::runtime_error, "compare");
}

// Export
TEST_F(CompiledModelBaseTests, canForwardExport) {
    std::stringstream out_model;
    EXPECT_CALL(*mock_compiled_model.get(), export_model(_)).Times(1);
    EXPECT_NO_THROW(compiled_model.export_model(out_model));
}

TEST_F(CompiledModelBaseTests, canReportErrorInExport) {
    std::stringstream out_model;
    EXPECT_CALL(*mock_compiled_model.get(), export_model(_)).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(compiled_model.export_model(out_model), std::runtime_error, "compare");
}
