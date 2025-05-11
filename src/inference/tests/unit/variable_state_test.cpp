// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/variable_state.hpp"

#include <gmock/gmock.h>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iasync_infer_request.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icompiled_model.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_ivariable_state.hpp"

using namespace ::testing;
using namespace std;

namespace {

struct InferRequest_Impl {
    typedef std::shared_ptr<ov::IAsyncInferRequest> ov::InferRequest::*type;
    friend type get(InferRequest_Impl);
};

template <typename Tag, typename Tag::type M>
struct Rob {
    friend typename Tag::type get(Tag) {
        return M;
    }
};

template struct Rob<InferRequest_Impl, &ov::InferRequest::_impl>;

}  // namespace

class VariableStateTests : public ::testing::Test {
protected:
    shared_ptr<ov::MockIAsyncInferRequest> mock_infer_request;
    shared_ptr<ov::MockIVariableState> mock_variable_state;
    ov::SoPtr<ov::ITensor> state_tensor;
    ov::InferRequest req;

    void SetUp() override {
        mock_infer_request = make_shared<ov::MockIAsyncInferRequest>();
        mock_variable_state = make_shared<ov::MockIVariableState>();
        req.*get(InferRequest_Impl()) = mock_infer_request;
    }
};

class VariableStateMockImpl : public ov::IVariableState {
public:
    VariableStateMockImpl(const std::string& name) : ov::IVariableState(name) {}
    MOCK_METHOD0(reset, void());
};

TEST_F(VariableStateTests, VariableStateInternalCanSaveName) {
    std::shared_ptr<ov::IVariableState> pState(new VariableStateMockImpl("VariableStateMockImpl"));
    ASSERT_STREQ(pState->get_name().c_str(), "VariableStateMockImpl");
}

TEST_F(VariableStateTests, VariableStateInternalCanSaveState) {
    std::shared_ptr<ov::IVariableState> pState(new VariableStateMockImpl("VariableStateMockImpl"));
    float data[] = {123, 124, 125};
    state_tensor = ov::make_tensor(ov::element::f32, {3}, data);

    pState->set_state(state_tensor);
    auto saver = pState->get_state();

    ASSERT_NE(saver, nullptr);
    ASSERT_FLOAT_EQ(saver->data<float>()[0], 123);
    ASSERT_FLOAT_EQ(saver->data<float>()[1], 124);
    ASSERT_FLOAT_EQ(saver->data<float>()[2], 125);
}

TEST_F(VariableStateTests, VariableStateInternalCanSaveStateByReference) {
    std::shared_ptr<ov::IVariableState> pState(new VariableStateMockImpl("VariableStateMockImpl"));
    float data[] = {123, 124, 125};
    state_tensor = ov::make_tensor(ov::element::f32, {3}, data);

    pState->set_state(state_tensor);

    data[0] = 121;
    data[1] = 122;
    data[2] = 123;
    auto saver = pState->get_state();

    ASSERT_NE(saver, nullptr);
    ASSERT_FLOAT_EQ(saver->data<float>()[0], 121);
    ASSERT_FLOAT_EQ(saver->data<float>()[1], 122);
    ASSERT_FLOAT_EQ(saver->data<float>()[2], 123);
}

// Tests for InferRequest::QueryState
TEST_F(VariableStateTests, InferRequestCanConvertOneVariableStateFromCppToAPI) {
    std::vector<ov::SoPtr<ov::IVariableState>> toReturn(1);
    toReturn[0] = mock_variable_state;

    EXPECT_CALL(*mock_infer_request.get(), query_state()).Times(1).WillRepeatedly(Return(toReturn));

    auto state = req.query_state();
    ASSERT_EQ(state.size(), 1);
}

TEST_F(VariableStateTests, InferRequestCanConvertZeroVariableStateFromCppToAPI) {
    std::vector<ov::SoPtr<ov::IVariableState>> toReturn;

    EXPECT_CALL(*mock_infer_request.get(), query_state()).WillOnce(Return(toReturn));

    auto state = req.query_state();
    ASSERT_EQ(state.size(), 0);
}

TEST_F(VariableStateTests, InferRequestCanConvert2VariableStatesFromCPPtoAPI) {
    std::vector<ov::SoPtr<ov::IVariableState>> toReturn;
    toReturn.push_back(mock_variable_state);
    toReturn.push_back(mock_variable_state);

    EXPECT_CALL(*mock_infer_request.get(), query_state()).Times(1).WillRepeatedly(Return(toReturn));

    auto state = req.query_state();
    ASSERT_EQ(state.size(), 2);
}

TEST_F(VariableStateTests, InfReqVariableStatePropagatesReset) {
    std::vector<ov::SoPtr<ov::IVariableState>> toReturn;
    toReturn.push_back(mock_variable_state);

    EXPECT_CALL(*mock_infer_request.get(), query_state()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mock_variable_state.get(), reset()).Times(1);

    auto state = req.query_state();
    state.front().reset();
}

TEST_F(VariableStateTests, InfReqVariableStatePropagatesExceptionsFromReset) {
    std::vector<ov::SoPtr<ov::IVariableState>> toReturn;
    toReturn.push_back(mock_variable_state);

    EXPECT_CALL(*mock_infer_request.get(), query_state()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mock_variable_state.get(), reset()).WillOnce(Throw(std::logic_error("some error")));

    auto state = req.query_state();
    EXPECT_ANY_THROW(state.front().reset());
}

TEST_F(VariableStateTests, InfReqVariableStatePropagatesGetName) {
    std::vector<ov::SoPtr<ov::IVariableState>> toReturn;
    std::string test_name = "someName";
    toReturn.push_back(mock_variable_state);

    EXPECT_CALL(*mock_infer_request.get(), query_state()).Times(1).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mock_variable_state.get(), get_name()).WillOnce(ReturnRef(test_name));

    auto state = req.query_state();
    EXPECT_STREQ(state.front().get_name().c_str(), "someName");
}

TEST_F(VariableStateTests, InfReqVariableStateCanPropagateSetState) {
    std::vector<ov::SoPtr<ov::IVariableState>> toReturn;
    ov::SoPtr<ov::ITensor> saver;
    toReturn.push_back(mock_variable_state);

    EXPECT_CALL(*mock_infer_request.get(), query_state()).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mock_variable_state.get(), set_state(_)).WillOnce(SaveArg<0>(&saver));

    float data[] = {123, 124, 125};
    auto state_tensor = ov::Tensor(ov::element::f32, {3}, data);

    EXPECT_NO_THROW(req.query_state().front().set_state(state_tensor));
    ASSERT_FLOAT_EQ(saver->data<float>()[0], 123);
    ASSERT_FLOAT_EQ(saver->data<float>()[1], 124);
    ASSERT_FLOAT_EQ(saver->data<float>()[2], 125);
}

TEST_F(VariableStateTests, InfReqVariableStateCanPropagateGetLastState) {
    std::vector<ov::SoPtr<ov::IVariableState>> toReturn;

    float data[] = {123, 124, 125};
    state_tensor = ov::make_tensor(ov::element::f32, {3}, data);

    toReturn.push_back(mock_variable_state);

    EXPECT_CALL(*mock_infer_request.get(), query_state()).WillRepeatedly(Return(toReturn));
    EXPECT_CALL(*mock_variable_state.get(), get_state()).WillOnce([&]() -> ov::SoPtr<ov::ITensor>& {
        return state_tensor;
    });

    auto saver = req.query_state().front().get_state();
    ASSERT_TRUE(saver);
    ASSERT_FLOAT_EQ(saver.data<float>()[0], 123);
    ASSERT_FLOAT_EQ(saver.data<float>()[1], 124);
    ASSERT_FLOAT_EQ(saver.data<float>()[2], 125);
}
