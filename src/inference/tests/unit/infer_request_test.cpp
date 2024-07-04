// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/infer_request.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iasync_infer_request.hpp"

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

class OVInferRequestBaseTests : public ::testing::Test {
protected:
    std::shared_ptr<ov::MockIAsyncInferRequest> mock_impl;
    ov::InferRequest request;

    void SetUp() override {
        mock_impl.reset(new ov::MockIAsyncInferRequest());
        request.*get(InferRequest_Impl()) = mock_impl;
    }
};

// start_async
TEST_F(OVInferRequestBaseTests, canForwardStartAsync) {
    EXPECT_CALL(*mock_impl.get(), start_async()).Times(1);
    OV_ASSERT_NO_THROW(request.start_async());
}

TEST_F(OVInferRequestBaseTests, canReportErrorInStartAsync) {
    EXPECT_CALL(*mock_impl.get(), start_async()).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(request.start_async(), std::runtime_error, "compare");
}

// wait
TEST_F(OVInferRequestBaseTests, canForwardWait) {
    EXPECT_CALL(*mock_impl.get(), wait()).WillOnce(Return());
    OV_ASSERT_NO_THROW(request.wait());
}

TEST_F(OVInferRequestBaseTests, canReportErrorInWait) {
    EXPECT_CALL(*mock_impl.get(), wait()).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(request.wait(), std::runtime_error, "compare");
}

// Infer
TEST_F(OVInferRequestBaseTests, canForwardInfer) {
    EXPECT_CALL(*mock_impl.get(), infer()).Times(1);
    OV_ASSERT_NO_THROW(request.infer());
}

TEST_F(OVInferRequestBaseTests, canReportErrorInInfer) {
    EXPECT_CALL(*mock_impl.get(), infer()).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(request.infer(), std::runtime_error, "compare");
}

// get_profiling_info
TEST_F(OVInferRequestBaseTests, canForwardGetPerformanceCounts) {
    std::vector<ov::ProfilingInfo> info;
    EXPECT_CALL(*mock_impl.get(), get_profiling_info()).WillOnce(Return(std::vector<ov::ProfilingInfo>{}));
    OV_ASSERT_NO_THROW(request.get_profiling_info());
}

TEST_F(OVInferRequestBaseTests, canReportErrorInGetPerformanceCounts) {
    std::vector<ov::ProfilingInfo> info;
    EXPECT_CALL(*mock_impl.get(), get_profiling_info()).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(request.get_profiling_info(), std::runtime_error, "compare");
}

// get_tensor
TEST_F(OVInferRequestBaseTests, canForwardGetTensor) {
    ov::Tensor data;
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    param->output(0).set_names({"test_name"});
    std::vector<ov::Output<const ov::Node>> inputs{param->output(0)};
    EXPECT_CALL(*mock_impl.get(), get_inputs()).WillOnce(ReturnRef(inputs));
    EXPECT_CALL(*mock_impl.get(), get_outputs()).WillOnce(ReturnRef(inputs));
    EXPECT_CALL(*mock_impl.get(), get_tensors(_)).WillOnce(Return(std::vector<ov::SoPtr<ov::ITensor>>{}));
    EXPECT_CALL(*mock_impl.get(), get_tensor(_)).WillOnce(Return(ov::make_tensor(ov::element::f32, {1, 2, 3, 3})));
    OV_ASSERT_NO_THROW(request.get_tensor("test_name"));
}

TEST_F(OVInferRequestBaseTests, canReportErrorInGetTensor) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    param->output(0).set_names({"test_name"});
    std::vector<ov::Output<const ov::Node>> inputs{param->output(0)};
    EXPECT_CALL(*mock_impl.get(), get_inputs()).WillOnce(ReturnRef(inputs));
    EXPECT_CALL(*mock_impl.get(), get_outputs()).WillOnce(ReturnRef(inputs));
    EXPECT_CALL(*mock_impl.get(), get_tensors(_)).WillOnce(Return(std::vector<ov::SoPtr<ov::ITensor>>{}));
    EXPECT_CALL(*mock_impl.get(), get_tensor(_)).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(request.get_tensor("test_name"), std::runtime_error, "compare");
}

// set_tensor
TEST_F(OVInferRequestBaseTests, canForwardSetTensor) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    param->output(0).set_names({"test_name"});
    std::vector<ov::Output<const ov::Node>> inputs{param->output(0)};
    ov::Tensor data;
    EXPECT_CALL(*mock_impl.get(), get_inputs()).WillOnce(ReturnRef(inputs));
    EXPECT_CALL(*mock_impl.get(), get_outputs()).WillOnce(ReturnRef(inputs));
    EXPECT_CALL(*mock_impl.get(), set_tensor(_, _)).Times(1);
    OV_ASSERT_NO_THROW(request.set_tensor("test_name", data));
}

TEST_F(OVInferRequestBaseTests, canReportErrorInSetTensor) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    param->output(0).set_names({"test_name"});
    std::vector<ov::Output<const ov::Node>> inputs{param->output(0)};
    ov::Tensor data;
    EXPECT_CALL(*mock_impl.get(), get_inputs()).WillOnce(ReturnRef(inputs));
    EXPECT_CALL(*mock_impl.get(), get_outputs()).WillOnce(ReturnRef(inputs));
    EXPECT_CALL(*mock_impl.get(), set_tensor(_, _)).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(request.set_tensor("test_name", data), std::runtime_error, "compare");
}

// set_callback
TEST_F(OVInferRequestBaseTests, canForwardSetCompletionCallback) {
    EXPECT_CALL(*mock_impl.get(), set_callback(_)).Times(1);
    OV_ASSERT_NO_THROW(request.set_callback(nullptr));
}

TEST_F(OVInferRequestBaseTests, canReportErrorInSetCompletionCallback) {
    EXPECT_CALL(*mock_impl.get(), set_callback(_)).WillOnce(Throw(std::runtime_error("compare")));
    OV_EXPECT_THROW_HAS_SUBSTRING(request.set_callback(nullptr), std::runtime_error, "compare");
}
