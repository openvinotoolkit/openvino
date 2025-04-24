// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/wait.hpp"
#include "openvino/runtime/exception.hpp"

namespace ov {
namespace test {
namespace behavior {
void OVInferRequestWaitTests::SetUp() {
    OVInferRequestTests::SetUp();
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    req = execNet.create_infer_request();
    input = execNet.input();
    output = execNet.output();
}

std::string OVInferRequestWaitTests::getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
    return OVInferRequestTests::getTestCaseName(obj);
}

void OVInferRequestWaitTests::TearDown() {
    req = {};
    input = {};
    output = {};
    OVInferRequestTests::TearDown();
}

TEST_P(OVInferRequestWaitTests, CorrectOneAsyncInferWithGetInOutWithInfWait) {
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(output));
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(OVInferRequestWaitTests, canstart_asyncInferWithGetInOutWithStatusOnlyWait) {
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait_for({}));
}

TEST_P(OVInferRequestWaitTests, canWaitWithotStartSsync) {
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(req.wait_for({}));
    OV_ASSERT_NO_THROW(req.wait_for(std::chrono::milliseconds{1}));
}

TEST_P(OVInferRequestWaitTests, throwExceptionOnSetTensorAfterAsyncInfer) {
    auto&& config = configuration;
    auto itConfig = config.find(ov::num_streams.name());
    if (itConfig != config.end()) {
        if (itConfig->second.as<ov::streams::Num>() != ov::streams::AUTO) {
            if (std::stoi(itConfig->second.as<std::string>()) == 0) {
                GTEST_SKIP() << "Not applicable with disabled streams";
            }
        }
    }
    auto output_tensor = req.get_tensor(input);
    OV_ASSERT_NO_THROW(req.wait_for({}));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(try {
        req.set_tensor(input, output_tensor);
    } catch (const ov::Busy&) {});
    OV_ASSERT_NO_THROW(req.wait_for({}));
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestWaitTests, throwExceptionOnGetTensorAfterAsyncInfer) {
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(try {
        req.get_tensor(input);
    } catch (const ov::Busy&) {});
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestWaitTests, FailedAsyncInferWithNegativeTimeForWait) {
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    ASSERT_THROW(req.wait_for(std::chrono::milliseconds{-1}), ov::Exception);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
