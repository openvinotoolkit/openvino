// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/set_blob_of_kind.hpp"

#include <shared_test_classes/single_layer/cum_sum.hpp>

#include <ie_compound_blob.h>

using namespace InferenceEngine;

namespace BehaviorTestsDefinitions {

std::string SetBlobOfKindTest::getTestCaseName(testing::TestParamInfo<SetBlobOfKindParams> obj) {
    FuncTestUtils::BlobKind blobKind;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(blobKind, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "Kind=" << blobKind;
    result << " Device="<< targetDevice;
    return result.str();
}

namespace {

bool isBatchedBlobSupported(const std::shared_ptr<Core>& core, const LayerTestsUtils::TargetDevice& targetDevice) {
    const std::vector<std::string> supported_metrics = core->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_METRICS));

    if (std::find(supported_metrics.begin(), supported_metrics.end(),
                 METRIC_KEY(OPTIMIZATION_CAPABILITIES)) == supported_metrics.end()) {
        return false;
    }

    const std::vector<std::string> optimization_caps =
        core->GetMetric(targetDevice, METRIC_KEY(OPTIMIZATION_CAPABILITIES));

    return std::find(optimization_caps.begin(), optimization_caps.end(),
                    METRIC_VALUE(BATCHED_BLOB)) != optimization_caps.end();
}

bool isBlobKindSupported(const std::shared_ptr<Core>& core,
                         const LayerTestsUtils::TargetDevice& targetDevice,
                         FuncTestUtils::BlobKind blobKind) {
    switch (blobKind) {
    case FuncTestUtils::BlobKind::Simple:
        return true;
    case FuncTestUtils::BlobKind::Compound:
        return false;
    case FuncTestUtils::BlobKind::BatchOfSimple:
        return isBatchedBlobSupported(core, targetDevice);
    default:
        THROW_IE_EXCEPTION << "Test does not support the blob kind";
    }
}

} // namespace

Blob::Ptr SetBlobOfKindTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    return makeBlobOfKind(info.getTensorDesc(), blobKind);
}

void SetBlobOfKindTest::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LoadNetwork();

    if (isBlobKindSupported(core, targetDevice, blobKind)) {
        Infer();
    } else {
        ExpectSetBlobThrow();
    }
}

void SetBlobOfKindTest::ExpectSetBlobThrow() {
    inferRequest = executableNetwork.CreateInferRequest();

    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        auto blob = GenerateInput(*info);
        EXPECT_THROW(inferRequest.SetBlob(info->name(), blob),
                     InferenceEngine::details::InferenceEngineException);
    }
}

void SetBlobOfKindTest::SetUp() {
    SizeVector IS{4, 3, 6, 8};
    std::tie(blobKind, targetDevice, configuration) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {IS});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto axisNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<int64_t>{-1})->output(0);
    auto cumSum = std::dynamic_pointer_cast<ngraph::opset4::CumSum>(ngraph::builder::makeCumSum(paramOuts[0], axisNode, false, false));
    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(cumSum)};
    function = std::make_shared<ngraph::Function>(results, params, "InferSetBlob");
}

TEST_P(SetBlobOfKindTest, CompareWithRefs) {
    Run();
}

} // namespace BehaviorTestsDefinitions
