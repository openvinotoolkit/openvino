// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/set_blob.hpp"
#include <shared_test_classes/single_layer/cum_sum.hpp>

using namespace InferenceEngine;

namespace BehaviorTestsDefinitions {

std::ostream& operator<<(std::ostream & os, setType type) {
    switch (type) {
    case setType::INPUT:
        os << "INPUT";
        break;
    case setType::OUTPUT:
        os << "OUTPUT";
        break;
    case setType::BOTH:
        os << "BOTH";
        break;
    default:
        THROW_IE_EXCEPTION << "Not supported type for SetBlob";
    }
    return os;
}

std::string SetBlobTest::getTestCaseName(testing::TestParamInfo<SetBlobParams> obj) {
    Precision prec;
    setType type;
    std::string targetDevice;
    std::tie(prec, type, targetDevice) = obj.param;

    std::ostringstream result;
    result << "Type="<< type;
    result << " Device="<< targetDevice;
    result << " Precision=" << prec;
    return result.str();
}

inline void fillBlob(Blob::Ptr &blob) {
    switch (blob->getTensorDesc().getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_random<X>(blob); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::U64)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            THROW_IE_EXCEPTION << "Can't fill blob with precision: " << blob->getTensorDesc().getPrecision();
    }
}

void SetBlobTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        Blob::Ptr inBlob;
        if (type == setType::INPUT || type == setType::BOTH) {
            inBlob = make_blob_with_precision(precision, info->getTensorDesc());
            inBlob->allocate();
            fillBlob(inBlob);
        } else {
            inBlob = GenerateInput(*info);
        }
        inferRequest.SetBlob(info->name(), inBlob);
        inputs.push_back(inBlob);
    }

    if (type == setType::OUTPUT || type == setType::BOTH) {
        for (const auto &output : executableNetwork.GetOutputsInfo()) {
            const auto &info = output.second;
            Blob::Ptr outBlob = make_blob_with_precision(precision, info->getTensorDesc());
            outBlob->allocate();
            fillBlob(outBlob);
            inferRequest.SetBlob(info->getName(), outBlob);
        }
    }

    inferRequest.Infer();
}

void SetBlobTest::SetUp() {
    SizeVector IS{4, 5, 6, 7};
    std::tie(precision, type, targetDevice) = this->GetParam();

    if (type == setType::OUTPUT || type == setType::BOTH)
        outPrc = precision;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);
    auto params = ngraph::builder::makeParams(ngPrc, {IS});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto axisNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<int64_t>{-1})->output(0);
    auto cumSum = std::dynamic_pointer_cast<ngraph::opset4::CumSum>(ngraph::builder::makeCumSum(paramOuts[0], axisNode, false, false));
    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(cumSum)};
    function = std::make_shared<ngraph::Function>(results, params, "InferSetBlob");
}

TEST_P(SetBlobTest, CompareWithRefs) {
    Run();
}

} // namespace BehaviorTestsDefinitions
