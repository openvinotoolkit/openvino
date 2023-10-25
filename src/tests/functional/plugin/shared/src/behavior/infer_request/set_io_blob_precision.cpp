// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/set_io_blob_precision.hpp"
#include "ov_models/builders.hpp"

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
            IE_THROW() << "Not supported type for SetBlob";
    }
    return os;
}

std::string SetBlobTest::getTestCaseName(testing::TestParamInfo<SetBlobParams> obj) {
    Precision precNet, precNg;
    setType type;
    std::string targetDevice;
    std::tie(precNet, precNg, type, targetDevice) = obj.param;

    std::ostringstream result;
    result << "Type=" << type << "_";
    result << "Device=" << targetDevice << "_";
    result << "PrecisionInNet=" << precNet << "_";
    result << "PrecisionInNgraph=" << precNg;
    return result.str();
}

inline void fillBlob(Blob::Ptr &blob) {
    switch (blob->getTensorDesc().getPrecision()) {
#define CASE(X) case X: ov::test::utils::fill_data_random<X>(blob); break;
        CASE(Precision::U8)
        CASE(Precision::I8)
        CASE(Precision::U16)
        CASE(Precision::I16)
        CASE(Precision::U32)
        CASE(Precision::I32)
        CASE(Precision::U64)
        CASE(Precision::I64)
        CASE(Precision::BF16)
        CASE(Precision::FP16)
        CASE(Precision::FP32)
        CASE(Precision::FP64)
        CASE(Precision::BOOL)
#undef CASE
        default:
            IE_THROW() << "Can't fill blob with precision: " << blob->getTensorDesc().getPrecision();
    }
}

void SetBlobTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        Blob::Ptr inBlob;
        if (type == setType::INPUT || type == setType::BOTH) {
            inBlob = make_blob_with_precision(inPrc, info->getTensorDesc());
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
            Blob::Ptr outBlob = make_blob_with_precision(outPrc, info->getTensorDesc());
            outBlob->allocate();
            fillBlob(outBlob);
            inferRequest.SetBlob(info->getName(), outBlob);
        }
    }

    inferRequest.Infer();
}

void SetBlobTest::SetUp() {
    SizeVector IS{4, 5, 6, 7};
    Precision precNg;
    std::tie(precNet, precNg, type, targetDevice) = this->GetParam();

    if (type == setType::INPUT || type == setType::BOTH)
        inPrc = precNet;
    if (type == setType::OUTPUT || type == setType::BOTH)
        outPrc = precNet;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precNg);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(IS))};
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
