// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <tuple>
#include <vector>

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace GPULayerTestsDefinitions {

typedef std::tuple<int64_t,                      // keepK
                   int64_t,                      // axis
                   ov::op::TopKMode,             // mode
                   ov::op::TopKSortType,         // sort
                   bool,                         // stable
                   InferenceEngine::Precision,   // Net precision
                   InferenceEngine::Precision,   // Input precision
                   InferenceEngine::Precision,   // Output precision
                   InferenceEngine::Layout,      // Input layout
                   InferenceEngine::SizeVector,  // inputShape
                   std::string                   // Target device name
                   >
    TopKGPUParams;

class TopKLayerTestGPU : public testing::WithParamInterface<TopKGPUParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TopKGPUParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;
};

std::string TopKLayerTestGPU::getTestCaseName(const testing::TestParamInfo<TopKGPUParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    int64_t keepK, axis;
    ov::op::TopKMode mode;
    ov::op::TopKSortType sort;
    bool stable;
    std::tie(keepK, axis, mode, sort, stable, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
        obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "k=" << keepK << "_";
    result << "axis=" << axis << "_";
    result << "mode=" << mode << "_";
    result << "sort=" << sort << "_";
    result << "stable=" << stable << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void TopKLayerTestGPU::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    int64_t keepK, axis;
    ov::op::TopKMode mode;
    ov::op::TopKSortType sort;
    bool stable;
    std::tie(keepK, axis, mode, sort, stable, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
        this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto k = std::make_shared<ov::op::v0::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
    auto topk = std::dynamic_pointer_cast<ov::op::v11::TopK>(
        std::make_shared<ov::op::v11::TopK>(params[0], k, axis, mode, sort, ngraph::element::Type_t::i64, stable));

    ngraph::ResultVector results;
    for (size_t i = 0; i < topk->get_output_size(); i++) {
        results.push_back(std::make_shared<ov::op::v0::Result>(topk->output(i)));
    }
    function = std::make_shared<ngraph::Function>(results, params, "TopK");
}

InferenceEngine::Blob::Ptr TopKLayerTestGPU::GenerateInput(const InferenceEngine::InputInfo& info) const {
    IE_ASSERT(InferenceEngine::Precision::FP32 == info.getTensorDesc().getPrecision() ||
              InferenceEngine::Precision::BF16 == info.getTensorDesc().getPrecision() ||
              InferenceEngine::Precision::FP16 == info.getTensorDesc().getPrecision());

    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    int64_t keepK, axis;
    ov::op::TopKMode mode;
    ov::op::TopKSortType sort;
    bool stable;
    std::tie(keepK, axis, mode, sort, stable, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
        this->GetParam();

    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    // For unstable sorting, generate unrepeated input data.
    // While for stable sorting repeating values are explicitly set.

    size_t size = blob->size();
    int start = -static_cast<int>(size / 2);
    std::vector<int> data(size);
    size_t set_size = sort == ov::op::TopKSortType::SORT_VALUES && stable ? size / 2 : size;
    std::iota(data.begin(), data.begin() + set_size, start);
    if (sort == ov::op::TopKSortType::SORT_VALUES && stable) {
        std::copy(data.begin(), data.begin() + set_size, data.begin() + set_size);
    }
    std::mt19937 gen(0);
    std::shuffle(data.begin(), data.end(), gen);

    float divisor = size / 10.0;
    if (InferenceEngine::Precision::FP32 == info.getTensorDesc().getPrecision()) {
        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        for (size_t i = 0; i < size; i++) {
            rawBlobDataPtr[i] = static_cast<float>(data[i] / divisor);
        }
    } else if (InferenceEngine::Precision::BF16 == info.getTensorDesc().getPrecision()) {
        auto* rawBlobDataPtr = blob->buffer().as<ngraph::bfloat16*>();
        for (size_t i = 0; i < size; i++) {
            rawBlobDataPtr[i] = static_cast<ngraph::bfloat16>(data[i] / divisor);
        }
    } else if (InferenceEngine::Precision::FP16 == info.getTensorDesc().getPrecision()) {
        auto* rawBlobDataPtr = blob->buffer().as<ngraph::float16*>();
        for (size_t i = 0; i < size; i++) {
            rawBlobDataPtr[i] = static_cast<ngraph::float16>(data[i] / divisor);
        }
    }

    return blob;
}

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<int64_t> axes = {
    0,
    1,
    2,
};

const std::vector<int64_t> k = {
    1,
    5,
    10,
};

const std::vector<ov::op::TopKMode> modes = {
    ov::op::TopKMode::MIN,
    ov::op::TopKMode::MAX,
};

const std::vector<ov::op::TopKSortType> sortTypes = {
    ov::op::TopKSortType::SORT_INDICES,
    ov::op::TopKSortType::SORT_VALUES,
};

const std::vector<bool> stable = {
    false,
    true,
};

TEST_P(TopKLayerTestGPU, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_TopK,
                         TopKLayerTestGPU,
                         ::testing::Combine(::testing::ValuesIn(k),
                                            ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(stable),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({10, 10, 10})),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         TopKLayerTestGPU::getTestCaseName);
}  // namespace
}  // namespace GPULayerTestsDefinitions
