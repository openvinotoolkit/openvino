// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/ops.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/read_ir/compare_results.hpp"

namespace LayerTestsDefinitions {

namespace {
void compare(const std::shared_ptr<ngraph::Node> node,
             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expected,
             const std::vector<InferenceEngine::Blob::Ptr>& actual,
             float threshold) {
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> types(expected.size());
    auto outputs = node->outputs();
    LayerTestsUtils::LayerTestsCommon::Compare(expected, actual, threshold);
}

void compare(const std::shared_ptr<ngraph::op::v0::DetectionOutput> node,
             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expected,
             const std::vector<InferenceEngine::Blob::Ptr>& actual,
             float threshold) {
    ASSERT_EQ(expected.size(), actual.front()->byteSize());

    size_t expSize = 0;
    size_t actSize = 0;

    const auto &expectedBuffer = expected.data();
    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual.front());
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

    const float *expBuf = reinterpret_cast<const float *>(expectedBuffer);
    const float *actBuf = reinterpret_cast<const float *>(actualBuffer);
    for (size_t i = 0; i < actual.front()->size(); i+=7) {
        if (expBuf[i] == -1)
            break;
        expSize += 7;
    }
    for (size_t i = 0; i < actual.front()->size(); i+=7) {
        if (actBuf[i] == -1)
            break;
        actSize += 7;
    }
    ASSERT_EQ(expSize, actSize);
    LayerTestsUtils::LayerTestsCommon::Compare<float>(expBuf, actBuf, expSize, 1e-2f);
}

namespace Proposal {
template <class T>
void Compare(const T *expected, const T *actual, std::size_t size,
             T threshold, const std::size_t output_index, size_t& num_selected_boxes) {
    for (std::size_t i = 0; i < size; ++i) {
        const auto &ref = expected[i];
        const auto &res = actual[i];

        // verify until first -1 appears in the 1st output.
        if (output_index == 0 &&
            CommonTestUtils::ie_abs(ref - static_cast<T>(-1)) <= threshold) {
            // output0 shape = {x, 5}
            // output1 shape = {x}
            // setting the new_size for output1 verification
            num_selected_boxes = i / 5;
            return;
        }

        const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
        if (absoluteDifference <= threshold) {
            continue;
        }

        const auto max = std::max(CommonTestUtils::ie_abs(res),
                                  CommonTestUtils::ie_abs(ref));
        float diff =
                static_cast<float>(absoluteDifference) / static_cast<float>(max);
        ASSERT_TRUE(max != 0 && (diff <= static_cast<float>(threshold)))
                                    << "Relative comparison of values expected: " << ref
                                    << " and actual: " << res << " at index " << i
                                    << " with threshold " << threshold << " failed";
    }
}
} // namespace Proposal

void compare(const std::shared_ptr<ngraph::op::v4::Proposal> node,
             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
             const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs,
             float threshold) {
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0 ; outputIndex--) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];

        const auto &expectedBuffer = expected.second.data();
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const uint8_t *>();

        auto k =  static_cast<float>(expected.first.size()) / actual->getTensorDesc().getPrecision().size();
        // W/A for int4, uint4
        if (expected.first == ngraph::element::Type_t::u4 || expected.first == ngraph::element::Type_t::i4) {
            k /= 2;
        }
        if (outputIndex == 2) {
            if (expected.second.size() != k * actual->byteSize())
                throw std::runtime_error("Expected and actual size 3rd output have different size");
        }

        const auto &precision = actual->getTensorDesc().getPrecision();
        size_t size = expected.second.size() / (k * actual->getTensorDesc().getPrecision().size());
        switch (precision) {
            case InferenceEngine::Precision::FP32: {
                switch (expected.first) {
                    case ngraph::element::Type_t::f32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const float *>(expectedBuffer),
                                reinterpret_cast<const float *>(actualBuffer), size, 0);
                        break;
                    case ngraph::element::Type_t::f64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const double *>(expectedBuffer),
                                reinterpret_cast<const float *>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                }

                const auto fBuffer = lockedMemory.as<const float *>();
                for (int i = size; i < actual->size(); i++) {
                    ASSERT_TRUE(fBuffer[i] == -1.f) << "Invalid default value: " << fBuffer[i] << " at index: " << i;
                }
                break;
            }
            case InferenceEngine::Precision::I32: {
                switch (expected.first) {
                    case ngraph::element::Type_t::i32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const int32_t *>(expectedBuffer),
                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                        break;
                    case ngraph::element::Type_t::i64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const int64_t *>(expectedBuffer),
                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                }
                const auto iBuffer = lockedMemory.as<const int *>();
                for (int i = size; i < actual->size(); i++) {
                    ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
                }
                break;
            }
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }
}

void compare(const std::shared_ptr<ngraph::op::v5::NonMaxSuppression> node,
             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
             const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs,
             float threshold) {
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0 ; outputIndex--) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];

        const auto &expectedBuffer = expected.second.data();
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const uint8_t *>();

        const auto &precision = actual->getTensorDesc().getPrecision();
        size_t size = expected.second.size() / (actual->getTensorDesc().getPrecision().size());
        switch (precision) {
            case InferenceEngine::Precision::FP32: {
                switch (expected.first) {
                    case ngraph::element::Type_t::f32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const float *>(expectedBuffer),
                                reinterpret_cast<const float *>(actualBuffer), size, 0);
                        break;
                    case ngraph::element::Type_t::f64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const double *>(expectedBuffer),
                                reinterpret_cast<const float *>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                }

                const auto fBuffer = lockedMemory.as<const float *>();
                for (int i = size; i < actual->size(); i++) {
                    ASSERT_TRUE(fBuffer[i] == -1.f) << "Invalid default value: " << fBuffer[i] << " at index: " << i;
                }
                break;
            }
            case InferenceEngine::Precision::I32: {
                switch (expected.first) {
                    case ngraph::element::Type_t::i32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const int32_t *>(expectedBuffer),
                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                        break;
                    case ngraph::element::Type_t::i64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const int64_t *>(expectedBuffer),
                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                }
                const auto iBuffer = lockedMemory.as<const int *>();
                for (int i = size; i < actual->size(); i++) {
                    ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
                }
                break;
            }
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }
}

template<typename T>
void compareResults(const std::shared_ptr<ngraph::Node> node,
             const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expected,
             const std::vector<InferenceEngine::Blob::Ptr>& actual,
             float threshold) {
    return compare(ngraph::as_type_ptr<T>(node), expected, actual, threshold);
}
} // namespace

CompareMap getCompareMap() {
    CompareMap compareMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, compareResults<NAMESPACE::NAME>},
#include "ngraph/opsets/opset1_tbl.hpp"
#include "ngraph/opsets/opset2_tbl.hpp"
#include "ngraph/opsets/opset3_tbl.hpp"
#include "ngraph/opsets/opset4_tbl.hpp"
#include "ngraph/opsets/opset5_tbl.hpp"
#include "ngraph/opsets/opset6_tbl.hpp"
#undef NGRAPH_OP
    };
    return compareMap;
}

} // namespace LayerTestsDefinitions
