// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/include/functional_test_utils/blob_utils.hpp>
#include "ngraph_test_utils.hpp"
#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

template<class T_IE, class T_NGRAPH>
static void Compare(const T_NGRAPH *expected, const T_IE *actual, std::size_t size, float threshold, float abs_threshold = -1.f) {
    for (std::size_t i = 0; i < size; ++i) {
        const T_NGRAPH &ref = expected[i];
        const auto &res = actual[i];
        const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
        if (abs_threshold > 0.f && absoluteDifference > abs_threshold) {
            IE_THROW() << "Absolute comparison of values expected: " << std::to_string(ref) << " and actual: " << std::to_string(res)
                       << " at index " << i << " with absolute threshold " << abs_threshold
                       << " failed";
        }
        if (absoluteDifference <= threshold) {
            continue;
        }
        double max;
        if (sizeof(T_IE) < sizeof(T_NGRAPH)) {
            max = std::max(CommonTestUtils::ie_abs(T_NGRAPH(res)), CommonTestUtils::ie_abs(ref));
        } else {
            max = std::max(CommonTestUtils::ie_abs(res), CommonTestUtils::ie_abs(T_IE(ref)));
        }
        double diff = static_cast<float>(absoluteDifference) / max;
        if (max == 0 || (diff > static_cast<float>(threshold)) ||
            (std::isnan(static_cast<float>(res)) ^ std::isnan(static_cast<float>(ref)))) {
            IE_THROW() << "Relative comparison of values expected: " << std::to_string(ref) << " and actual: " << std::to_string(res)
                       << " at index " << i << " with threshold " << threshold
                       << " failed";
        }
    }
}

template <typename T_IE>
void callCompare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                 const T_IE* actualBuffer, size_t size, float threshold, float abs_threshold) {
    auto expectedBuffer = expected.second.data();
    switch (expected.first) {
        case ngraph::element::Type_t::i64:
            Compare<T_IE, int64_t>(reinterpret_cast<const int64_t *>(expectedBuffer),
                                                         actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::i32:
            Compare<T_IE, int32_t>(reinterpret_cast<const int32_t *>(expectedBuffer),
                                                         actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::i16:
            Compare<T_IE, int16_t>(reinterpret_cast<const int16_t *>(expectedBuffer),
                                                         actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::i8:
            Compare<T_IE, int8_t>(reinterpret_cast<const int8_t *>(expectedBuffer),
                                                        actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::u64:
            Compare<T_IE, uint64_t>(reinterpret_cast<const uint64_t *>(expectedBuffer),
                                                          actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::u32:
            Compare<T_IE, uint32_t>(reinterpret_cast<const uint32_t *>(expectedBuffer),
                                                          actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::u16:
            Compare<T_IE, uint16_t>(reinterpret_cast<const uint16_t *>(expectedBuffer),
                                                          actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::boolean:
        case ngraph::element::Type_t::u8:
            Compare<T_IE, uint8_t>(reinterpret_cast<const uint8_t *>(expectedBuffer),
                                                         actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::f64:
            Compare<T_IE, double>(reinterpret_cast<const double *>(expectedBuffer),
                                                        actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::f32:
            Compare<T_IE, float>(reinterpret_cast<const float *>(expectedBuffer),
                                                       actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::f16:
            Compare<T_IE, ngraph::float16>(reinterpret_cast<const ngraph::float16 *>(expectedBuffer),
                                                                 actualBuffer, size, threshold, abs_threshold);
            break;
        case ngraph::element::Type_t::bf16:
            Compare<T_IE, ngraph::bfloat16>(reinterpret_cast<const ngraph::bfloat16 *>(expectedBuffer),
                                                                  actualBuffer, size, threshold, abs_threshold);
            break;
//        case ngraph::element::Type_t::i4: {
//            auto expectedOut = ngraph::helpers::convertOutputPrecision(
//                    expected.second,
//                    expected.first,
//                    ngraph::element::Type_t::i8,
//                    size);
//            Compare<T_IE, int8_t>(reinterpret_cast<const int8_t *>(expectedOut.data()),
//                                                        actualBuffer, size, threshold, abs_threshold);
//            break;
//        }
//        case ngraph::element::Type_t::u4: {
//            auto expectedOut = ngraph::helpers::convertOutputPrecision(
//                    expected.second,
//                    expected.first,
//                    ngraph::element::Type_t::u8,
//                    size);
//            Compare<T_IE, uint8_t>(reinterpret_cast<const uint8_t *>(expectedOut.data()),
//                                                         actualBuffer, size, threshold, abs_threshold);
//            break;
//        }
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::undefined:
            Compare<T_IE, T_IE>(reinterpret_cast<const T_IE *>(expectedBuffer), actualBuffer, size, threshold, abs_threshold);
            break;
        default: FAIL() << "Comparator for " << expected.first << " precision isn't supported";
    }
    return;
}


void Compare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                    const InferenceEngine::Blob::Ptr &actual,
                    float threshold,
                    float abs_threshold) {
    const auto &precision = actual->getTensorDesc().getPrecision();
    auto k =  static_cast<float>(expected.first.size()) / precision.size();
    // W/A for int4, uint4
    if (expected.first == ngraph::element::Type_t::u4 || expected.first == ngraph::element::Type_t::i4) {
        k /= 2;
    } else if (expected.first == ngraph::element::Type_t::undefined || expected.first == ngraph::element::Type_t::dynamic) {
        k = 1;
    }
    ASSERT_EQ(expected.second.size(), actual->byteSize() * k);

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

    const auto &size = actual->size();
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            callCompare<float>(expected, reinterpret_cast<const float *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::I32:
            callCompare<int32_t>(expected, reinterpret_cast<const int32_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::I64:
            callCompare<int64_t>(expected, reinterpret_cast<const int64_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::I8:
            callCompare<int8_t>(expected, reinterpret_cast<const int8_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::U16:
            callCompare<uint16_t>(expected, reinterpret_cast<const uint16_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::I16:
            callCompare<int16_t>(expected, reinterpret_cast<const int16_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::BOOL:
        case InferenceEngine::Precision::U8:
            callCompare<uint8_t>(expected, reinterpret_cast<const uint8_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::U64:
            callCompare<uint64_t>(expected, reinterpret_cast<const uint64_t *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::BF16:
            callCompare<ngraph::bfloat16>(expected, reinterpret_cast<const ngraph::bfloat16 *>(actualBuffer), size, threshold, abs_threshold);
            break;
        case InferenceEngine::Precision::FP16:
            callCompare<ngraph::float16>(expected, reinterpret_cast<const ngraph::float16 *>(actualBuffer), size, threshold, abs_threshold);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                    const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs,
                    float threshold, float abs_threshold) {
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto &expected = expectedOutputs[outputIndex];
        const auto &actual = actualOutputs[outputIndex];
        Compare(expected, actual, threshold, abs_threshold);
    }
}
} // namespace

void TransformationTestsF::accuracy_check(std::shared_ptr<ov::Model> ref_function,
                                          std::shared_ptr<ov::Model> cur_function) {
    try {
        if (ref_function->is_dynamic() || cur_function->is_dynamic()) {
            return;
        }
        std::vector<std::vector<uint8_t>> input_data;
        ngraph::element::TypeVector types;
        for (auto param : ref_function->get_parameters()) {
            types.push_back(param->get_element_type());

            auto layout = InferenceEngine::Layout::ANY;
            if (ov::is_scalar(param->get_shape())) {
                layout = InferenceEngine::Layout::SCALAR;
            }
            InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, param->get_shape(), layout);
            const auto &input = FuncTestUtils::createAndFillBlob(td);
            const auto &input_size = input->byteSize();

            std::vector<uint8_t> data;
            data.resize(input_size);

            auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
            IE_ASSERT(memory);

            const auto lockedMemory = memory->wmap();
            const auto buffer = lockedMemory.as<const std::uint8_t *>();
            std::copy(buffer, buffer + input_size, data.data());

            input_data.push_back(std::move(data));
        }

        auto ref_outputs = ngraph::helpers::interpreterFunction(ref_function, input_data, types);
        auto outputs = ngraph::helpers::interpreterFunction(cur_function, input_data, types);

        IE_ASSERT(ref_outputs.size() == outputs.size());

        for (size_t i = 0; i < ref_outputs.size(); ++i) {
            IE_ASSERT(ref_outputs[i].second.size() == outputs[i].second.size());
            auto * ref = reinterpret_cast<float *>(ref_outputs[i].second.data());
            auto * out = reinterpret_cast<float *>(outputs[i].second.data());
            size_t size = ref_outputs[i].second.size() / sizeof(float);
            IE_ASSERT(size > 0);
            Compare<float, float>(ref, out, size, 1e-5);
        }
    }
    catch (const std::runtime_error &re) {
        GTEST_FATAL_FAILURE_(re.what());
    } catch (const std::exception &ex) {
        GTEST_FATAL_FAILURE_(ex.what());
    } catch (...) {
        GTEST_FATAL_FAILURE_("Unknown failure occurred.");
    }
}
