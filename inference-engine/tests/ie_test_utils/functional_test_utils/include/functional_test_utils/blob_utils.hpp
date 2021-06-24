// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <type_traits>

#include <gtest/gtest.h>
#include "blob_factory.hpp"
#include "blob_transform.hpp"
#include "ie_compound_blob.h"
#include "precision_utils.h"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/test_constants.hpp"


namespace FuncTestUtils {
namespace Bf16TestUtils {
inline short reducePrecisionBitwiseS(const float in);
}  // namespace Bf16TestUtils

enum CompareType{
    ABS,
    REL,
    ABS_AND_REL  //  if absolute and relative differences are too high, an exception is thrown
};
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Pointer to considered blob
 * @param ref Pointer to reference blob
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparison
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
inline void compareRawBuffers(const dType *res, const dType *ref,
                              size_t resSize, size_t refSize,
                              CompareType compareType, float thr1 = 0.01, float thr2 = 0.01,
                              bool printData = false) {
    if (printData) {
        std::cout << "Reference results: " << std::endl;
        for (size_t i = 0; i < refSize; i++) {
            std::cout << ref[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Test results: " << std::endl;
        for (size_t i = 0; i < resSize; i++) {
            std::cout << res[i] << " ";
        }
        std::cout << std::endl;
    }

    switch (compareType) {
        case CompareType::ABS:
            for (size_t i = 0; i < refSize; i++) {
                float absDiff = std::abs(res[i] - ref[i]);
                ASSERT_LE(absDiff, thr1) << "Relative comparison of values ref: " << ref[i] << " and res: "
                                               << res[i] << " , index in blobs: " << i << " failed!";
            }
            break;
        case CompareType::REL:
            for (size_t i = 0; i < refSize; i++) {
                float absDiff = std::abs(res[i] - ref[i]);
                float relDiff = absDiff / std::max(res[i], ref[i]);
                ASSERT_LE(relDiff, thr2) << "Relative comparison of values ref: " << ref[i] << " and res: "
                                               << res[i] << " , index in blobs: " << i << " failed!";
            }
            break;
        case CompareType::ABS_AND_REL:
            for (size_t i = 0; i < refSize; i++) {
                float absDiff = std::abs(res[i] - ref[i]);
                if (absDiff > thr1) {
                    float relDiff = absDiff / std::max(res[i], ref[i]);
                    ASSERT_LE(relDiff, thr2) << "Comparison of values ref: " << ref[i] << " and res: "
                                                   << res[i] << " , index in blobs: " << i << " failed!";
                }
            }
            break;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Pointer to considered blob
 * @param ref Pointer to reference blob
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData Flag if data printing is demanded
 */
template<typename dType>
inline void compareRawBuffers(const dType *res, const dType *ref,
                              size_t resSize, size_t refSize,
                              float thr = 0.01,
                              bool printData = false) {
    compareRawBuffers(res, ref, resSize, refSize, CompareType::ABS_AND_REL, thr, thr, printData);
}
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparision
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
inline void compareRawBuffers(const std::vector<dType *> res, const std::vector<dType *> ref,
                              const std::vector<size_t> &resSizes, const std::vector<size_t> &refSizes,
                              CompareType compareType,
                              float thr1 = 0.01, float thr2 = 0.01, bool printData = false) {
    ASSERT_TRUE(res.size() == ref.size()) << "Reference and Results vector have to be same length";
    ASSERT_TRUE(res.size() == resSizes.size()) << "Results vector and elements count vector have to be same length";
    ASSERT_TRUE(ref.size() == refSizes.size()) << "Reference vector and elements count vector have to be same length";
    for (size_t i = 0; i < res.size(); i++) {
        if (printData) std::cout << "BEGIN CHECK BUFFER [" << i << "]" << std::endl;
        compareRawBuffers(res[i], ref[i], resSizes[i], refSizes[i], compareType, thr1, thr2, printData);
        if (printData) std::cout << "END CHECK BUFFER [" << i << "]" << std::endl;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
inline void compareRawBuffers(const std::vector<dType *> res, const std::vector<dType *> ref,
                              const std::vector<size_t> &resSizes, const std::vector<size_t> &refSizes,
                              float thr = 0.01, bool printData = false) {
    compareRawBuffers(res, ref, resSizes, refSizes, CompareType::ABS_AND_REL, thr, thr, printData);
}
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparision
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
inline void compareRawBuffers(const std::vector<dType *> res, const std::vector<std::shared_ptr<dType *>> ref,
                              const std::vector<size_t> &resSizes, const std::vector<size_t> &refSizes,
                              CompareType compareType,
                              float thr1 = 0.01, float thr2 = 0.01, bool printData = false) {
    ASSERT_TRUE(res.size() == ref.size()) << "Reference and Results vector have to be same length";
    ASSERT_TRUE(res.size() == resSizes.size()) << "Results vector and elements count vector have to be same length";
    ASSERT_TRUE(ref.size() == refSizes.size()) << "Reference vector and elements count vector have to be same length";
    for (size_t i = 0; i < res.size(); i++) {
        if (printData) std::cout << "BEGIN CHECK BUFFER [" << i << "]" << std::endl;
        compareRawBuffers(res[i], *ref[i], resSizes[i], refSizes[i], compareType, thr1, thr2, printData);
        if (printData) std::cout << "END CHECK BUFFER [" << i << "]" << std::endl;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
inline void compareRawBuffers(const std::vector<dType *> res, const std::vector<std::shared_ptr<dType *>> ref,
                              const std::vector<size_t> &resSizes, const std::vector<size_t> &refSizes,
                              float thr = 0.01, bool printData = false) {
    compareRawBuffers(res, ref, resSizes, refSizes, CompareType::ABS_AND_REL, thr, thr, printData);
}

template<InferenceEngine::Precision::ePrecision PRC>
inline void
compareBlobData(const InferenceEngine::Blob::Ptr &res, const InferenceEngine::Blob::Ptr &ref, float max_diff = 0.01,
                const std::string &assertDetails = "", bool printData = false) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    const dataType *res_ptr = res->cbuffer().as<dataType *>();
    size_t res_size = res->byteSize();

    const dataType *ref_ptr = ref->cbuffer().as<dataType *>();
    size_t ref_size = ref->byteSize();

    ASSERT_EQ(res_size, ref_size) << "Comparing blobs have different size. " << assertDetails;
    if (printData) {
        std::cout << "Reference results: " << std::endl;
        for (size_t i = 0; i < ref_size / sizeof(dataType); i++) {
            std::cout << ref_ptr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Test results: " << std::endl;
        for (size_t i = 0; i < res_size / sizeof(dataType); i++) {
            std::cout << res_ptr[i] << " ";
        }
        std::cout << std::endl;
    }

    for (size_t i = 0; i < ref_size / sizeof(dataType); i++) {
        auto resVal = PRC == InferenceEngine::Precision::FP16 ? InferenceEngine::PrecisionUtils::f16tof32(static_cast<InferenceEngine::ie_fp16>(res_ptr[i]))
                                                              : static_cast<float>(res_ptr[i]);
        auto refVal = PRC == InferenceEngine::Precision::FP16 ? InferenceEngine::PrecisionUtils::f16tof32(static_cast<InferenceEngine::ie_fp16>(ref_ptr[i]))
                                                              : static_cast<float>(ref_ptr[i]);
        float absDiff = std::abs(resVal - refVal);
        if (absDiff > max_diff) {
            float relDiff = absDiff / std::max(res_ptr[i], ref_ptr[i]);
            ASSERT_LE(relDiff, max_diff) << "Relative comparison of values ref: " << ref_ptr[i] << " and res: "
                                         << res_ptr[i] << " , index in blobs: " << i << " failed!" << assertDetails;
        }
    }
}


template<InferenceEngine::Precision::ePrecision PRC>
inline void
compareBlobData(const std::vector<InferenceEngine::Blob::Ptr> &res, const std::vector<InferenceEngine::Blob::Ptr> &ref,
                float max_diff = 0.01,
                const std::string &assertDetails = "", bool printData = false) {
    IE_ASSERT(res.size() == ref.size()) << "Length of comparing and references blobs vector are not equal!"
                                        << assertDetails;
    for (size_t i = 0; i < res.size(); i++) {
        if (printData)
            std::cout << "BEGIN CHECK BLOB [" << i << "]" << std::endl;
        compareBlobData<PRC>(res[i], ref[i], max_diff, assertDetails, printData);
        if (printData)
            std::cout << "END CHECK BLOB [" << i << "]" << std::endl;
    }
}

inline void
compareBlobs(const InferenceEngine::Blob::Ptr &res, const InferenceEngine::Blob::Ptr &ref, float max_diff = 0.01,
             const std::string &assertDetails = "", bool printData = false) {
    ASSERT_EQ(res->byteSize(), ref->byteSize()) << "Blobs have different byteSize(): "
                                                << res->byteSize() << " and " << ref->byteSize();

    ASSERT_EQ(res->getTensorDesc(), ref->getTensorDesc()) << "Blobs have different TensorDesc()";

    switch (res->getTensorDesc().getPrecision()) {
#define COMPARE_WITH_REF(TYPE) case TYPE: { \
                                          FuncTestUtils::compareBlobData<TYPE>(res, \
                                                                                 ref, \
                                                                                 max_diff, \
                                                                                 assertDetails, \
                                                                                 printData); break; }
        COMPARE_WITH_REF(InferenceEngine::Precision::FP32);
        COMPARE_WITH_REF(InferenceEngine::Precision::FP16);
        COMPARE_WITH_REF(InferenceEngine::Precision::I64);
#undef COMPARE_WITH_REF
        default:
            IE_THROW() << "Precision " << res->getTensorDesc().getPrecision().name()
                               << " is not covered by FuncTestUtils::compareBlobs() method";
    }
}

inline void GetComparisonThreshold(InferenceEngine::Precision prc, float &absoluteThreshold, float &relativeThreshold) {
    switch (prc) {
        case InferenceEngine::Precision::FP32:
            absoluteThreshold = relativeThreshold = 1e-4f;
            break;
        case InferenceEngine::Precision::FP16:
            absoluteThreshold = relativeThreshold = 1e-2f;
            break;
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::I8:
        case InferenceEngine::Precision::U8:
            absoluteThreshold = relativeThreshold = 1;
            break;
        default:
            IE_THROW() << "Unhandled precision " << prc << " passed to the GetComparisonThreshold()";
    }
}

inline float GetComparisonThreshold(InferenceEngine::Precision prc) {
    float res;
    GetComparisonThreshold(prc, res, res);
    return res;
}

// Copy from net_pass.h
template<InferenceEngine::Precision::ePrecision PREC_FROM, InferenceEngine::Precision::ePrecision PREC_TO>
inline void convertArrayPrecision(typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type *dst,
                                  const typename InferenceEngine::PrecisionTrait<PREC_FROM>::value_type *src,
                                  size_t nelem) {
    using dst_type = typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type;

    for (size_t i = 0; i < nelem; i++) {
        dst[i] = static_cast<dst_type>(src[i]);
    }
}

template<>
inline void
convertArrayPrecision<InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32>(float *dst, const short *src,
                                                                                          size_t nelem) {
    InferenceEngine::PrecisionUtils::f16tof32Arrays(dst, src, nelem, 1.0f, 0.0f);
}

template<>
inline void
convertArrayPrecision<InferenceEngine::Precision::BF16, InferenceEngine::Precision::FP32>(float *dst, const short *src,
                                                                                          size_t nelem) {
    auto srcBf16 = reinterpret_cast<const ngraph::bfloat16*>(src);
    for (size_t i = 0; i < nelem; i++) {
        dst[i] = static_cast<float>(srcBf16[i]);
    }
}

template<InferenceEngine::Precision::ePrecision PREC_FROM, InferenceEngine::Precision::ePrecision PREC_TO>
inline InferenceEngine::Blob::Ptr convertBlobPrecision(const InferenceEngine::Blob::Ptr &blob) {
    using from_d_type = typename InferenceEngine::PrecisionTrait<PREC_FROM>::value_type;
    using to_d_type = typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type;

    auto tensor_desc = blob->getTensorDesc();
    InferenceEngine::Blob::Ptr new_blob = InferenceEngine::make_shared_blob<to_d_type>(
            InferenceEngine::TensorDesc{PREC_TO, tensor_desc.getDims(), tensor_desc.getLayout()});
    new_blob->allocate();
    auto target = new_blob->buffer().as<to_d_type *>();
    auto source = blob->buffer().as<from_d_type *>();
    convertArrayPrecision<PREC_FROM, PREC_TO>(target, source, blob->size());
    return new_blob;
}
// Copy from net_pass.h


template<InferenceEngine::Precision::ePrecision targetPRC>
inline InferenceEngine::Blob::Ptr copyBlobWithCast(const InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::Blob::Ptr newBlob;
    switch (blob->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::FP32, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::FP16:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::FP16, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I16:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::I16, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I8:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::I8, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::U8:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::U8, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I32:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::I32, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::BOOL:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::BOOL, targetPRC>(blob);
            break;
        default:
            IE_THROW() << "Conversion from blob with precision " << blob->getTensorDesc().getPrecision().name()
                               << " not implemented yet!";
    }
    return newBlob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlobFloatNormalDistribution(const InferenceEngine::TensorDesc &td,
                                                                           const float mean,
                                                                           const float stddev,
                                                                           const int32_t seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_normal_random_float<X>(blob, mean, stddev, seed); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlobFloat(const InferenceEngine::TensorDesc &td,
        const uint32_t range = 10,
        const int32_t start_from = 0,
        const int32_t resolution = 1,
        const int32_t seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);

    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_random_float<X>(blob, range, start_from, resolution, seed); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

template<typename T>
inline InferenceEngine::Blob::Ptr createAndFillBlobWithFloatArray(const InferenceEngine::TensorDesc &td,
                                                                  const T values[],
                                                                  const int size) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_float_array<X, T>(blob, values, size); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlob(const InferenceEngine::TensorDesc &td,
        const uint32_t range = 10,
        const int32_t start_from = 0,
        const int32_t resolution = 1,
        const int seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_random<X>(blob, range, start_from, resolution, seed); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::BF16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::U64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlobConsistently(
    const InferenceEngine::TensorDesc &td,
    const uint32_t range,
    const int32_t start_from,
    const int32_t resolution) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_consistently<X>(blob, range, start_from, resolution); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
    default:
        IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr createAndFillBlobUniqueSequence(
    const InferenceEngine::TensorDesc &td,
    const int32_t start_from = 0,
    const int32_t resolution = 1,
    const int32_t seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    auto shape = td.getDims();
    auto range = std::accumulate(begin(shape), end(shape), uint64_t(1), std::multiplies<uint64_t>()) * 2;
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_random_unique_sequence<X>(blob, range, start_from, resolution, seed); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::I32)
#undef CASE
    default:
        IE_THROW() << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

inline InferenceEngine::Blob::Ptr convertBlobLayout(const InferenceEngine::Blob::Ptr& in,
                                                    InferenceEngine::Layout layout) {
    IE_ASSERT(in != nullptr) << "Got NULL pointer";

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getLayout() == layout) {
        return in;
    }

    const auto outDesc = InferenceEngine::TensorDesc(inDesc.getPrecision(), inDesc.getDims(), layout);

    const auto out = make_blob_with_precision(outDesc);
    out->allocate();

    InferenceEngine::blob_copy(in, out);

    return out;
}

template<typename dType>
inline void fillInputsBySinValues(dType* data, size_t size) {
    if (std::is_same<dType, float>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = sin(static_cast<float>(i));
        }
    } else if (std::is_same<dType, short>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(sin(static_cast<float>(i)));
        }
    }
}

template<typename dType>
inline void fillInputsByCosValues(dType* data, size_t size) {
    if (std::is_same<dType, float>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = sin(static_cast<float>(i));
        }
    } else if (std::is_same<dType, short>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(sin(static_cast<float>(i)));
        }
    }
}

inline int fillInputsBySinValues(InferenceEngine::Blob::Ptr blob) {
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        return -1;
    }
    if (mblob->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
        return -2;
    }
    auto lm = mblob->rwmap();
    fillInputsBySinValues(lm.as<float*>(), mblob->size());
    return 0;
}

inline int fillInputsByCosValues(InferenceEngine::Blob::Ptr blob) {
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        return -1;
    }
    if (mblob->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
        return -2;
    }
    auto lm = mblob->rwmap();
    fillInputsByCosValues(lm.as<float*>(), mblob->size());
    return 0;
}


namespace Bf16TestUtils {

#if defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

inline float reducePrecisionBitwise(const float in) {
    float f = in;
    int* i = reinterpret_cast<int*>(&f);
    int t2 = *i & 0xFFFF0000;
    float ft1 = *(reinterpret_cast<float*>(&t2));
    if ((*i & 0x8000) && (*i & 0x007F0000) != 0x007F0000) {
        t2 += 0x10000;
        ft1 = *(reinterpret_cast<float*>(&t2));
    }
    return ft1;
}

inline short reducePrecisionBitwiseS(const float in) {
    float f = reducePrecisionBitwise(in);
    int intf = *reinterpret_cast<int*>(&f);
    intf = intf >> 16;
    short s = intf;
    return s;
}

#if defined __GNUC__
# pragma GCC diagnostic pop
#endif

}  // namespace Bf16TestUtils

enum class BlobKind {
    Simple,
    Compound,
    BatchOfSimple
};

inline std::ostream& operator<<(std::ostream& os, BlobKind kind) {
    switch (kind) {
    case BlobKind::Simple:
        return os << "Simple";
    case BlobKind::Compound:
        return os << "Compound";
    case BlobKind::BatchOfSimple:
        return os << "BatchOfSimple";
    default:
        IE_THROW() << "Test does not support the blob kind";
  }
}

inline InferenceEngine::Blob::Ptr makeBlobOfKind(const InferenceEngine::TensorDesc& td, BlobKind blobKind) {
    using namespace ::InferenceEngine;
    switch (blobKind) {
    case BlobKind::Simple:
        return createAndFillBlob(td);
    case BlobKind::Compound:
        return make_shared_blob<CompoundBlob>(std::vector<Blob::Ptr>{});
    case BlobKind::BatchOfSimple: {
        const auto subBlobsNum = td.getDims()[0];
        auto subBlobDesc = td;
        subBlobDesc.getDims()[0] = 1;
        std::vector<Blob::Ptr> subBlobs;
        for (size_t i = 0; i < subBlobsNum; i++) {
            subBlobs.push_back(makeBlobOfKind(subBlobDesc, BlobKind::Simple));
        }
        return make_shared_blob<BatchedBlob>(subBlobs);
    }
    default:
        IE_THROW() << "Test does not support the blob kind";
    }
}

}  // namespace FuncTestUtils
