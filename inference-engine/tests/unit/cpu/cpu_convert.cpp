// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ngraph/type/float16.hpp>
#include <vector>
#include "nodes/common/cpu_convert.h"

using namespace InferenceEngine;
using namespace ngraph;

using cpuConvertTestParamsSet = std::tuple<Precision,  // input precision
                                           Precision>; // target precision

class CpuConvertTest : virtual public ::testing::Test, public testing::WithParamInterface<cpuConvertTestParamsSet> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<cpuConvertTestParamsSet> obj) {
        Precision inputPrecision, targetPrecision;
        std::tie(inputPrecision, targetPrecision) = obj.param;

        std::ostringstream result;

        result << "InputPrecision=" << inputPrecision << "_";
        result << "TargetPrecision=" << targetPrecision;

        return result.str();
    }

protected:
    void SetUp() override {
        const size_t size = 100;
        Precision inputPrecision, targetPrecision;
        std::tie(inputPrecision, targetPrecision) = this->GetParam();

        std::vector<uint8_t> srcData(size * inputPrecision.size());
        std::vector<uint8_t> dstData(size * targetPrecision.size());
        std::vector<uint8_t> refData(size * targetPrecision.size());
        fillBuffer(inputPrecision, size, srcData.data());
        fillBuffer(targetPrecision, size, refData.data());

        cpu_convert(srcData.data(), dstData.data(), inputPrecision, targetPrecision, size);

        ASSERT_TRUE(compare(targetPrecision, size, dstData.data(), refData.data()));
    }

private:
    template<class Type>
    void fill(const size_t size, void *ptr) {
        Type *castPtr = reinterpret_cast<Type *>(ptr);
        for (size_t i = 0; i < size; i++)
            castPtr[i] = static_cast<Type>(i);
    }

    void fillBuffer(Precision precision, const size_t size, void *ptr) {
        switch (precision) {
            case Precision::U8:
                fill<uint8_t>(size, ptr);
                break;
            case Precision::I8:
                fill<int8_t>(size, ptr);
                break;
            case Precision::U16:
                fill<uint16_t>(size, ptr);
                break;
            case Precision::I16:
                fill<int16_t>(size, ptr);
                break;
            case Precision::I32:
                fill<int32_t>(size, ptr);
                break;
            case Precision::U64:
                fill<uint64_t>(size, ptr);
                break;
            case Precision::I64:
                fill<int64_t>(size, ptr);
                break;
            case Precision::FP16:
                fill<float16>(size, ptr);
                break;
            case Precision::FP32:
                fill<float>(size, ptr);
                break;
            default:
                std::string error = "Can't fill buffer with " + std::string(precision.name()) + " precision";
                throw std::runtime_error(error);
        }
    }

    template<class Type>
    bool compare(const size_t size, void *ptr1, void *ptr2) {
        Type *castPtr1 = reinterpret_cast<Type *>(ptr1);
        Type *castPtr2 = reinterpret_cast<Type *>(ptr2);
        for (size_t i = 0; i < size; i++) {
            if (abs(castPtr1[i] - castPtr2[i]) > static_cast<Type>(0.001))
                return false;
        }
        return true;
    }

    bool compare(Precision precision, const size_t size, void *ptr1, void *ptr2) {
        switch (precision) {
            case Precision::U8:
                return compare<uint8_t>(size, ptr1, ptr2);
            case Precision::I8:
                return compare<int8_t>(size, ptr1, ptr2);
            case Precision::U16:
                return compare<uint16_t>(size, ptr1, ptr2);
            case Precision::I16:
                return compare<int16_t>(size, ptr1, ptr2);
            case Precision::I32:
                return compare<int32_t>(size, ptr1, ptr2);
            case Precision::U64:
                return compare<uint16_t>(size, ptr1, ptr2);
            case Precision::I64:
                return compare<int16_t>(size, ptr1, ptr2);
            case Precision::FP16:
                return compare<float16>(size, ptr1, ptr2);
            case Precision::FP32:
                return compare<float>(size, ptr1, ptr2);
            default:
                std::string error = "Can't compare buffer with " + std::string(precision.name()) + " precision";
                throw std::runtime_error(error);
        }
        return true;
    }
};

TEST_P(CpuConvertTest, CompareWithRefs) {}

const std::vector<Precision> precisions = {
        Precision::U8,
        Precision::I8,
        Precision::U16,
        Precision::I16,
        Precision::I32,
        Precision::U64,
        Precision::I64,
        Precision::FP32,
        Precision::FP16
};

INSTANTIATE_TEST_CASE_P(smoke_Check, CpuConvertTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(precisions),
                            ::testing::ValuesIn(precisions)), CpuConvertTest::getTestCaseName);
