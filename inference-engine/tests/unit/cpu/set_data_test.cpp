// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_memory.h"
#include "mkldnn_debug.h"
#include "mkldnn_extension_utils.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace MKLDNNPlugin;

using setDataTestParamsSet = std::tuple<mkldnn::memory::data_type, // input data type
                                        mkldnn::memory::data_type, // output data type
                                        mkldnn::memory::format,    // input memory foramt
                                        mkldnn::memory::format,    // output memory foramt
                                        std::vector<int16_t>>;     // reference

class SetDataTest : virtual public ::testing::Test, public testing::WithParamInterface<setDataTestParamsSet> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<setDataTestParamsSet> obj) {
        mkldnn::memory::data_type inDataType, outDataType;
        mkldnn::memory::format inFormat, outFormat;
        std::vector<int16_t> reference;
        std::tie(inDataType, outDataType, inFormat, outFormat, reference) = obj.param;

        std::ostringstream result;

        result << "Input=" << mkldnn_dt2str(static_cast<mkldnn_data_type_t>(inDataType)) << "_"
                            << mkldnn_fmt2str(static_cast<mkldnn_memory_format_t>(inFormat)) << "__";
        result << "Output=" << mkldnn_dt2str(static_cast<mkldnn_data_type_t>(outDataType)) << "_"
                            << mkldnn_fmt2str(static_cast<mkldnn_memory_format_t>(outFormat));

        return result.str();
    }

protected:
    void SetUp() override {
        mkldnn::memory::data_type inDataType, outDataType;
        mkldnn::memory::format inFormat, outFormat;
        std::vector<int16_t> reference;
        std::tie(inDataType, outDataType, inFormat, outFormat, reference) = this->GetParam();
        bool copyRef = inFormat == outFormat;

        MKLDNNDims dims(InferenceEngine::SizeVector{1, 3, 1, 2});

        const void *inData = nullptr;
        if (inDataType == mkldnn::memory::data_type::s8) {
            int8_t *data = new int8_t[dims.size()];
            for (size_t i = 0; i < dims.size(); i++)
                data[i] = i;
            inData = data;
        } else if (inDataType == mkldnn::memory::data_type::s16) {
            int16_t *data = new int16_t[dims.size()];
            for (size_t i = 0; i < dims.size(); i++)
                data[i] = i;
            inData = data;
        } else {
            throw std::runtime_error("Unsupported input data type!");
        }

        MKLDNNMemoryDesc srcDesc(dims, inDataType, inFormat);

        mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));
        MKLDNNMemory dst(eng);
        MKLDNNMemoryDesc dstDesc(dims, outDataType, outFormat);

        const void *outData = nullptr, *ref = nullptr;
        if (outDataType == mkldnn::memory::data_type::s8) {
            int8_t *data = new int8_t[dims.size()];
            outData = data;

            if (copyRef) {
                int8_t *refPtr = new int8_t[dims.size()];
                for (size_t i = 0; i < dims.size(); i++)
                    refPtr[i] = i;
                ref = refPtr;
            }
        } else if (outDataType == mkldnn::memory::data_type::s16) {
            int16_t *data = new int16_t[dims.size()];
            outData = data;

            if (copyRef) {
                int16_t *refPtr = new int16_t[dims.size()];
                for (size_t i = 0; i < dims.size(); i++)
                    refPtr[i] = i;
                ref = refPtr;
            }
        } else {
            throw std::runtime_error("Unsupported output data type!");
        }

        dst.Create(dstDesc, outData);
        dst.SetData(srcDesc, inData, dims.size() * MKLDNNExtensionUtils::sizeOfDataType(inDataType));
        ASSERT_EQ(reference.size(), dst.GetElementsCount());

        if (outDataType == mkldnn::memory::data_type::s8) {
            int8_t *ptr = reinterpret_cast<int8_t *>(dst.GetData());
            if (copyRef) {
                const int8_t *refPtr = reinterpret_cast<const int8_t *>(ref);
                for (size_t i = 0; i < dst.GetElementsCount(); i++) {
                    ASSERT_EQ(refPtr[i], ptr[i]);
                }
            } else {
                for (size_t i = 0; i < dst.GetElementsCount(); i++) {
                    ASSERT_EQ(reference[i], ptr[i]);
                }
            }
            delete [] reinterpret_cast<const int8_t *>(outData);
            if (copyRef)
                delete [] reinterpret_cast<const int8_t *>(ref);
        } else if (outDataType == mkldnn::memory::data_type::s16) {
            int16_t *ptr = reinterpret_cast<int16_t *>(dst.GetData());
            if (copyRef) {
                const int16_t *refPtr = reinterpret_cast<const int16_t *>(ref);
                for (size_t i = 0; i < dst.GetElementsCount(); i++) {
                    ASSERT_EQ(refPtr[i], ptr[i]);
                }
            } else {
                for (size_t i = 0; i < dst.GetElementsCount(); i++) {
                    ASSERT_EQ(reference[i], ptr[i]);
                }
            }
            delete [] reinterpret_cast<const int16_t *>(outData);
            if (copyRef)
                delete [] reinterpret_cast<const int16_t *>(ref);
        }

        if (inDataType == mkldnn::memory::data_type::s8) {
            delete [] reinterpret_cast<const int8_t *>(inData);
        } else if (inDataType == mkldnn::memory::data_type::s16) {
            delete [] reinterpret_cast<const int16_t *>(inData);
        }
    }
};

TEST_P(SetDataTest, CompareWithRefs) {}

const std::vector<mkldnn::memory::data_type> dataTypes = {mkldnn::memory::data_type::s8, mkldnn::memory::data_type::s16};
const std::vector<mkldnn::memory::format> memFormats = {mkldnn::memory::format::nhwc, mkldnn::memory::format::nchw};
std::vector<int16_t> reference{0, 2, 4, 1, 3, 5};

auto params = ::testing::Combine(::testing::ValuesIn(dataTypes),
                                 ::testing::ValuesIn(dataTypes),
                                 ::testing::ValuesIn(memFormats),
                                 ::testing::Values(memFormats[0]),
                                 ::testing::Values(reference));
INSTANTIATE_TEST_CASE_P(smoke_SetDataTest, SetDataTest, params, SetDataTest::getTestCaseName);
