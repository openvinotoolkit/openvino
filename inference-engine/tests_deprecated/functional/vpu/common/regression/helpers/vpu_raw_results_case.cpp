// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_raw_results_case.hpp"

#include <format_reader_ptr.h>
#include <fstream>

std::vector <float> operator + (std::vector <float> && l, const std::vector <float> & r) {
    l.insert(l.end(), r.begin(), r.end());

    return std::move(l);
}

//------------------------------------------------------------------------------
// Implementation of public methods of class VpuNoRawResultsRegression
//------------------------------------------------------------------------------

std::string VpuNoRawResultsRegression::getTestCaseName(
        TestParamInfo<RawResultsTestVpuParam ::ParamType> param) {

    return VpuNoRegressionBase::getTestCaseName(get<0>(param.param),
                                                get<1>(param.param),
                                                get<2>(param.param),
                                                get<3>(param.param));
}


std::vector<float> VpuNoRawResultsRegression::fromBinaryFile(std::string inputTensorBinary) {
    std::vector <float> result;
    std::ifstream in(inputTensorBinary, std::ios_base::binary | std::ios_base::ate);

    int sizeFile = in.tellg();
    in.seekg(0, std::ios_base::beg);
    size_t count = sizeFile / sizeof(float);

    if(in.good()) {
        for (size_t i = 0; i < count; i++) {
            float tmp;
            in.read(reinterpret_cast<char *>(&tmp), sizeof(float));
            result.push_back(tmp);
        }
    } else {
        IE_THROW() << "Can't open file "<< inputTensorBinary;
    }

    return result;
}


//------------------------------------------------------------------------------
// Implementation of private methods of class VpuNoRawResultsRegression
//------------------------------------------------------------------------------

void  VpuNoRawResultsRegression::SetUp() {
    TestsCommon::SetUp();

    plugin_name_ = get<0>(RawResultsTestVpuParam::GetParam()).first;
    device_name_ = get<0>(RawResultsTestVpuParam::GetParam()).second;
    in_precision_= get<1>(RawResultsTestVpuParam::GetParam());
    batch_= get<2>(RawResultsTestVpuParam::GetParam());
    do_reshape_= get<3>(RawResultsTestVpuParam::GetParam());

    InitConfig();
}

void VpuNoRawResultsRegression::InitConfig() {
    VpuNoRegressionBase::InitConfig();
}

bool VpuNoRawResultsRegression::loadImage(const std::string &imageFilename, const InferenceEngine::Blob::Ptr &blob,
    bool bgr, InferenceEngine::Layout layout) {

    auto precision = blob->getTensorDesc().getPrecision();
    if (precision != InferenceEngine::Precision::FP16
        && precision != InferenceEngine::Precision::FP32
        && precision != InferenceEngine::Precision::U8) {
        std::cout << "loadImage error: Input must have U8, FP16, FP32 precision" << std::endl;
        return false;
    }

    if ((layout != InferenceEngine::Layout::NCHW) && (layout != InferenceEngine::Layout::NHWC)) {
        std::cout << "Support only two layouts NCHW and NHWC" << std::endl;
        return false;
    }

    FormatReader::ReaderPtr reader(imageFilename.c_str());
    if (reader.get() == nullptr) {
        std::cout << "loadImage error: image " << imageFilename << " cannot be read!" << std::endl;
        return false;
    }

    size_t w = blob->getTensorDesc().getDims()[3];
    size_t h = blob->getTensorDesc().getDims()[2];
    if (reader->width() != w || reader->height() != h) {
        std::cout << "loadImage error: Input sizes mismatch, got " << reader->width() << "x" << reader->height()
                  << " expecting " << w << "x" << h << std::endl;
        return false;
    }

    auto numBlobChannels = blob->getTensorDesc().getDims()[1];
    size_t numImageChannels = reader->size() / (reader->width() * reader->height());
    if (numBlobChannels != numImageChannels && numBlobChannels != 1) {
        std::cout << "loadImage error: Input channels mismatch: image channels " << numImageChannels << ", "
                  << "network channels " << numBlobChannels << ", expecting count of image channels are equal "
                  << "to count if network channels or count of network channels are equal to 1" << std::endl;
        return false;
    }

    auto nPixels = w * h;
    uint8_t *BGR8 = reader->getData().get();
    for (unsigned int i = 0; i < nPixels; i++) {
        for (unsigned int j = 0; j < numBlobChannels; j++) {
            uint8_t val = bgr ? BGR8[i * numImageChannels + j] : BGR8[i * numBlobChannels + (numBlobChannels - j - 1)];
            size_t index = (layout == InferenceEngine::Layout::NCHW) ? (i + j * nPixels) : (i * numBlobChannels + j) ;

            switch (blob->getTensorDesc().getPrecision()){
                case Precision::U8:
                {
                    uint8_t *inputDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<uint8_t>>(blob)->data();
                    inputDataPtr[index] = val;
                    break;
                }
                case Precision::FP16:
                {
                    // for fp16 unsigned short used see PrecisionTrait<Precision::FP16>::value_type see cldnn plugin for details
                    auto buf = blob->buffer();
                    ie_fp16 *inputDataPtr = buf.as<ie_fp16*>();
                    inputDataPtr[index] = PrecisionUtils::f32tof16(static_cast<float>(val));
                    break;
                }
                case Precision::FP32:
                {
                    float *inputDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<float>>(blob)->data();
                    inputDataPtr[index] = static_cast<float>(val);
                    break;
                }
                default:
                    IE_THROW() << "Unsupported precision!";
            }
        }
    }
    return true;
}

bool VpuNoRawResultsRegression::generateSeqIndLPR(InferenceEngine::Blob::Ptr &seq_ind) {

    if (seq_ind->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP16
        && seq_ind->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32
        && seq_ind->getTensorDesc().getPrecision() != InferenceEngine::Precision::U8) {
        std::cout << "generateSeqIndLPR error: Input must have U8, FP16, FP32 precision" << std::endl;
        return false;
    }

    switch (seq_ind->getTensorDesc().getPrecision()){
        case Precision::U8:
        {
            uint8_t *input_data = seq_ind->buffer().as<uint8_t*>();
            input_data[0] = 0;
            for (size_t i = 1; i < seq_ind->size(); i++) {
                input_data[i] = 1 ;
            }
            break;
        }
        case Precision::FP16:
        {
            ie_fp16 *input_data = seq_ind->buffer().as<ie_fp16*>();
            input_data[0] = PrecisionUtils::f32tof16(0.0);
            for (size_t i = 1; i < seq_ind->size(); i++) {
                input_data[i] = PrecisionUtils::f32tof16(1.0) ;
            }
            break;
        }
        case Precision::FP32:
        {
            float *input_data = seq_ind->buffer().as<float*>();
            input_data[0] = 0.0;
            for (size_t i = 1; i < seq_ind->size(); i++) {
                input_data[i] = 1.0;
            }
            break;
        }
        default:
            IE_THROW() << "Unsupported precision!";
    }

    return true;
}

bool VpuNoRawResultsRegression::loadTensorDistance(InferenceEngine::Blob::Ptr blob1, const std::vector<float> &input1) {
    
    auto blob_precision = blob1->getTensorDesc().getPrecision();

    if (blob_precision != InferenceEngine::Precision::FP16
        && blob_precision != InferenceEngine::Precision::FP32) {
        std::cout << "loadTensorDistance error: Input must have FP16, FP32 precision" << std::endl;
        return false;
    }
    size_t sizeBlob;
    sizeBlob = blob1->size() / blob1->getTensorDesc().getDims()[0];

    if (sizeBlob > input1.size()) {
        std::cout << "Blobs must have same sizes with inputs" << std::endl;
        return false;
    }

    switch (blob_precision){
        case Precision::FP16:
        {
            auto buf1 = blob1->buffer();
            ie_fp16 *inputDataPtr1 = buf1.as<ie_fp16*>();
            for(size_t i = 0; i < sizeBlob; i++) {
                inputDataPtr1[i] = PrecisionUtils::f32tof16(input1[i]);
            }

            break;
        }
        case Precision::FP32:
        {
            float *inputDataPtr1 = std::dynamic_pointer_cast<InferenceEngine::TBlob<float>>(blob1)->data();

            for(size_t i = 0; i < sizeBlob; i++) {
                inputDataPtr1[i] = input1[i];
            }

            break;
        }
        default:
            IE_THROW() << "Unsupported precision!";
    }

    return true;
}
