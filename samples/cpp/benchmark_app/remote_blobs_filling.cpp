// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "remote_blobs_filling.hpp"
// clang-format on

namespace gpu {

template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
void fillBufferRandom(void* inputBuffer,
                      size_t elementsNum,
                      T rand_min = std::numeric_limits<uint8_t>::min(),
                      T rand_max = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(0);
    uniformDistribution<T2> distribution(rand_min, rand_max);
    auto inputBufferData = static_cast<T*>(inputBuffer);
    for (size_t i = 0; i < elementsNum; i++) {
        inputBufferData[i] = static_cast<T>(distribution(gen));
    }
}

void fillBuffer(void* inputBuffer, size_t elementsNum, InferenceEngine::Precision precision) {
    if (precision == InferenceEngine::Precision::FP32) {
        fillBufferRandom<float, float>(inputBuffer, elementsNum);
    } else if (precision == InferenceEngine::Precision::FP16) {
        fillBufferRandom<short, short>(inputBuffer, elementsNum);
    } else if (precision == InferenceEngine::Precision::I32) {
        fillBufferRandom<int32_t, int32_t>(inputBuffer, elementsNum);
    } else if (precision == InferenceEngine::Precision::I64) {
        fillBufferRandom<int64_t, int64_t>(inputBuffer, elementsNum);
    } else if (precision == InferenceEngine::Precision::U8) {
        // uniform_int_distribution<uint8_t> is not allowed in the C++17
        // standard and vs2017/19
        fillBufferRandom<uint8_t, uint32_t>(inputBuffer, elementsNum);
    } else if (precision == InferenceEngine::Precision::I8) {
        // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
        // and vs2017/19
        fillBufferRandom<int8_t, int32_t>(inputBuffer, elementsNum);
    } else if (precision == InferenceEngine::Precision::U16) {
        fillBufferRandom<uint16_t, uint16_t>(inputBuffer, elementsNum);
    } else if (precision == InferenceEngine::Precision::I16) {
        fillBufferRandom<int16_t, int16_t>(inputBuffer, elementsNum);
    } else if (precision == InferenceEngine::Precision::BOOL) {
        fillBufferRandom<uint8_t, uint32_t>(inputBuffer, elementsNum, 0, 1);
    } else {
        IE_THROW() << "Requested precision is not supported";
    }
}

size_t getBytesPerElement(InferenceEngine::Precision precision) {
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        return 4;
    case InferenceEngine::Precision::FP16:
        return 2;
    case InferenceEngine::Precision::I32:
        return 4;
    case InferenceEngine::Precision::I64:
        return 8;
    case InferenceEngine::Precision::U8:
        return 1;
    case InferenceEngine::Precision::I8:
        return 1;
    case InferenceEngine::Precision::U16:
        return 2;
    case InferenceEngine::Precision::I16:
        return 2;
    case InferenceEngine::Precision::BOOL:
        return 1;
    default:
        IE_THROW() << "Requested precision is not supported";
    }
}

void fillRemoteBlobs(const std::vector<std::string>& inputFiles,
                     const size_t& batchSize,
                     benchmark_app::InputsInfo& app_inputs_info,
                     std::vector<InferReqWrap::Ptr> requests,
                     const InferenceEngine::ExecutableNetwork& exeNetwork) {
#ifdef HAVE_DEVICE_MEM_SUPPORT
    slog::info << "Device memory will be used for input and output blobs" << slog::endl;
    if (inputFiles.size()) {
        slog::warn << "Device memory supports only random data at this moment, input images will be ignored"
                   << slog::endl;
    }
    auto context = exeNetwork.GetContext();
    auto oclContext = std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(context)->get();
    auto oclInstance = std::make_shared<OpenCL>(oclContext);

    auto setShared = [&](size_t requestId,
                         const std::string name,
                         const InferenceEngine::TensorDesc& desc,
                         bool fillRandom = false) {
        cl_int err;
        auto inputDims = desc.getDims();
        auto elementsNum = std::accumulate(begin(inputDims), end(inputDims), 1, std::multiplies<size_t>());
        auto inputSize = elementsNum * getBytesPerElement(desc.getPrecision());

        cl::Buffer sharedBuffer =
            cl::Buffer(oclInstance->_context, CL_MEM_READ_WRITE, (cl::size_type)inputSize, NULL, &err);

        if (fillRandom) {
            void* mappedPtr = oclInstance->_queue.enqueueMapBuffer(sharedBuffer,
                                                                   CL_TRUE,
                                                                   CL_MEM_READ_WRITE,
                                                                   0,
                                                                   (cl::size_type)inputSize);
            fillBuffer(mappedPtr, elementsNum, desc.getPrecision());
            oclInstance->_queue.enqueueUnmapMemObject(sharedBuffer, mappedPtr);
        }

        InferenceEngine::Blob::Ptr sharedBlob = InferenceEngine::gpu::make_shared_blob(desc, context, sharedBuffer);

        requests.at(requestId)->setBlob(name, sharedBlob);
    };

    for (size_t requestId = 0; requestId < requests.size(); requestId++) {
        for (auto& item : exeNetwork.GetInputsInfo())
            setShared(requestId, item.first, item.second->getTensorDesc(), true);

        for (auto& item : exeNetwork.GetOutputsInfo())
            setShared(requestId, item.first, item.second->getTensorDesc());
    }
#else
    IE_THROW() << "Device memory requested for GPU device, but OpenCL was not linked";
#endif
}

}  // namespace gpu
