// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX
#include <utility>
#include <ie_blob.h>
#include <description_buffer.hpp>
#include <ie_layouts.h>
#include <precision_utils.h>
#include <blob_factory.hpp>

#include <vpu/utils/error.hpp>
#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/utils/shape_io.hpp>
#include "vpu/configuration/options/enable_receiving_tensor_time.hpp"
#include "vpu/configuration/options/perf_report_mode.hpp"
#include "vpu/configuration/options/tensor_strides.hpp"

#include "myriad_executable_network.h"
#include "myriad_infer_request.h"

using namespace vpu;
using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;

MyriadInferRequest::MyriadInferRequest(GraphDesc &graphDesc,
                                       const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                       const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                       DataInfo& compilerInputsInfo,
                                       DataInfo& compilerOutputsInfo,
                                       const std::vector<StageMetaInfo> &blobMetaData,
                                       const PluginConfiguration& myriadConfig,
                                       const Logger::Ptr &log,
                                       const MyriadExecutorPtr &executor,
                                       std::map<std::string, ie::Blob::Ptr> constDatas,
                                       bool isNetworkConstant = true) :
        IInferRequestInternal(inputs, outputs), _executor(executor),
        _log(log), _stagesMetaData(blobMetaData), _config(myriadConfig),
        _inputInfo(compilerInputsInfo), _outputInfo(compilerOutputsInfo),
        _graphDesc(graphDesc), _constDatas(constDatas), _isNetworkConstant(isNetworkConstant) {
    CreateInferRequest();
}

MyriadInferRequest::MyriadInferRequest(GraphDesc &graphDesc,
                                       InferenceEngine::InputsDataMap networkInputs,
                                       InferenceEngine::OutputsDataMap networkOutputs,
                                       DataInfo& compilerInputsInfo,
                                       DataInfo& compilerOutputsInfo,
                                       const std::vector<StageMetaInfo> &blobMetaData,
                                       const PluginConfiguration& myriadConfig,
                                       const Logger::Ptr &log,
                                       const MyriadExecutorPtr &executor,
                                       std::map<std::string, ie::Blob::Ptr> constDatas,
                                       bool isNetworkConstant = true) :
        IInferRequestInternal(networkInputs, networkOutputs), _executor(executor),
        _log(log), _stagesMetaData(blobMetaData), _config(myriadConfig),
        _inputInfo(compilerInputsInfo), _outputInfo(compilerOutputsInfo),
        _graphDesc(graphDesc), _constDatas(constDatas), _isNetworkConstant(isNetworkConstant) {
    CreateInferRequest();
}

void MyriadInferRequest::CreateInferRequest() {
    VPU_PROFILE(MyriadInferRequest);

    const auto& ioStrides = _config.get<TensorStridesOption>();
    // allocate inputs
    for (auto &networkInput : _networkInputs) {
        IE_ASSERT(ioStrides.find(networkInput.first) == ioStrides.end())
            << " input blob with strides is not supported";

        SizeVector dims      = networkInput.second->getTensorDesc().getDims();
        Precision  precision = networkInput.second->getTensorDesc().getPrecision();
        Layout     layout    = networkInput.second->getTensorDesc().getLayout();

        Blob::Ptr inputBlob = make_blob_with_precision(TensorDesc(
            precision,
            dims,
            layout));

        // allocate the input blob
        // TODO We are allocating temporary input buffer of enough size. Wrap this buffer in blobs
        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
    }
    // allocate outputs
    for (auto &networkOutput : _networkOutputs) {
        IE_ASSERT(ioStrides.find(networkOutput.first) == ioStrides.end())
            << " output blob with strides is not supported";

        SizeVector dims      = networkOutput.second->getTensorDesc().getDims();
        Precision  precision = networkOutput.second->getTensorDesc().getPrecision();
        Layout     layout    = networkOutput.second->getTensorDesc().getLayout();

        Blob::Ptr outputBlob = make_blob_with_precision(TensorDesc(
            precision,
            dims,
            layout));

        // allocate the output blob
        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
    }

    inputBuffer .resize(_inputInfo.totalSize);
    resultBuffer.resize(_outputInfo.totalSize);

    VPU_THROW_UNLESS(
        !_networkOutputs.empty() && !(_networkInputs.empty() && !_isNetworkConstant),
        "No information about network's output/input");
}

void MyriadInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

static bool needsTypeConvert(const Precision& precision) {
    switch (precision) {
        case Precision::FP16:
        case Precision::FP32:
        case Precision::I32:
            return false;
        case Precision::I64:
        case Precision::U64:
        case Precision::BOOL:
            return true;
        default:
            return false;
    }
    return false;
}

template <typename T, typename U>
static void convert(const T* const src, U* dst, size_t n) {
    std::transform(src, src + n, dst, [] (T i) -> U { return static_cast<U>(i); });
}

static const char* const NOT_ENOUGH_INPUT_SPACE_ERR_MSG = "Not enough space available in inputBuffer. Input size is too big";

static void convertInput(const uint8_t* const src, uint8_t* dst, const Precision& precision, size_t size, size_t remainingSize) {
    size_t numElements = 0;
    // U32 -> I32 is handled in copyInput by std::copy_n, unless blob layout != vpuLayout
    // then assert "Unimplemented blob transformation from precision .. to .." in ie::blob_copy fires
    switch (precision) {
        case Precision::I64:
            numElements = size / sizeof(int64_t);
            IE_ASSERT((numElements * sizeof(int32_t)) <= remainingSize) << NOT_ENOUGH_INPUT_SPACE_ERR_MSG;
            convert(reinterpret_cast<const int64_t* const>(src), reinterpret_cast<int32_t*>(dst), numElements);
            return;
        case Precision::U64:
            numElements = size / sizeof(uint64_t);
            IE_ASSERT((numElements * sizeof(int32_t)) <= remainingSize) << NOT_ENOUGH_INPUT_SPACE_ERR_MSG;
            convert(reinterpret_cast<const uint64_t* const>(src), reinterpret_cast<int32_t*>(dst), numElements);
            return;
        case Precision::BOOL:
            numElements = size / sizeof(bool);
            IE_ASSERT((numElements * sizeof(int32_t)) <= remainingSize) << NOT_ENOUGH_INPUT_SPACE_ERR_MSG;
            convert(reinterpret_cast<const bool* const>(src), reinterpret_cast<int32_t*>(dst), numElements);
            return;
        default:
            return;
    }
}

static void convertOutput(const uint8_t* src, uint8_t* dst, const Precision& precision, size_t size) {
    switch (precision) {
        case Precision::I64:
            convert(reinterpret_cast<const int32_t*>(src), reinterpret_cast<int64_t*>(dst), size / sizeof(int64_t));
            return;
        case Precision::U64:
            convert(reinterpret_cast<const int32_t*>(src), reinterpret_cast<uint64_t*>(dst), size / sizeof(uint64_t));
            return;
        case Precision::BOOL:
            convert(reinterpret_cast<const int32_t*>(src), reinterpret_cast<bool*>(dst), size / sizeof(bool));
            return;
        default:
            return;
    }
}

static void copyInput(const ie::Blob::Ptr& inputBlob, uint8_t* dst, size_t size, size_t remainingSize, ie::Layout layout, ie::Layout vpuLayout) {
    const auto& precision = inputBlob->getTensorDesc().getPrecision();
    bool needsConvert = needsTypeConvert(precision);
    if (!needsConvert) {
        IE_ASSERT(size <= remainingSize) << NOT_ENOUGH_INPUT_SPACE_ERR_MSG;
        if (layout != vpuLayout) {
            copyBlob(inputBlob, vpuLayout, dst);
        } else {
            std::copy_n(inputBlob->buffer().as<uint8_t*>(), size, dst);
        }
    } else {
        IE_ASSERT(layout == vpuLayout) << "Can't convert blob with layout not matching vpu layout";
        convertInput(inputBlob->buffer().as<uint8_t*>(), dst, precision, size, remainingSize);
    }
}

static void copyOutput(uint8_t* src, const ie::Blob::Ptr& outputBlob, const Precision& outPrec, const SizeVector& outDims, ie::Layout vpuLayout) {
    bool needsConvert = needsTypeConvert(outPrec);
    const auto layout = outputBlob->getTensorDesc().getLayout();
    if (!needsConvert) {
        if (layout != vpuLayout) {
            // TODO: TensorDesc doesn't update internal BlockingDesc and strides when setLayout is called
            const auto tempTensorDesc = ie::TensorDesc{outPrec, outDims, vpuLayout};
            const auto tmpBlob = make_blob_with_precision(tempTensorDesc, src);
            copyBlob(tmpBlob, outputBlob);
        } else {
            std::copy_n(src, outputBlob->byteSize(), outputBlob->buffer().as<uint8_t*>());
        }
    } else {
        IE_ASSERT(layout == vpuLayout) << "Can't convert blob with layout not matching vpu layout";
        convertOutput(src, outputBlob->buffer().as<uint8_t*>(), outPrec, outputBlob->byteSize());
    }
}

void MyriadInferRequest::InferAsync() {
    if (_isNetworkConstant) {
        return;
    }
    VPU_PROFILE(InferAsync);

    // execute input pre-processing
    execDataPreprocessing(_inputs, true);  // "true" stands for serial preprocessing in case of OpenMP

    auto inputInfo = _inputInfo;
    auto networkInputs = _networkInputs;

    auto getOffset = [&inputInfo] (const std::string& name) {
        const auto offsetIt = inputInfo.offset.find(name);
        IE_ASSERT(offsetIt != inputInfo.offset.end()) << "MyriadInferRequest::InferAsync()\n"
                                                    << "Input offset [" << name << "] is not provided.";
        return offsetIt->second;
    };

    auto getNetInputInfo = [&networkInputs] (const std::string& name) {
        const auto foundBlob = networkInputs.find(name);
        IE_ASSERT(foundBlob != networkInputs.end()) << "MyriadInferRequest::InferAsync()\n"
                                                    << "Input [" << name << "] is not provided.";
        return foundBlob;
    };

    for (const auto& input : _inputs) {
        const auto& name = input.first;
        const auto& blob = input.second;

        const auto offset = getOffset(name);
        const auto byteSize = blob->byteSize();
        const auto remainingSize = inputBuffer.size() - vpu::checked_cast<size_t>(offset);
        const auto foundBlob = getNetInputInfo(name);
        const auto vpuLayout = foundBlob->second->getTensorDesc().getLayout();
        const auto layout = blob->getTensorDesc().getLayout();

        copyInput(blob, &inputBuffer[offset], byteSize, remainingSize, layout, vpuLayout);

        const auto offsetShape = inputInfo.offset.find(name+"_real_shape");
        if (offsetShape == inputInfo.offset.end()) {
            continue;
        }
        auto dimSize = sizeof(int32_t);
        const auto offsetDims = (offsetShape->second) / dimSize;
        const auto blobDims = blob->getTensorDesc().getDims();
        for (size_t i = 0; i < blobDims.size(); ++i) {
            int32_t dim = static_cast<int32_t>(blobDims[i]);
            reinterpret_cast<int32_t*>(inputBuffer.data())[offsetDims + i] = dim;
        }
    }

    _executor->queueInference(_graphDesc, inputBuffer.data(),
                            _inputInfo.totalSize, nullptr, 0);
}

static void copyBlobAccordingUpperBound(
    const Blob::Ptr& in,
    const Blob::Ptr& out) {
    const auto& inDesc = in->getTensorDesc();
    const auto& outDesc = out->getTensorDesc();
    const auto inLayout = inDesc.getLayout();
    const auto outLayout = outDesc.getLayout();

    const auto& outPrec = outDesc.getPrecision();
    bool needsConvert = needsTypeConvert(outPrec);

    const auto& inBlockingDesc = inDesc.getBlockingDesc();
    const auto& outBlockingDesc = outDesc.getBlockingDesc();

    const auto& inDims = inBlockingDesc.getBlockDims();
    const auto& outDims = outBlockingDesc.getBlockDims();
    const auto inTotalDimSize = in->byteSize();

    // Strides in blocking description is presented by elements.
    // So we need to multiply them by element size
    auto inStrides = inBlockingDesc.getStrides();
    std::transform(inStrides.begin(), inStrides.end(), inStrides.begin(),
                   std::bind(std::multiplies<size_t>(), std::placeholders::_1, in->element_size()));

    IE_ASSERT(inLayout == outLayout);

    auto inPtr = in->cbuffer().as<uint8_t *>();
    IE_ASSERT(inPtr != nullptr);

    auto outPtr = out->cbuffer().as<uint8_t *>();
    IE_ASSERT(outPtr != nullptr);

    const auto inLineByteSize = inDims[inDims.size() - 1] * in->element_size();
    const auto outLineByteSize = outDims[inDims.size() - 1] * out->element_size();

    for (size_t inByteOffset = 0, outByteOffset = 0; inByteOffset < inTotalDimSize; inByteOffset += inLineByteSize) {
        auto offset = inByteOffset;
        bool isGarbageLine = false;
        for (size_t dim = 0; dim < inStrides.size() - 1; ++dim) {
            const auto coordAlongDim = offset / inStrides[dim];
            if (coordAlongDim > outDims[dim] - 1) {
                isGarbageLine = true;
                break;
            }

            offset %= inStrides[dim];
        }
        if (!isGarbageLine) {
            // We transfer outLineByteSize bytes, so garbage data at the end of the line is not copied.
            if (!needsConvert) {
                std::copy_n(inPtr + inByteOffset, outLineByteSize, outPtr + outByteOffset);
            } else {
                convertOutput(inPtr + inByteOffset, outPtr + outByteOffset, outPrec, outLineByteSize);
            }
            outByteOffset += outLineByteSize;
        }
    }
}

void MyriadInferRequest::GetResult() {
    VPU_PROFILE(GetResult);

    auto networkOutputs = _networkOutputs;
    const auto getVpuLayout = [&networkOutputs] (const std::string& name){
        const auto foundBlob = networkOutputs.find(name);
        IE_ASSERT(foundBlob != networkOutputs.end()) << "MyriadInferRequest::InferAsync()\n"
                                                    << "Output [" << name << "] is not provided.";
        return foundBlob->second->getTensorDesc().getLayout();
    };
    if (_isNetworkConstant) {
        for (const auto& output : _outputs) {
            const auto& ieBlobName = output.first;
            const auto& ieBlob = output.second;
            IE_ASSERT(_constDatas.find(ieBlobName) != _constDatas.end()) <<
            "Input [" << ieBlobName << "] is not provided.";
            std::copy_n(
                _constDatas[ieBlobName]->cbuffer().as<uint8_t *>(),
                _constDatas[ieBlobName]->byteSize(),
                ieBlob->buffer().as<uint8_t *>());
        }
        return;
    }
    // For networks with only one output
    if (_outputInfo.offset.size() == 1) {
        const auto& it = _outputs.begin();
        const auto& name = (*it).first;
        const auto& blob = (*it).second;

        if (blob->getTensorDesc().getLayout() == getVpuLayout(name) && !needsTypeConvert(blob->getTensorDesc().getPrecision())) {
            _executor->getResult(_graphDesc, blob->buffer(), static_cast<unsigned>(blob->byteSize()));
            return;
        }
    }

    _executor->getResult(_graphDesc, resultBuffer.data(), static_cast<unsigned>(resultBuffer.size()));

    for (const auto& output : _outputs) {
        const auto& ieBlobName = output.first;
        const auto& ieBlob = output.second; // Original IE output blob

        const auto resultOffset = [&](const std::string& name) {
            const auto offset_it = _outputInfo.offset.find(name);
            IE_ASSERT(offset_it != _outputInfo.offset.end())  << "MyriadInferRequest::InferAsync()\n"
                                                                    << "Output offset [" << name << "] error.";
            const auto offset = vpu::checked_cast<size_t>(offset_it->second);
            IE_ASSERT(offset <= resultBuffer.size())  << "MyriadInferRequest::InferAsync()\n"
                                                    << "Input offset is too big."
                                                    << "Required offset: " << offset
                                                    << "Result buffer size: " << resultBuffer.size();
            return offset;
        };

        const auto& ieOutDesc = ieBlob->getTensorDesc();
        const auto& ieOutPrc = ieOutDesc.getPrecision();

        auto ieOutDims = ieOutDesc.getDims();

        const auto& shapeInfo = _outputInfo.offset.find(createIOShapeName(ieBlobName));
        // if (isDynamic)
        if (shapeInfo != _outputInfo.offset.end()) {
            auto outData = networkOutputs[ieBlobName];
            const auto& descFromPlugin = _outputInfo.descFromPlugin.find(ieBlobName);
            VPU_THROW_UNLESS(descFromPlugin != _outputInfo.descFromPlugin.end(),
                "Can not find tensor descriptor by plugin for {} output", ieBlobName);
            const auto& dynOutputDesc = descFromPlugin->second;

            if (ieBlob->getTensorDesc().getDims() != dynOutputDesc.getDims()) {
                ieBlob->deallocate();
                ieBlob->getTensorDesc().reshape(dynOutputDesc.getDims(), dynOutputDesc.getLayout());
                ieBlob->allocate();
                outData->reshape(dynOutputDesc.getDims(), dynOutputDesc.getLayout());
            }

            const auto shapeResultOffset = resultOffset(shapeInfo->first);
            const auto shapePtr = reinterpret_cast<const int32_t*>(resultBuffer.data() + shapeResultOffset);

            auto shapeRank = dynOutputDesc.getDims().size();
            ieOutDims.resize(shapeRank);
            for (size_t idx = 0; idx < shapeRank; ++idx) {
                ieOutDims[idx] = shapePtr[idx];
            }

            outData->setDims(ieOutDims);
            ieBlob->getTensorDesc().setDims(ieOutDims);

            // TODO: TensorDesc doesn't update internal BlockingDesc and strides when setLayout is called
            const auto tempTensorDesc = ie::TensorDesc{ieOutPrc, dynOutputDesc.getDims(), dynOutputDesc.getLayout()};
            const auto tmpBlob = make_blob_with_precision(tempTensorDesc, resultBuffer.data() + resultOffset(ieBlobName));

            copyBlobAccordingUpperBound(tmpBlob, ieBlob);
        } else {
            copyOutput(resultBuffer.data() + resultOffset(ieBlobName), ieBlob, ieOutPrc, ieOutDims, getVpuLayout(ieBlobName));
        }
    }
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> MyriadInferRequest::GetPerformanceCounts() const {
    auto perfInfo = _executor->getPerfTimeInfo(_graphDesc._graphHandle);

    if (_log->isActive(LogLevel::Info)) {
        if (!perfInfo.empty()) {
            _log->info("Device execution time: %f ms", perfInfo[perfInfo.size()- 1]);
        }
    }

    return vpu::parsePerformanceReport(
        _stagesMetaData,
        perfInfo.data(), static_cast<int>(perfInfo.size()),
        _config.get<PerfReportModeOption>(), _config.get<EnableReceivingTensorTimeOption>());
}
