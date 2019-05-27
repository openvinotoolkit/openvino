// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX
#include <ie_blob.h>
#include <ie_plugin.hpp>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>
#include <precision_utils.h>

#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/ie_helpers.hpp>

#include "myriad_executable_network.h"
#include "myriad_infer_request.h"

using namespace vpu;
using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;

#define MEMCPY(dst, src, bytes) std::copy_n((src), (bytes), (dst))

MyriadInferRequest::MyriadInferRequest(GraphDesc &graphDesc,
                                        InferenceEngine::InputsDataMap networkInputs,
                                        InferenceEngine::OutputsDataMap networkOutputs,
                                        DataInfo& inputInfo,
                                        DataInfo& outputInfo,
                                        const std::vector<StageMetaInfo> &blobMetaData,
                                        const std::shared_ptr<MyriadConfig> &myriadConfig,
                                        const Logger::Ptr &log,
                                        const MyriadExecutorPtr &executor) :
        InferRequestInternal(networkInputs, networkOutputs), _executor(executor),
        _log(log), _stagesMetaData(blobMetaData), _config(myriadConfig),
        _inputInfo(inputInfo), _outputInfo(outputInfo),
        _graphDesc(graphDesc) {
    _deviceLayout = _config->compileConfig.hwOptimization ? NCHW : NHWC;
    if (_config->compileConfig.forceLayout == ComputeLayout::NCHW)
        _deviceLayout = NCHW;
    if (_config->compileConfig.forceLayout == ComputeLayout::NHWC)
        _deviceLayout = NHWC;
    // allocate inputs
    for (auto &networkInput : _networkInputs) {
        // TODO: use TensorDesc instead of deprecated methods
        SizeVector dims      = networkInput.second->getDims();
        Precision  precision = networkInput.second->getInputPrecision();
        Layout     layout    = networkInput.second->getTensorDesc().getLayout();

        if (precision != Precision::FP32 &&
            precision != Precision::FP16 &&
            precision != Precision::U8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: "
                                   << precision << "! Supported precisions: FP32, FP16 and U8";
        }
        Blob::Ptr inputBlob = make_blob_with_precision(precision, layout, dims);

        // allocate the input blob
        // TODO We are allocating temporary input buffer of enough size. Wrap this buffer in blobs
        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
    }
    // allocate outputs
    for (auto &networkOutput : _networkOutputs) {
        SizeVector dims      = networkOutput.second->dims;
        Precision  precision = networkOutput.second->getPrecision();
        Layout     layout    = networkOutput.second->layout;

        if (precision != Precision::FP32 &&
            precision != Precision::FP16) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: "
                                << precision << "! Supported precisions: FP32, FP16";
        }
        Blob::Ptr outputBlob = make_blob_with_precision(precision, layout, dims);
        // allocate the output blob
        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
    }

    inputBuffer .resize(inputInfo.totalSize);
    resultBuffer.resize(outputInfo.totalSize);

    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }
}

void MyriadInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void MyriadInferRequest::InferAsync() {
    for (auto input : _inputs) {
        auto const inputBlobPtr = input.second;
        if (inputBlobPtr->precision() != Precision::FP16
            && inputBlobPtr->precision() != Precision::FP32
            && inputBlobPtr->precision() != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input blob precision";
    }
    for (auto output : _outputs) {
        auto const outputBlobPtr = output.second;
        if (outputBlobPtr->precision() != Precision::FP16
            && outputBlobPtr->precision() != Precision::FP32)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output blob precision";
    }

    // execute input pre-processing
    execDataPreprocessing(_inputs, true);  // "true" stands for serial preprocessing in case of OpenMP

    Blob::Ptr tmpBlob;

    void* inputPtr = nullptr;
    size_t inputSize = _inputInfo.totalSize;

    if (_inputs.size() > 1) {
        for (auto&& input : _inputs) {
            auto inputBlob = input.second;
            size_t byteSize = inputBlob->byteSize();
            Layout layout = inputBlob->getTensorDesc().getLayout();
            if (layout != _deviceLayout && (layout == NCHW || layout == NHWC)) {
                // TODO copyBlob allocates new memory, but we already have allocated buffer of enough size
                inputBlob = copyBlob(inputBlob, _deviceLayout);
            }

            const auto input_offset_it = _inputInfo.offset.find(input.first);
            if (input_offset_it != _inputInfo.offset.end()) {
                size_t required_buff_size = checked_cast<size_t>(input_offset_it->second) + byteSize;
                IE_ASSERT(required_buff_size <= inputBuffer.size());
                MEMCPY(&inputBuffer[input_offset_it->second], inputBlob->buffer().as<uint8_t*>(), byteSize);
            }
        }

        inputPtr = inputBuffer.data();
    } else {
        auto dataName = _networkInputs.begin()->first;
        auto foundInputBlob = _inputs.find(dataName);
        if (foundInputBlob == _inputs.end())
            THROW_IE_EXCEPTION << "Error: input [" << dataName << "] is not provided.";

        tmpBlob = foundInputBlob->second;
        Layout layout = tmpBlob->getTensorDesc().getLayout();
        if (layout != _deviceLayout && (layout == NCHW || layout == NHWC)) {
            // TODO copyBlob allocates new memory, but we already have allocated buffer of enough size
            tmpBlob = copyBlob(tmpBlob, _deviceLayout);
        }

        inputPtr = tmpBlob->buffer();
    }

    _executor->queueInference(_graphDesc, inputPtr, inputSize, nullptr, 0);
}

void MyriadInferRequest::GetResult() {
    _executor->getResult(_graphDesc, resultBuffer.data(), resultBuffer.size());

    for (auto pp : _outputs) {
        const auto offset_it = _outputInfo.offset.find(pp.first);

        if (offset_it !=  _outputInfo.offset.end()) {
            size_t resultOffset = checked_cast<size_t>(offset_it->second);
            if (resultOffset > resultBuffer.size()) {
                THROW_IE_EXCEPTION << "unexpected result data size";
            }

            auto outputBlob = pp.second;
            auto outDesc = outputBlob->getTensorDesc();

            // TODO: TensorDesc doesn't update internal BlockingDesc and strides when setLayout is called
            auto vpuLayout = (outDesc.getLayout() == NCHW || outDesc.getLayout() == NHWC) ? _deviceLayout : outDesc.getLayout();
            ie::TensorDesc tempTensorDesc(outDesc.getPrecision(), outDesc.getDims(), vpuLayout);
            auto tmpBlob = make_blob_with_precision(tempTensorDesc, resultBuffer.data() + resultOffset);

            copyBlob(tmpBlob, outputBlob);
        }
    }
}

void MyriadInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    auto perfInfo = _executor->getPerfTimeInfo(_graphDesc._graphHandle);

    if (_log->level() >= LogLevel::Info) {
        if (!perfInfo.empty()) {
            _log->info("** Device execution time %f **", perfInfo[perfInfo.size()- 1]);
        }
    }

    perfMap = vpu::parsePerformanceReport(
        _stagesMetaData,
        perfInfo.data(), perfInfo.size(),
        _config->perfReport, _config->printReceiveTensorTime);
}
