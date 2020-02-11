// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX
#include <utility>
#include <ie_blob.h>
#include <ie_plugin.hpp>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>
#include <precision_utils.h>

#include <vpu/utils/error.hpp>
#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/profiling.hpp>

#include "myriad_executable_network.h"
#include "myriad_infer_request.h"

using namespace vpu;
using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;

#define MEMCPY(dst, src, bytes) std::copy_n((src), (bytes), (dst))

MyriadInferRequest::MyriadInferRequest(GraphDesc &graphDesc,
                                       InferenceEngine::InputsDataMap networkInputs,
                                       InferenceEngine::OutputsDataMap networkOutputs,
                                       DataInfo& compilerInputsInfo,
                                       DataInfo& compilerOutputsInfo,
                                       const std::vector<StageMetaInfo> &blobMetaData,
                                       const MyriadConfig& myriadConfig,
                                       const Logger::Ptr &log,
                                       const MyriadExecutorPtr &executor) :
        InferRequestInternal(networkInputs, networkOutputs), _executor(executor),
        _log(log), _stagesMetaData(blobMetaData), _config(myriadConfig),
        _inputInfo(compilerInputsInfo), _outputInfo(compilerOutputsInfo),
        _graphDesc(graphDesc) {
    VPU_PROFILE(MyriadInferRequest);

    const auto& ioStrides = _config.compileConfig().ioStrides;
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

    inputBuffer .resize(compilerInputsInfo.totalSize);
    resultBuffer.resize(compilerOutputsInfo.totalSize);

    VPU_THROW_UNLESS(
        !_networkOutputs.empty() && !_networkInputs.empty(),
        "No information about network's output/input");
}

void MyriadInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void MyriadInferRequest::InferAsync() {
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
        const auto requiredSize = vpu::checked_cast<size_t>(offset) + byteSize;
        IE_ASSERT(requiredSize <= inputBuffer.size())  << "MyriadInferRequest::InferAsync()\n"
                                                       << "Input offset is too big."
                                                       << "Required size: " << requiredSize
                                                       << "Input buffer size: " << inputBuffer.size();

        const auto foundBlob = getNetInputInfo(name);
        const auto vpuLayout = foundBlob->second->getTensorDesc().getLayout();
        const auto layout = blob->getTensorDesc().getLayout();

        if (layout != vpuLayout) {
            copyBlob(blob, vpuLayout, &inputBuffer[offset]);
        } else {
            MEMCPY(&inputBuffer[offset], blob->buffer().as<uint8_t*>(), byteSize);
        }
    }

    _executor->queueInference(_graphDesc, inputBuffer.data(),
                              _inputInfo.totalSize, nullptr, 0);
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

    if (_outputs.size() == 1) {
        const auto& it = _outputs.begin();
        const auto& name = (*it).first;
        const auto& blob = (*it).second;

        if (blob->getTensorDesc().getLayout() == getVpuLayout(name)) {
            _executor->getResult(_graphDesc, blob->buffer(), blob->byteSize());
            return;
        }
    }

    _executor->getResult(_graphDesc, resultBuffer.data(), resultBuffer.size());

    for (const auto& output : _outputs) {
        const auto& name = output.first;
        const auto& blob = output.second;

        const auto resultOffset = [&] {
            const auto offset_it = _outputInfo.offset.find(name);
            IE_ASSERT(offset_it != _outputInfo.offset.end())  << "MyriadInferRequest::InferAsync()\n"
                                                                       << "Output offset [" << name << "] error.";
            const auto offset = vpu::checked_cast<size_t>(offset_it->second);
            IE_ASSERT(offset <= resultBuffer.size())  << "MyriadInferRequest::InferAsync()\n"
                                                      << "Input offset is too big."
                                                      << "Required offset: " << offset
                                                      << "Result buffer size: " << resultBuffer.size();
            return offset;
        }();

        const auto outDesc = blob->getTensorDesc();
        // TODO: TensorDesc doesn't update internal BlockingDesc and strides when setLayout is called
        const auto tempTensorDesc = ie::TensorDesc{outDesc.getPrecision(), outDesc.getDims(), getVpuLayout(name)};
        const auto tmpBlob = make_blob_with_precision(tempTensorDesc, resultBuffer.data() + resultOffset);

        copyBlob(tmpBlob, blob);
    }
}

void MyriadInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    auto perfInfo = _executor->getPerfTimeInfo(_graphDesc._graphHandle);

    if (_log->isActive(LogLevel::Info)) {
        if (!perfInfo.empty()) {
            _log->info("Device execution time: %f ms", perfInfo[perfInfo.size()- 1]);
        }
    }

    perfMap = vpu::parsePerformanceReport(
        _stagesMetaData,
        perfInfo.data(), perfInfo.size(),
        _config.perfReport(), _config.printReceiveTensorTime());
}
