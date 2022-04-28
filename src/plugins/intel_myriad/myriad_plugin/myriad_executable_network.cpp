// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <legacy/cnn_network_impl.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>
#include "exec_graph_info.hpp"
#include <myriad_executable_network.h>
#include <vpu/blob_reader.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/utils/runtime_graph.hpp>
#include <legacy/net_pass.h>
#include <vpu/compile_env.hpp>
#include <vpu/configuration/options/log_level.hpp>
#include <vpu/configuration/options/throughput_streams.hpp>
#include <vpu/configuration/options/exclusive_async_requests.hpp>
#include <vpu/configuration/options/performance_hint.hpp>
#include "vpu/configuration/options/performance_hint_num_requests.hpp"
#include <vpu/configuration/options/ov_throughput_streams.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <ngraph/opsets/opset3.hpp>
// FIXME: Please remove relative path
#include "../../../src/core/include/openvino/core/interval.hpp"

using namespace InferenceEngine;

static const char importedNetworkName[] = "__importedExecutableNetworkFromBlobName";

namespace vpu {
namespace MyriadPlugin {

ExecutableNetwork::ExecutableNetwork(
        std::shared_ptr<IMvnc> mvnc,
        const PluginConfiguration& config,
        const std::shared_ptr<ie::ICore> core) :
            _config(config),
            _core(core) {
    VPU_PROFILE(ExecutableNetwork);

    const auto& logLevel = _config.get<LogLevelOption>();

    _log = std::make_shared<Logger>(
        "MyriadPlugin",
        logLevel,
        consoleOutput());

    _executor = std::make_shared<MyriadExecutor>(false, std::move(mvnc), logLevel, _log);

    _supportedMetrics = {
        METRIC_KEY(NETWORK_NAME),
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
        METRIC_KEY(DEVICE_THERMAL)
    };
}

void ExecutableNetwork::openDevice(std::vector<DevicePtr>& devicePool) {
    _device = _executor->openDevice(devicePool, _config);
    int executors = 0;
    if (_config.get<ThroughputStreamsOption>().hasValue()) {
        executors = _config.get<ThroughputStreamsOption>().get();
    } else if (_config.get<OvThroughputStreamsOption>().hasValue()) {
        executors = _config.get<OvThroughputStreamsOption>().get();
    } else if (!_config.get<PerformanceHintOption>().empty()) {
        executors = _config.get<PerformanceHintOption>() == CONFIG_VALUE(LATENCY) ? 1 : 2;
    }
    _actualNumExecutors = executors ? executors : DefaultAllocation::numStreams(_config);
}

ExecutableNetwork::ExecutableNetwork(
        const ie::CNNNetwork& network,
        std::shared_ptr<IMvnc> mvnc,
        std::vector<DevicePtr>& devicePool,
        const PluginConfiguration& config,
        const std::shared_ptr<ie::ICore> core) :
            ExecutableNetwork(std::move(mvnc), config, core) {
    VPU_PROFILE(ExecutableNetwork);

    const auto compilerLog = std::make_shared<Logger>(
        "GraphCompiler",
        _config.get<LogLevelOption>(),
        consoleOutput());

    ie::CNNNetwork copyNetwork = network;
    if (copyNetwork.getFunction() && copyNetwork.getFunction()->is_dynamic()) {
        copyNetwork = InferenceEngine::details::cloneNetwork(network);
        auto function = copyNetwork.getFunction();
        for (const auto& input : function->get_parameters()) {
            if (input->get_partial_shape().is_dynamic()) {
                auto inputShape = input->get_partial_shape();
                const auto inDataParam = std::make_shared<ngraph::opset3::Parameter>(
                    input->get_output_element_type(0), inputShape.get_max_shape());
                const auto inDataShapeParam = std::make_shared<ngraph::opset3::Parameter>(
                    ngraph::element::i64, ov::Shape{inputShape.get_max_shape().size()});
                inDataShapeParam->set_friendly_name(input->get_friendly_name()+"_real_shape");
                inDataParam->set_friendly_name(input->get_friendly_name());
                inDataParam->get_output_tensor(0).set_names(input->get_output_tensor(0).get_names());
                const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
                    inDataParam, inDataShapeParam,
                    ngraph::vpu::op::DynamicShapeResolverMode::INFER_DYNAMIC_SHAPE, input->get_partial_shape());
                function->replace_node(input, dsr);
                function->remove_parameter(input);
                function->add_parameters({inDataShapeParam, inDataParam});
            }
        }
        copyNetwork = ie::CNNNetwork(function);
        for (const auto& inputInf : network.getInputsInfo()) {
            copyNetwork.getInputsInfo()[inputInf.first]->setPrecision(inputInf.second->getPrecision());
            copyNetwork.getInputsInfo()[inputInf.first]->setLayout(inputInf.second->getLayout());
            copyNetwork.getInputsInfo()[inputInf.first]->getPreProcess() = inputInf.second->getPreProcess();
        }
        for (const auto& outputInf : network.getOutputsInfo()) {
            *copyNetwork.getOutputsInfo()[outputInf.first].get() = *outputInf.second.get();
        }
    }
    auto compiledGraph = compileNetwork(
        copyNetwork,
        _config,
        compilerLog,
        _core);

    _actualNumExecutors = compiledGraph->numExecutors;
    _graphBlob = std::move(compiledGraph->blob);
    _graphMetaData = std::move(compiledGraph->graphMeta);

    _inputInfo  = std::move(compiledGraph->inputInfo);
    _outputInfo = std::move(compiledGraph->outputInfo);

    const auto& networkName = network.getName();
    if (_config.get<ExclusiveAsyncRequestsOption>()) {
        _taskExecutor = executorManager()->getExecutor("MYRIAD");
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
    if (_inputInfo.totalSize == 0) {
        _isNetworkConstant = true;
        const auto& nGraphFunc = network.getFunction();
        const auto& sortedLayers = nGraphFunc->get_ordered_ops();
        for (const auto& layer : sortedLayers) {
            if (strcmp(layer->get_type_info().name, "Constant") == 0) {
                const auto& constOp = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(layer);
                auto name = constOp->get_friendly_name();
                _constDatas[name] = ie::details::shareWeights(constOp);
            }
        }
        return;
    }
    openDevice(devicePool);
    _executor->allocateGraph(_device, _graphDesc, _graphBlob, compiledGraph->blobHeader, compiledGraph->numActiveStages,
                             networkName, _actualNumExecutors, _config);
}

void ExecutableNetwork::Import(std::istream& strm, std::vector<DevicePtr> &devicePool, const PluginConfiguration& configuration) {
    auto currentPos = strm.tellg();
    strm.seekg(0, strm.end);
    auto blobSize = strm.tellg() - currentPos;
    _graphBlob.resize(static_cast<size_t>(blobSize));
    strm.seekg(currentPos, strm.beg);
    strm.read(&_graphBlob[0], blobSize);

    std::string networkName = importedNetworkName;

    BlobReader blobReader;
    blobReader.parse(_graphBlob);

    this->_networkInputs  = blobReader.getNetworkInputs();
    this->_networkOutputs = blobReader.getNetworkOutputs();
    if (blobSize == blobReader.getFileSize()) {
        _log->warning(
            "Older version of blob. Unable to get information about network "
            "parameters/results. Please recompile blob");
    }
    this->setInputs(blobReader.getNetworkParemeters());
    this->setOutputs(blobReader.getNetworkResults());

    _inputInfo  = blobReader.getInputInfo();
    _outputInfo = blobReader.getOutputInfo();

    std::size_t numStages = blobReader.getStageCount();
    auto blobHeader = blobReader.getHeader();

    openDevice(devicePool);
    _executor->allocateGraph(_device, _graphDesc, _graphBlob, blobHeader, numStages, networkName, _actualNumExecutors, _config);
    _graphMetaData.stagesMeta.resize(numStages);
    for (auto &meta : _graphMetaData.stagesMeta) {
        meta.stageName = meta.stageType = meta.layerName = meta.layerType = "UNKNOWN";
        meta.status = InferenceEngineProfileInfo::LayerStatus::EXECUTED;
    }

    if (_config.get<ExclusiveAsyncRequestsOption>()) {
        _taskExecutor = executorManager()->getExecutor("MYRIAD");
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

ExecutableNetwork::ExecutableNetwork(std::istream& strm,
                               std::shared_ptr<IMvnc> mvnc,
                               std::vector<DevicePtr> &devicePool,
                               const PluginConfiguration& config,
                               const std::shared_ptr<ie::ICore> core) :
    ExecutableNetwork(std::move(mvnc), config, core) {
    VPU_PROFILE(ExecutableNetwork);
    Import(strm, devicePool, config);
}

ExecutableNetwork::ExecutableNetwork(
        const std::string& blobFilename,
        std::shared_ptr<IMvnc> mvnc,
        std::vector<DevicePtr>& devicePool,
        const PluginConfiguration& config,
        const std::shared_ptr<ie::ICore> core) :
    ExecutableNetwork(std::move(mvnc), config, core) {
    VPU_PROFILE(ExecutableNetwork);
    std::ifstream blobFile{blobFilename, std::ios::binary};
    Import(blobFile, devicePool, config);
}

InferenceEngine::Parameter ExecutableNetwork::GetMetric(const std::string &name) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _graphDesc._name);
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, _supportedMetrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>());
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int optimalNumOfInferRequests = static_cast<unsigned int>(2u * _actualNumExecutors);

        if (!_config.get<PerformanceHintOption>().empty()) {
            optimalNumOfInferRequests =
                    _config.get<PerformanceHintOption>() == CONFIG_VALUE(THROUGHPUT) ? optimalNumOfInferRequests : 1;
        }
        if (_config.get<PerformanceHintNumRequestsOption>() != 0) {
            optimalNumOfInferRequests =
                    std::min(optimalNumOfInferRequests,
                             static_cast<unsigned int>(_config.get<PerformanceHintNumRequestsOption>()));
        }
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNumOfInferRequests);
    } else if (name == METRIC_KEY(DEVICE_THERMAL)) {
        IE_SET_METRIC_RETURN(DEVICE_THERMAL, _executor->GetThermal(_device));
    } else {
        IE_THROW(NotImplemented);
    }
}

InferenceEngine::Parameter ExecutableNetwork::GetConfig(const std::string &name) const {
    auto confValues = _config.getValues();
    auto it = confValues.find(name);
    if (it != confValues.end()) {
        return it->second;
    }
    VPU_THROW_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
}

std::shared_ptr<ngraph::Function> ExecutableNetwork::GetExecGraphInfo() {
    auto perfInfo = _executor->getPerfTimeInfo(_graphDesc._graphHandle);
    if (_graphDesc._name == importedNetworkName)
        IE_THROW() <<
        "GetExecGraphInfo() can't be called for ExecutableNetwork that was imported from a compiled blob as far getting"
        " original stage names, types, and topological order from the compiled blob is not implemented for now.";
    return buildRuntimeGraph(_graphMetaData, perfInfo);
}

}  // namespace MyriadPlugin
}  // namespace vpu
