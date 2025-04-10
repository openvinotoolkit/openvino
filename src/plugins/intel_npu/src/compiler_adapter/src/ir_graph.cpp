// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ir_graph.hpp"

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"

namespace intel_npu {

IRGraph::IRGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                 const Config& config,
                 NetworkMetadata metadata,
                 std::unique_ptr<BlobContainer> blobPtr)
    : IGraph(nullptr, metadata, config, std::move(blobPtr)),
      _zeroInitStruct(zeroInitStruct),
      _logger("IRGraph", config.get<LOG_LEVEL>()) {
    _zeGraphExt = std::make_shared<ZeGraphExtWrappers>(_zeroInitStruct);
    _logger.info("Only create fake metadata in \"Graph\" constructor");

    // TODO: get descriptor from driver, we use pfnGetArgumentProperties3 to get the data
    ze_graph_argument_properties_3_t arg3{};
    _input_descriptors = {ArgumentDescriptor{arg3, 0}};
    _output_descriptors = {ArgumentDescriptor{arg3, 1}};

    initialize(config);
}

size_t IRGraph::export_blob(std::ostream& stream) const {
    stream.write(reinterpret_cast<const char*>(_blobPtr->get_ptr()), _blobPtr->size());

    if (!stream) {
        _logger.error("Write blob to stream failed. Blob is broken!");
        return 0;
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        std::uint32_t result = 1171117u;
        for (const uint8_t* it = reinterpret_cast<const uint8_t*>(_blobPtr->get_ptr());
             it != reinterpret_cast<const uint8_t*>(_blobPtr->get_ptr()) + _blobPtr->size();
             ++it) {
            result = ((result << 7) + result) + static_cast<uint32_t>(*it);
        }

        std::stringstream str;
        str << "Blob size: " << _blobPtr->size() << ", hash: " << std::hex << result;
        _logger.info(str.str().c_str());
    }
    _logger.info("Write blob to stream successfully.");
    return _blobPtr->size();
}

std::vector<ov::ProfilingInfo> IRGraph::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                 const Config& config) const {
    // TODO: need to process profiling data
    //  OPENVINO_THROW("Profiling post-processing is not supported.");
    return {};
}

void IRGraph::set_argument_value(uint32_t argi, const void* argv) const {
    // OPENVINO_THROW("Deferred to backend");
    (void)argi;
    (void)argv;
}

void IRGraph::initialize(const Config& config) {
    _logger.debug("Graph initialize start");
    if (config.has<WORKLOAD_TYPE>()) {
        set_workload_type(config.get<WORKLOAD_TYPE>());
    }

    if (config.get<BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER) {
        _batch_size = get_batch_size(_metadata);
    }
    _logger.debug("Graph initialize finish");

    ze_device_properties_t deviceProperties = {};
    deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_zeroInitStruct->getDevice(), &deviceProperties));
    auto groupOrdinal =
        intel_npu::zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                           ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    bool turbo = false;
    if (config.has<TURBO>()) {
        turbo = config.get<TURBO>();
    }
    _command_queue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                    zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                    groupOrdinal,
                                                    turbo);

    // TODO:  In driver_graph, _zeGraphExt->initializeGraph() will load weights to NPU memory, so we can release blob,
    // but can not do now since grpah handle is not created
}

}  // namespace intel_npu
