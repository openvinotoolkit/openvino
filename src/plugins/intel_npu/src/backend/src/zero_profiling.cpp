// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_profiling.hpp"

#include <ze_graph_profiling_ext.h>

#include "intel_npu/config/compiler.hpp"
#include "intel_npu/profiling.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "zero_profiling.hpp"

namespace intel_npu {
namespace zeroProfiling {

/// @brief Type trait mapping from ZE data type to enum value
template <typename T>
struct ZeProfilingTypeId {};

template <>
struct ZeProfilingTypeId<ze_profiling_layer_info> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_LAYER_LEVEL;
};

template <>
struct ZeProfilingTypeId<ze_profiling_task_info> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_TASK_LEVEL;
};

template <>
struct ZeProfilingTypeId<uint8_t> {
    static const ze_graph_profiling_type_t value = ZE_GRAPH_PROFILING_RAW;
};

bool ProfilingPool::create() {
    auto ret =
        _init_structs->getProfilingDdiTable().pfnProfilingPoolCreate(_graph->get_handle(), _profiling_count, &_handle);
    return ((ZE_RESULT_SUCCESS == ret) && (_handle != nullptr));
}

ProfilingPool::~ProfilingPool() {
    if (_handle) {
        _init_structs->getProfilingDdiTable().pfnProfilingPoolDestroy(_handle);
    }
}

void ProfilingQuery::create(const std::shared_ptr<ProfilingPool>& profiling_pool) {
    _profiling_pool =
        profiling_pool;  // store profiling pool to make sure it is keeped alive until profiling query is destroyed
    THROW_ON_FAIL_FOR_LEVELZERO(
        "pfnProfilingQueryCreate",
        _init_structs->getProfilingDdiTable().pfnProfilingQueryCreate(_profiling_pool->_handle, _index, &_handle));
}

LayerStatistics ProfilingQuery::getLayerStatistics() const {
    verifyProfilingProperties();
    auto layerData = getData<ze_profiling_layer_info>();
    return profiling::convertLayersToIeProfilingInfo(layerData);
}

ProfilingQuery::~ProfilingQuery() {
    if (_handle) {
        _init_structs->getProfilingDdiTable().pfnProfilingQueryDestroy(_handle);
    }
}

void ProfilingQuery::queryGetData(const ze_graph_profiling_type_t profilingType,
                                  uint32_t* pSize,
                                  uint8_t* pData) const {
    if (_handle && pSize) {
        THROW_ON_FAIL_FOR_LEVELZERO(
            "pfnProfilingQueryGetData",
            _init_structs->getProfilingDdiTable().pfnProfilingQueryGetData(_handle, profilingType, pSize, pData));
    }
}

template <class ProfilingData>
std::vector<ProfilingData> ProfilingQuery::getData() const {
    ze_graph_profiling_type_t type = ZeProfilingTypeId<ProfilingData>::value;
    uint32_t size = 0;

    // Obtain the size of the buffer
    queryGetData(type, &size, nullptr);

    OPENVINO_ASSERT(size % sizeof(ProfilingData) == 0);

    // Allocate enough memory and copy the buffer
    std::vector<ProfilingData> profilingData(size / sizeof(ProfilingData));
    queryGetData(type, &size, reinterpret_cast<uint8_t*>(profilingData.data()));
    return profilingData;
}

template std::vector<uint8_t> ProfilingQuery::getData<uint8_t>() const;

void ProfilingQuery::getProfilingProperties(ze_device_profiling_data_properties_t* properties) const {
    if (_handle && properties) {
        THROW_ON_FAIL_FOR_LEVELZERO(
            "getProfilingProperties",
            _init_structs->getProfilingDdiTable().pfnDeviceGetProfilingDataProperties(_init_structs->getDevice(),
                                                                                      properties));
    }
}

void ProfilingQuery::verifyProfilingProperties() const {
    if (!_handle) {
        OPENVINO_THROW("No available profiling data.");
    }
    const auto stringifyVersion = [](auto version) -> std::string {
        return std::to_string(ZE_MAJOR_VERSION(version)) + "." + std::to_string(ZE_MINOR_VERSION(version));
    };

    ze_device_profiling_data_properties_t profProp;
    getProfilingProperties(&profProp);
    const auto currentProfilingVersion = ze_profiling_data_ext_version_t::ZE_PROFILING_DATA_EXT_VERSION_CURRENT;

    if (ZE_MAJOR_VERSION(profProp.extensionVersion) != ZE_MAJOR_VERSION(currentProfilingVersion)) {
        OPENVINO_THROW("Unsupported NPU driver.",
                       "Profiling API version: plugin: ",
                       stringifyVersion(currentProfilingVersion),
                       ", driver: ",
                       stringifyVersion(profProp.extensionVersion));
    }
    if (currentProfilingVersion > profProp.extensionVersion) {
        auto log = Logger::global().clone("ZeroProfilingQuery");
        log.warning("Outdated NPU driver detected. Some features might not be available! "
                    "Profiling API version: plugin: %s, driver: %s",
                    stringifyVersion(currentProfilingVersion).c_str(),
                    stringifyVersion(profProp.extensionVersion).c_str());
    }
}

NpuInferStatistics NpuInferProfiling::getNpuInferStatistics() const {
    NpuInferStatistics npuPerfCounts;

    /// if the log isn't full/rolled over yet = skip reporting empty logs
    uint32_t stat_cnt = (_npu_infer_stats_cnt < _npu_infer_log_maxsize) ? _npu_infer_stats_cnt : _npu_infer_log_maxsize;
    if (stat_cnt != 0 && _loglevel >= ov::log::Level::WARNING) {
        /// Populate npuinferstatistics vector
        for (unsigned i = 0; i < stat_cnt; i++) {
            ov::ProfilingInfo info;

            info.status = ov::ProfilingInfo::Status::EXECUTED;
            info.real_time = std::chrono::microseconds(convertCCtoUS(_npu_infer_duration_log[i]));
            info.cpu_time = std::chrono::microseconds(convertCCtoUS(_npu_infer_duration_log[i]));
            info.node_name = std::to_string(i);
            info.exec_type = "INFER_REQ";
            info.node_type = "INFER_REQ";

            npuPerfCounts.push_back(std::move(info));
        }
    }

    /// sanity check to avoid division by 0
    if (_npu_infer_stats_cnt == 0) {
        return {};
    }

    /// Add final statistics
    ov::ProfilingInfo info_avg = {
        ov::ProfilingInfo::Status::EXECUTED,
        std::chrono::microseconds(convertCCtoUS(_npu_infer_stats_accu_cc / _npu_infer_stats_cnt)),
        std::chrono::microseconds(convertCCtoUS(_npu_infer_stats_accu_cc / _npu_infer_stats_cnt)),
        "AVG",
        "AVG",
        "AVG"};
    npuPerfCounts.push_back(std::move(info_avg));
    ov::ProfilingInfo info_min = {ov::ProfilingInfo::Status::EXECUTED,
                                  std::chrono::microseconds(convertCCtoUS(_npu_infer_stats_min_cc)),
                                  std::chrono::microseconds(convertCCtoUS(_npu_infer_stats_min_cc)),
                                  "MIN",
                                  "MIN",
                                  "MIN"};
    npuPerfCounts.push_back(std::move(info_min));
    ov::ProfilingInfo info_max = {ov::ProfilingInfo::Status::EXECUTED,
                                  std::chrono::microseconds(convertCCtoUS(_npu_infer_stats_max_cc)),
                                  std::chrono::microseconds(convertCCtoUS(_npu_infer_stats_max_cc)),
                                  "MAX",
                                  "MAX",
                                  "MAX"};
    npuPerfCounts.push_back(std::move(info_max));
    return npuPerfCounts;
}

NpuInferProfiling::NpuInferProfiling(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                     ov::log::Level loglevel)
    : _init_structs(init_structs),
      _loglevel(loglevel),
      _logger("InferProfiling", loglevel) {
    /// Fetch and store the device timer resolution
    _dev_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_init_structs->getDevice(), &_dev_properties));
    /// Request mem allocations
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                     nullptr,
                                     ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED};
    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeMemAllocHost",
        zeMemAllocHost(_init_structs->getContext(),
                       &desc,
                       sizeof(uint64_t),
                       64,
                       &npu_ts_infer_start));  // align to 64 bytes to match npu l2 cache line size
    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeMemAllocHost",
        zeMemAllocHost(_init_structs->getContext(),
                       &desc,
                       sizeof(uint64_t),
                       64,
                       &npu_ts_infer_end));  // alight to 64 bytes to match npu l2 cache line size
}

void NpuInferProfiling::sampleNpuTimestamps() {
    int64_t infer_duration_cc = static_cast<int64_t>(*(reinterpret_cast<uint64_t*>(npu_ts_infer_end)) -
                                                     *(reinterpret_cast<uint64_t*>(npu_ts_infer_start)));

    /// Update extremas
    if (infer_duration_cc < _npu_infer_stats_min_cc)
        _npu_infer_stats_min_cc = infer_duration_cc;
    if (infer_duration_cc > _npu_infer_stats_max_cc)
        _npu_infer_stats_max_cc = infer_duration_cc;
    _npu_infer_stats_accu_cc += infer_duration_cc;
    _npu_infer_stats_cnt++;
    /// only log individual infer durations if requested
    if (_loglevel >= ov::log::Level::WARNING) {
        _npu_infer_duration_log[_npu_infer_logidx++] = infer_duration_cc;
        if (_npu_infer_logidx >= _npu_infer_log_maxsize)
            _npu_infer_logidx = 0;
    }
}

int64_t NpuInferProfiling::convertCCtoUS(int64_t val_cc) const {
    return (int64_t)(val_cc * 1000 * 1000 / _dev_properties.timerResolution);
}

NpuInferProfiling::~NpuInferProfiling() {
    /// deallocate npu_ts_infer_start and npu_ts_infer_end, allocated externally by ze driver
    if (npu_ts_infer_start != nullptr) {
        auto ze_ret = zeMemFree(_init_structs->getContext(), npu_ts_infer_start);
        if (ZE_RESULT_SUCCESS != ze_ret) {
            _logger.error("zeMemFree on npu_ts_infer_start failed %#X", uint64_t(ze_ret));
        }
    }
    if (npu_ts_infer_end != nullptr) {
        auto ze_ret = zeMemFree(_init_structs->getContext(), npu_ts_infer_end);
        if (ZE_RESULT_SUCCESS != ze_ret) {
            _logger.error("zeMemFree on npu_ts_infer_end failed %#X", uint64_t(ze_ret));
        }
    }
}

}  // namespace zeroProfiling
}  // namespace intel_npu
