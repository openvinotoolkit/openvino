// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <climits>
#include <map>

#include "intel_npu/config/compiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {
namespace zeroProfiling {

using LayerStatistics = std::vector<ov::ProfilingInfo>;

constexpr uint32_t POOL_SIZE = 1;

struct ProfilingPool {
    ProfilingPool(ze_graph_handle_t graph_handle,
                  uint32_t profiling_count,
                  ze_graph_profiling_dditable_ext_curr_t& graph_profiling_ddi_table_ext)
        : _graph_handle(graph_handle),
          _profiling_count(profiling_count),
          _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext) {}
    ProfilingPool(const ProfilingPool&) = delete;
    ProfilingPool& operator=(const ProfilingPool&) = delete;
    bool create();

    ~ProfilingPool();

    ze_graph_handle_t _graph_handle;
    const uint32_t _profiling_count;
    ze_graph_profiling_pool_handle_t _handle = nullptr;
    ze_graph_profiling_dditable_ext_curr_t& _graph_profiling_ddi_table_ext;
};

struct ProfilingQuery {
    ProfilingQuery(uint32_t index,
                   ze_device_handle_t device_handle,
                   ze_graph_profiling_dditable_ext_curr_t& graph_profiling_ddi_table_ext)
        : _index(index),
          _device_handle(device_handle),
          _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext) {}
    ProfilingQuery(const ProfilingQuery&) = delete;
    ProfilingQuery& operator=(const ProfilingQuery&) = delete;
    void create(const ze_graph_profiling_pool_handle_t& profiling_pool);
    ze_graph_profiling_query_handle_t getHandle() const {
        return _handle;
    }
    LayerStatistics getLayerStatistics() const;
    template <class ProfilingData>
    std::vector<ProfilingData> getData() const;
    ~ProfilingQuery();

private:
    void queryGetData(const ze_graph_profiling_type_t profilingType, uint32_t* pSize, uint8_t* pData) const;
    void getProfilingProperties(ze_device_profiling_data_properties_t* properties) const;
    void verifyProfilingProperties() const;

    const uint32_t _index;
    ze_device_handle_t _device_handle;
    ze_graph_profiling_query_handle_t _handle = nullptr;
    ze_graph_profiling_dditable_ext_curr_t& _graph_profiling_ddi_table_ext;
};

extern template std::vector<uint8_t> ProfilingQuery::getData<uint8_t>() const;

using NpuInferStatistics = std::vector<ov::ProfilingInfo>;

struct NpuInferProfiling final {
    explicit NpuInferProfiling(ze_context_handle_t context, ze_device_handle_t device_handle, ov::log::Level loglevel);
    NpuInferProfiling(const NpuInferProfiling&) = delete;
    NpuInferProfiling& operator=(const NpuInferProfiling&) = delete;
    NpuInferProfiling(NpuInferProfiling&&) = delete;
    NpuInferProfiling& operator=(NpuInferProfiling&&) = delete;

    void sampleNpuTimestamps();
    NpuInferStatistics getNpuInferStatistics() const;

    ~NpuInferProfiling();

    /// Buffers allocated by ZE driver
    void* npu_ts_infer_start = 0;
    void* npu_ts_infer_end = 0;

private:
    ze_context_handle_t _context = nullptr;
    ze_device_handle_t _device_handle;
    ov::log::Level _loglevel;
    Logger _logger;
    ze_device_properties_t _dev_properties = {};
    int64_t _npu_infer_stats_min_cc = LLONG_MAX;
    int64_t _npu_infer_stats_max_cc = 0;
    int64_t _npu_infer_stats_accu_cc = 0;
    uint32_t _npu_infer_stats_cnt = 0;
    uint32_t _npu_infer_logidx = 0;
    static const uint32_t _npu_infer_log_maxsize = 1024;
    /// rolling buffer to store duration of last <_npu_infer_log_maxsize number> infers
    int64_t _npu_infer_duration_log[_npu_infer_log_maxsize];

    /// Helper function to convert npu clockcycles to usec
    int64_t convertCCtoUS(int64_t val_cc) const;
};

}  // namespace zeroProfiling
}  // namespace intel_npu
