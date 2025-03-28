// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <climits>
#include <map>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {
namespace zeroProfiling {

using LayerStatistics = std::vector<ov::ProfilingInfo>;

constexpr uint32_t POOL_SIZE = 1;

struct ProfilingPool {
    ProfilingPool(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                  const std::shared_ptr<IGraph>& graph,
                  uint32_t profiling_count)
        : _init_structs(init_structs),
          _graph(graph),
          _profiling_count(profiling_count) {}
    ProfilingPool(const ProfilingPool&) = delete;
    ProfilingPool& operator=(const ProfilingPool&) = delete;
    bool create();

    ~ProfilingPool();

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    std::shared_ptr<IGraph> _graph;
    const uint32_t _profiling_count;

    ze_graph_profiling_pool_handle_t _handle = nullptr;
};

struct ProfilingQuery {
    ProfilingQuery(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, uint32_t index)
        : _init_structs(init_structs),
          _index(index) {}
    ProfilingQuery(const ProfilingQuery&) = delete;
    ProfilingQuery& operator=(const ProfilingQuery&) = delete;
    void create(const std::shared_ptr<ProfilingPool>& profiling_pool);
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

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    const uint32_t _index;

    std::shared_ptr<ProfilingPool> _profiling_pool = nullptr;

    ze_graph_profiling_query_handle_t _handle = nullptr;
};

extern template std::vector<uint8_t> ProfilingQuery::getData<uint8_t>() const;

using NpuInferStatistics = std::vector<ov::ProfilingInfo>;

struct NpuInferProfiling final {
    explicit NpuInferProfiling(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, ov::log::Level loglevel);
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
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
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
