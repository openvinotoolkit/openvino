// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_memory.hpp"
#include "zero_profiling.hpp"
#include "zero_tensor.hpp"

namespace intel_npu {

struct Pipeline {
public:
    Pipeline(const Config& config,
             const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
             const std::shared_ptr<IGraph>& graph,
             const std::vector<std::vector<std::shared_ptr<ov::ITensor>>>& input_tensors,
             const std::vector<std::shared_ptr<ov::ITensor>>& output_tensors);

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    virtual ~Pipeline() = default;

    void push();
    void pull();
    void reset() const;

    void update_graph_arguments(uint32_t arg_index, const void* arg_data, size_t byte_size);
    void update_graph_arguments_batching(uint32_t arg_index, const void* arg_data, size_t batch_index);

    std::vector<ov::ProfilingInfo> get_profiling_info() const;

protected:
    std::shared_ptr<IGraph> _graph;
    const Config _config;
    const uint32_t _id;

    std::unique_ptr<zeroProfiling::ProfilingQuery> _profiling_query;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npu_profiling;

    /**
     * @brief Indicates how many command lists will be used inside the pipeline.
     * @details Leveraging multiple command lists implies distributing the input/output buffers accross the batch axis
     * between these lists.
     *
     * If batching is handled on compiler's side then a single command list shall be used, we don't do any
     * specific operation inside the plugin in this case.
     */
    size_t _number_of_command_lists;

    std::shared_ptr<CommandQueue> _command_queue;
    std::vector<std::unique_ptr<CommandList>> _command_lists;
    std::vector<std::unique_ptr<Fence>> _fences;
    std::shared_ptr<EventPool> _event_pool;
    std::vector<std::shared_ptr<Event>> _events;
    bool _sync_output_with_fences = true;
    Logger _logger;
};

}  // namespace intel_npu
