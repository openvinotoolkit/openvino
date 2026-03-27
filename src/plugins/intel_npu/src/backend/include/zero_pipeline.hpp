// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_tensor.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_profiling.hpp"

namespace intel_npu {

class IPipeline {
public:
    IPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
              const std::shared_ptr<IGraph>& graph,
              size_t batch_size,
              const Config& config,
              const char* logName);

    IPipeline(const IPipeline&) = delete;
    IPipeline& operator=(const IPipeline&) = delete;
    IPipeline(IPipeline&&) = delete;
    IPipeline& operator=(IPipeline&&) = delete;

    virtual void push() = 0;
    virtual void pull() = 0;
    virtual void reset() const = 0;

    virtual void update_graph_arguments(uint32_t index,
                                        const std::shared_ptr<ZeroTensor>& tensor,
                                        const std::shared_ptr<ov::ITensor>& userTensor = nullptr) = 0;
    virtual void update_graph_arguments(uint32_t index,
                                        const std::shared_ptr<ZeroTensor>& tensor,
                                        size_t batch_index,
                                        const std::shared_ptr<ov::ITensor>& userTensor = nullptr) = 0;

    std::vector<ov::ProfilingInfo> get_profiling_info() const;

    virtual ~IPipeline() = default;

protected:
    void enable_profiling();

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    std::shared_ptr<IGraph> _graph;
    const Config _config;

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
    size_t _batch_size;

    std::vector<std::unique_ptr<Fence>> _fences;
    std::shared_ptr<EventPool> _event_pool;
    std::vector<std::shared_ptr<Event>> _events;
    bool _sync_output_with_fences = true;
    uint32_t _extension_version;
    bool _run_inferences_sequentially = false;
    const uint32_t _pipeline_unique_id_per_graph;

    Logger _logger;
};

class Pipeline final : public IPipeline {
public:
    Pipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
             const std::shared_ptr<IGraph>& graph,
             const Config& config,
             const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
             const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors,
             size_t batch_size = 1);

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    ~Pipeline() override = default;

    void push() override;
    void pull() override;
    void reset() const override;

    void update_graph_arguments(uint32_t index,
                                const std::shared_ptr<ZeroTensor>& tensor,
                                const std::shared_ptr<ov::ITensor>& userTensor = nullptr) override;
    void update_graph_arguments(uint32_t index,
                                const std::shared_ptr<ZeroTensor>& tensor,
                                size_t batch_index,
                                const std::shared_ptr<ov::ITensor>& userTensor = nullptr) override;

private:
    std::vector<std::unique_ptr<CommandList>> _command_lists;
};

}  // namespace intel_npu
