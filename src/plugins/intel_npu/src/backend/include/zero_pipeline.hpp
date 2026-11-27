// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

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

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const = 0;

    virtual ~IPipeline() = default;

    // Helper function handling strides for Zero Pipeline, ZeroDynamicPipeline and ZeroExecutionList. Stateless
    // regarding pipeline instances, hence static
    static std::vector<size_t> get_strides(const ov::ITensor& tensor, bool reverse_order = false);

protected:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    std::shared_ptr<IGraph> _graph;
    const Config _config;

    /**
     * @brief Indicates how many command lists will be used inside the pipeline.
     * @details Leveraging multiple command lists implies distributing the input/output buffers accross the batch axis
     * between these lists.
     *
     * If batching is handled on compiler's side then a single command list shall be used, we don't do any
     * specific operation inside the plugin in this case.
     */
    size_t _batch_size;

    std::shared_ptr<CommandQueue> _command_queue = nullptr;
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

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    void configure_profiling();

    std::unique_ptr<zeroProfiling::ProfilingQuery> _profiling_query;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npu_profiling;
    std::vector<std::unique_ptr<CommandList>> _command_lists;
};

/**
 * @brief Pipeline of a request that is executed by an execution list. The list records and submits the graph itself,
 * reading the request's tensors directly, so this pipeline neither records nor submits anything.
 */
class ExternalPipeline final : public IPipeline {
public:
    ExternalPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                     const std::shared_ptr<IGraph>& graph,
                     const Config& config)
        : IPipeline(init_structs, graph, 1, config, "ExternalPipeline") {}

    void push() override {
        OPENVINO_THROW("The request is executed by an execution list and can't be started on its own");
    }
    void pull() override {
        OPENVINO_THROW("The request is executed by an execution list and can't be waited for on its own");
    }
    void reset() const override {}

    // The execution list detects changed tensor addresses itself
    void update_graph_arguments(uint32_t, const std::shared_ptr<ZeroTensor>&, const std::shared_ptr<ov::ITensor>&)
        override {}
    void update_graph_arguments(uint32_t,
                                const std::shared_ptr<ZeroTensor>&,
                                size_t,
                                const std::shared_ptr<ov::ITensor>&) override {}

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
};

}  // namespace intel_npu
