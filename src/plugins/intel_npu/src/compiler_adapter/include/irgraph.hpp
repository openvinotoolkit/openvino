// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/icompiler.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/so_ptr.hpp"

#ifdef NPU_LLVM_BACKEND
namespace intel_npu {

class IRGraph final : public IGraph {
public:
    struct MemRefType {
        void *basePtr;
        void *data;
        int64_t offset;
        int64_t sizes[4];
        int64_t strides[4];
        MemRefType() {
            basePtr = nullptr;
            data = nullptr;
            offset = 0;
        }

        void setSize(const intel_npu::IODescriptor& descriptor);
        void updateStride();
    };

    class Impl {
        using MemRefType = IRGraph::MemRefType;

    public:
        virtual void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata, std::vector<ArgumentDescriptor>& inputs, std::vector<ArgumentDescriptor>& outputs) = 0;
        virtual void setArgumentValue(uint32_t argi, const void* argv) = 0;
        virtual uint64_t getNumSubgraphs() = 0;
        virtual void initializeGraph(uint64_t command_queue_group_ordinal) = 0;
        virtual void executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t fence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t profiling) = 0;
        virtual ~Impl() {};
    };

    IRGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
          std::optional<ov::Tensor> blob,
          bool blobAllocatedByPlugin,
          const Config& config,
          const ov::SoPtr<ICompiler>& compiler = {nullptr});

    size_t export_blob(std::ostream& stream) const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const Config& config) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize(const Config& config) override;

    ~IRGraph() override;

    void execute(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t inferenceFence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t profiling) override;

private:

    bool release_blob(const Config& config);

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    // In the case of the import path, the blob is released after graph initialization so it can not be any longer
    // exported
    bool _blobIsReleased = false;
    bool _blobAllocatedByPlugin = false;

    const ov::SoPtr<ICompiler> _compiler;
    Logger _logger;

    std::unique_ptr<Impl> _impl;
};

}  // namespace intel_npu
#endif // NPU_LLVM_BACKEND