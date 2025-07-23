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

namespace intel_npu {

#ifdef NPU_LLVM_BACKEND
class IRGraph final : public IGraph {
public:
    struct MemRefType {
        const void *basePtr;
        const void *data;
        int64_t offset;
        int64_t sizes[4];
        int64_t strides[4];
        MemRefType() {
            basePtr = nullptr;
            data = nullptr;
            offset = 0;
        }

        void setArg(const void* arg);
        void setSize(const intel_npu::IODescriptor& descriptor);
        void updateStride();
    };

    struct GraphArguments {
        std::vector<MemRefType*> _inputs;
        std::vector<MemRefType*> _outputs;

        GraphArguments() = default;
        GraphArguments(const GraphArguments& args);
        GraphArguments& operator=(const GraphArguments& args);
        ~GraphArguments();
    };

    class Impl {
        using MemRefType = IRGraph::MemRefType;

    public:
        virtual void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata, std::vector<ArgumentDescriptor>& inputs, std::vector<ArgumentDescriptor>& outputs) = 0;
        virtual void setArgumentValue(uint32_t argi, const void* argv) = 0;
        virtual uint64_t getNumSubgraphs() = 0;
        virtual void initializeGraph(uint64_t command_queue_group_ordinal) = 0;
        virtual void getBinding(GraphArguments& binding) = 0;
        virtual void executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, GraphArguments& args, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t fence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t profiling) = 0;
        virtual ~Impl() {};
    };

    IRGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
          std::optional<ov::Tensor> blob,
          bool blobAllocatedByPlugin,
          const Config& config,
          const ov::SoPtr<ICompiler>& compiler = {nullptr});

    std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const Config& config) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize(const Config& config) override;

    ~IRGraph() override;

    void execute(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct, GraphArguments& args, std::vector<ze_command_list_handle_t>& commandLists, ze_command_queue_handle_t commandQueue, ze_fence_handle_t inferenceFence, ze_event_handle_t event, ze_graph_profiling_pool_handle_t profiling);

    void getBinding(GraphArguments& args);
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

#endif // NPU_LLVM_BACKEND

inline bool is_dynamic_shape_blob( const ov::Tensor& blob ) {
    // TODO: A way to detect if the blob is ELF or IR, check if first 20 bytes has 'ELF' string
    // Check If blob is ELF, if not, create Graph for LLVM IR

    size_t blobSize = blob.get_byte_size();
    // Temporarily use 20 as header length
    size_t headerSize = blobSize > 20 ? 20 : blobSize;
    std::string header(reinterpret_cast<const char*>(blob.data()), headerSize);

    return (header.find("ELF") == std::string::npos);
}

}  // namespace intel_npu