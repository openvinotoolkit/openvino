// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/icompiler.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "npu_mlir_runtime_api.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {
class IRGraph final : public IGraph {
public:
    struct MemRefType {
        npu_mlir_runtime_mem_ref_handle_t memRef;
        const void* basePtr;
        const void* data;
        int64_t offset;
        std::vector<int64_t> sizes;
        std::vector<int64_t> strides;
        int64_t dimsCount;

        MemRefType()
            : memRef(nullptr),
              basePtr(nullptr),
              data(nullptr),
              offset(0),
              sizes({}),
              strides({}),
              dimsCount(0) {}

        MemRefType(const void* basePtr,
                   const void* data,
                   int64_t offset,
                   const std::vector<int64_t>& sizes,
                   const std::vector<int64_t>& strides,
                   int64_t dimsCount)
            : memRef(nullptr),
              basePtr(basePtr),
              data(data),
              offset(offset),
              sizes(sizes),
              strides(strides),
              dimsCount(dimsCount) {}

        MemRefType(const MemRefType& other)
            : memRef(nullptr),
              basePtr(other.basePtr),
              data(other.data),
              offset(other.offset),
              sizes(other.sizes),
              strides(other.strides),
              dimsCount(other.dimsCount) {}

        MemRefType(MemRefType&& other) noexcept
            : memRef(nullptr),
              basePtr(other.basePtr),
              data(other.data),
              offset(other.offset),
              sizes(std::move(other.sizes)),
              strides(std::move(other.strides)),
              dimsCount(other.dimsCount) {}

        MemRefType& operator=(const MemRefType& other) {
            if (this != &other) {
                basePtr = other.basePtr;
                data = other.data;
                offset = other.offset;
                sizes = other.sizes;
                strides = other.strides;
                dimsCount = other.dimsCount;
            }
            return *this;
        }

        MemRefType& operator=(MemRefType&& other) noexcept {
            if (this != &other) {
                basePtr = other.basePtr;
                data = other.data;
                offset = other.offset;
                sizes = std::move(other.sizes);
                strides = std::move(other.strides);
                dimsCount = other.dimsCount;
            }
            return *this;
        }

        ~MemRefType() {
            destroyMemRef();
        }

        void setArg(const void* arg);
        void setSize(const intel_npu::IODescriptor& descriptor);
        void updateStride();

        friend std::ostream& operator<<(std::ostream& os, const MemRefType& memRef) {
            os << "BasePtr: " << memRef.basePtr << ", Data: " << memRef.data << ", Offset: " << memRef.offset
               << ", Sizes: [";
            for (int64_t size : memRef.sizes) {
                os << size << " ";
            }
            os << "], Strides: [";
            for (int64_t stride : memRef.strides) {
                os << stride << " ";
            }
            os << "]";
            return os;
        }

        std::string toString() {
            std::stringstream stream;
            stream << *this;
            return stream.str();
        }

        void UpdateMemRefHandleStatus() {
            // Update current MemRef handle to use latest metadata
            if (memRef == nullptr) {
                createMemRef();
            } 
            auto result =
                npuMLIRRuntimeSetMemRef(memRef, basePtr, data, offset, sizes.data(), strides.data(), dimsCount);
            if (result != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
                throw std::runtime_error("Failed to update MemRef handle");
            }
        }

	void alignWithHandle();

    private:
        void createMemRef() {
            if (memRef == nullptr) {
                auto result = npuMLIRRuntimeCreateMemRef(dimsCount, &memRef);
                if (result != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
                    throw std::runtime_error("Failed to create MemRef handle");
                }
            }
        }

        void destroyMemRef() {
            if (memRef != nullptr) {
                npuMLIRRuntimeDestroyMemRef(memRef);
                memRef = nullptr;
            }
        }
    };

    struct GraphArguments {
        std::vector<MemRefType> _inputs;
        std::vector<MemRefType> _outputs;
        std::vector<npu_mlir_runtime_mem_ref_handle_t> _inputMemRefs;
        std::vector<npu_mlir_runtime_mem_ref_handle_t> _outputMemRefs;
        npu_mlir_runtime_execute_params_t _executeParams = {};
        Logger _logger = Logger("GraphArguments", Logger::global().level());

        void setArgumentValue(uint32_t argi, const void* argv);
        void setArgumentProperties(uint32_t argi,
                                   const void* argv,
                                   const ov::Shape& shapes,
                                   const std::vector<size_t>& strides);
    };

    class Impl {
        using MemRefType = IRGraph::MemRefType;

    public:
        virtual void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) = 0;
        virtual void setArgumentValue(uint32_t argi, const void* argv) = 0;
        virtual void setArgumentValueWithStrides(uint32_t argi,
                                                 const void* argv,
                                                 const std::vector<size_t>& strides) = 0;
        virtual uint64_t getNumSubgraphs() = 0;
        virtual void initializeGraph(uint64_t command_queue_group_ordinal) = 0;
        virtual void getBinding(GraphArguments& binding) = 0;
        virtual void executeGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                                  GraphArguments& args,
                                  std::vector<ze_command_list_handle_t>& commandLists,
                                  ze_command_queue_handle_t commandQueue,
                                  ze_fence_handle_t fence,
                                  ze_event_handle_t event,
                                  ze_graph_profiling_pool_handle_t profiling) = 0;
        virtual void predictOutputShape(std::vector<MemRefType>& inputDescriptors,
                                        std::vector<MemRefType>& outputDescriptors) = 0;
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

    void set_argument_value_with_strides(uint32_t id,
                                         const void* data,
                                         const std::vector<size_t>& strides) const override;

    ze_graph_handle_t get_handle() const override;
    BlobType get_blob_type() override {
        return BlobType::LLVM;
    }

    void initialize(const Config& config) override;

    ~IRGraph() override;

    const NetworkMetadata& get_metadata() const override;

    void update_network_name(std::string_view name) override;

    const std::shared_ptr<CommandQueue>& get_command_queue() const override;
    uint32_t get_command_queue_group_ordinal() const override;

    void set_workload_type(const ov::WorkloadType workloadType) const override;

    void set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) override;
    const std::shared_ptr<Event>& get_last_submitted_event(size_t indexOfCommandList) const override;
    void resize_last_submitted_event(size_t batch) override;
    void set_batch_size(std::size_t batch) override;

    const std::optional<std::size_t> get_batch_size() const override;

    uint32_t get_unique_id() override;
    void set_last_submitted_id(uint32_t id_index) override;
    uint32_t get_last_submitted_id() const override;

    void execute(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                 GraphArguments& args,
                 std::vector<ze_command_list_handle_t>& commandLists,
                 ze_command_queue_handle_t commandQueue,
                 ze_fence_handle_t inferenceFence,
                 ze_event_handle_t event,
                 ze_graph_profiling_pool_handle_t profiling);

    void getBinding(GraphArguments& args);

    uint64_t get_num_subgraphs() const;

    void predict_output_shape(std::vector<MemRefType>& inputDescriptors, std::vector<MemRefType>& outputDescriptors);

private:
    bool release_blob(const Config& config);
    std::optional<size_t> determine_batch_size();

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    NetworkMetadata _metadata;

    /**
     * @brief Stores the number of subgraphs for dynamic models
     * @note the number of subgraphs will be one for static models
     */
    uint64_t _num_of_subgraphs = 1;

    std::shared_ptr<CommandQueue> _commandQueue;
    uint32_t _commandQueueGroupOrdinal = 0;
    std::vector<std::shared_ptr<Event>> _lastSubmittedEvent;

    std::optional<ov::Tensor> _blob;

    // In the case of the import path, the blob is released after graph initialization so it can not be any longer
    // exported
    bool _blobIsReleased = false;
    bool _blobAllocatedByPlugin = false;

    uint32_t _uniqueId = 0;
    uint32_t _lastSubmittedId = 0;

    /**
     * @brief The batch size used by the corresponding model.
     * @details The attribute contains a value only if the plugin performs the batches splitting operation.
     */
    std::optional<std::size_t> _batchSize = std::nullopt;

    const ov::SoPtr<ICompiler> _compiler;
    Logger _logger;

    std::unique_ptr<Impl> _impl;
};

}  // namespace intel_npu
