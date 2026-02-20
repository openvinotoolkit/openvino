// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include "compiler_impl.hpp"
#include "intel_npu/common/idynamic_graph.hpp"
#include "intel_npu/network_metadata.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "npu_mlir_runtime_api.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {
class DynamicGraph final : public IDynamicGraph {
public:
    struct MemRefTypeImpl {
        npu_mlir_runtime_mem_ref_handle_t _memRef;

        MemRefTypeImpl() : _memRef(nullptr) {}

        ~MemRefTypeImpl() {
            destroyMemRef();
        }

        void UpdateMemRefHandleStatus(MemRefType& memref) {
            // Update current MemRef handle to use latest metadata
            if (_memRef == nullptr) {
                createMemRef(memref._dimsCount);
            }
            auto result = npuMLIRRuntimeSetMemRef(_memRef,
                                                  memref._basePtr,
                                                  memref._data,
                                                  memref._offset,
                                                  memref._sizes.data(),
                                                  memref._strides.data(),
                                                  memref._dimsCount);
            if (result != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
                throw std::runtime_error("Failed to update MemRef handle");
            }
        }

        void alignWithHandle(MemRefType& memref) {
            if (_memRef == nullptr) {
                return;
            }

            if (npuMLIRRuntimeParseMemRef(_memRef,
                                          &memref._basePtr,
                                          &memref._data,
                                          &memref._offset,
                                          memref._sizes.data(),
                                          memref._strides.data(),
                                          &memref._dimsCount) != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
                throw std::runtime_error("Failed to parse MemRef handle");
            }
        }

    private:
        void createMemRef(int64_t dimsCount) {
            if (_memRef == nullptr) {
                auto result = npuMLIRRuntimeCreateMemRef(dimsCount, &_memRef);
                if (result != NPU_MLIR_RUNTIME_RESULT_SUCCESS) {
                    OPENVINO_THROW("Failed to create MemRef handle");
                }
            }
        }

        void destroyMemRef() {
            if (_memRef != nullptr) {
                npuMLIRRuntimeDestroyMemRef(_memRef);
                _memRef = nullptr;
            }
        }
    };

    struct GraphArgumentsImpl : public GraphArguments {
        std::vector<npu_mlir_runtime_mem_ref_handle_t> _inputMemRefs;
        std::vector<npu_mlir_runtime_mem_ref_handle_t> _outputMemRefs;
        npu_mlir_runtime_execute_params_t _executeParams = {};
    };

    class Impl {
        using MemRefTypeImpl = DynamicGraph::MemRefTypeImpl;

    public:
        virtual void initialize(std::optional<ov::Tensor>& blob, NetworkMetadata& metadata) = 0;
        virtual void setArgumentValue(uint32_t argi, const void* argv) = 0;
        virtual void setArgumentValueWithStrides(uint32_t argi,
                                                 const void* argv,
                                                 const std::vector<size_t>& strides) = 0;
        virtual uint64_t getNumSubgraphs() = 0;
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

    DynamicGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                 std::optional<ov::Tensor> blob,
                 bool blobAllocatedByPlugin,
                 const Config& config,
                 const ov::SoPtr<VCLCompilerImpl>& compiler = {nullptr});

    std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const Config& config) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void set_argument_value_with_strides(uint32_t id,
                                         const void* data,
                                         const std::vector<size_t>& strides) const override;

    ze_graph_handle_t get_handle() const override;

    void initialize(const Config& config) override;

    ~DynamicGraph() override;

    const NetworkMetadata& get_metadata() const override;

    void update_network_name(std::string_view name) override;

    const std::shared_ptr<CommandQueue>& get_command_queue() const override;
    uint32_t get_command_queue_group_ordinal() const override;

    void set_workload_type(const ov::WorkloadType workloadType) const override;

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
                 ze_graph_profiling_pool_handle_t profiling) override;

    void getBinding(GraphArguments& args) override;

    uint64_t get_num_subgraphs() const override;

    void predict_output_shape(std::vector<MemRefType>& inputDescriptors,
                              std::vector<MemRefType>& outputDescriptors) override;

    std::optional<bool> is_profiling_blob() const override;

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

    const ov::SoPtr<VCLCompilerImpl> _compiler;
    Logger _logger;

    std::unique_ptr<Impl> _impl;
};

}  // namespace intel_npu
