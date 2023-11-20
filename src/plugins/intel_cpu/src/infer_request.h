// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.h"
#include "cpu_tensor.h"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "memory_state.h"

namespace ov {
namespace intel_cpu {

class CompiledModel;
class AsyncInferRequest;

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    SyncInferRequest(std::shared_ptr<const CompiledModel> compiled_model);
    virtual ~SyncInferRequest();

    void infer() override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& _port) const override;

    /**
     * @brief      Sets the pointer to asynchronous inference request that holds this request
     * @param[in]  asyncRequest Pointer to asynchronous inference request
     */
    void set_async_request(AsyncInferRequest* asyncRequest);

    /**
     * @brief If `m_asyncRequest` is initialized throw exception with `ov::Cancelled` status if inference request is
     * canceled
     */

    void throw_if_canceled() const;

private:
    class OutputControlBlock {
    public:
        using MemMngrPtr = std::shared_ptr<MemoryMngrWithReuse>;

    public:
        OutputControlBlock(const ov::element::Type& precision, const Shape& shape);

        OutputControlBlock(const OutputControlBlock&) = delete;
        OutputControlBlock& operator=(const OutputControlBlock&) = delete;

        OutputControlBlock(OutputControlBlock&&) = default;
        OutputControlBlock& operator=(OutputControlBlock&&) = default;

        std::shared_ptr<Tensor> tensor() const {
            return m_tensor;
        }

        const void* rawPtr() const {
            return m_tensor->get_memory()->getData();
        }

        MemMngrPtr currentMemMngr() const {
            return m_buffers[m_buffIndx];
        }

        MemMngrPtr nextMemMngr() {
            m_buffIndx ^= 0x1;
            if (!m_buffers[m_buffIndx]) {
                m_buffers[m_buffIndx] = std::make_shared<MemoryMngrWithReuse>();
            }
            return m_buffers[m_buffIndx];
        }

        void update() {
            m_proxyMemMngr->setMemMngrResize(currentMemMngr());
        }

    private:
        std::shared_ptr<Tensor> m_tensor = nullptr;
        ProxyMemoryMngrPtr m_proxyMemMngr = nullptr;
        std::array<MemMngrPtr, 2> m_buffers;
        int m_buffIndx = 0;
    };

private:
    void create_infer_request();
    void init_tensor(const std::string& name);

    void push_input_data();
    void redefine_memory_for_input_nodes();
    void assign_states();
    void commit_states();
    void update_external_tensor_ptrs();
    void change_default_ptr();

    const ov::Output<const ov::Node>& get_internal_port(const ov::Output<const ov::Node>& port) const;

private:
    std::unordered_map<std::string, OutputControlBlock> m_outputControlBlocks;

    Graph* m_graph = nullptr;
    std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_external_ptr;

    bool m_is_legacy_api = false;

    std::shared_ptr<const CompiledModel> m_compiled_model;
    openvino::itt::handle_t m_profiling_task;
    std::vector<MemStatePtr> m_memory_states;
    AsyncInferRequest* m_asyncRequest = nullptr;

    std::unordered_map<std::string, ov::Output<const ov::Node>> m_input_ports_map;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_output_ports_map;
    std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_outputs;
};

}  // namespace intel_cpu
}  // namespace ov
