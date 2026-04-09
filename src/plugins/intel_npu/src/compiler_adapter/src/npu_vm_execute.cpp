// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_vm_execute.hpp"

namespace intel_npu {

void vm_execute_graph(npu_vm_runtime_handle_t engine,
                      const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                      IDynamicGraph::GraphArguments& args,
                      std::vector<ze_command_list_handle_t>& commandLists,
                      ze_command_queue_handle_t commandQueue,
                      ze_fence_handle_t fence,
                      ze_event_handle_t event) {
    // On the first inference, build the cached handle vectors.
    // On subsequent inferences the _impl is reused and only pointer values
    // inside the existing handles are refreshed.
    std::shared_ptr<GraphArgumentsImpl> argsImpl =
        args._impl ? std::static_pointer_cast<GraphArgumentsImpl>(args._impl)
                   : std::make_shared<GraphArgumentsImpl>();

    npu_vm_runtime_execute_params_t* params = &argsImpl->_executeParams;

    for (auto& in : args._inputs) {
        auto inImpl = std::static_pointer_cast<MemRefTypeImpl>(in._impl);
        if (inImpl == nullptr) {
            inImpl = std::make_shared<MemRefTypeImpl>();
            in._impl = inImpl;
        }
        inImpl->UpdateMemRefHandleStatus(in);
        if (args._impl == nullptr) {
            argsImpl->_inputMemRefs.push_back(inImpl->_memRef);
        }
    }

    for (auto& out : args._outputs) {
        auto outImpl = std::static_pointer_cast<MemRefTypeImpl>(out._impl);
        if (outImpl == nullptr) {
            outImpl = std::make_shared<MemRefTypeImpl>();
            out._impl = outImpl;
        }
        outImpl->UpdateMemRefHandleStatus(out);
        if (args._impl == nullptr) {
            argsImpl->_outputMemRefs.push_back(outImpl->_memRef);
        }
    }

    params->pInputs = argsImpl->_inputMemRefs.data();
    params->numOfInputs = static_cast<uint32_t>(argsImpl->_inputMemRefs.size());
    params->pOutputs = argsImpl->_outputMemRefs.data();
    params->numOfOutputs = static_cast<uint32_t>(argsImpl->_outputMemRefs.size());
    params->ctx = zeroInitStruct->getContext();
    params->device = zeroInitStruct->getDevice();
    params->graphDdiTableExt = zeroInitStruct->getGraphDdiTable().getImpl();
    params->commandLists = commandLists.data();
    params->numCommandLists = static_cast<uint64_t>(commandLists.size());
    params->commandQueue = commandQueue;
    params->inferenceFence = fence;
    params->event = event;

    if (npuVMRuntimeExecute(engine, params) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to execute VM runtime engine");
    }

    if (args._impl == nullptr) {
        args._impl = argsImpl;
    }
}

}  // namespace intel_npu
