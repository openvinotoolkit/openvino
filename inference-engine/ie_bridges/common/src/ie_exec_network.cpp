#include "helpers.h"
#include "ie_exec_network.h"

InferenceEngineBridge::IEExecNetwork::IEExecNetwork(const std::string &name, size_t num_requests) :
        infer_requests(num_requests), name(name) {
}

InferenceEngineBridge::IENetwork InferenceEngineBridge::IEExecNetwork::GetExecGraphInfo() {
    InferenceEngine::ResponseDesc response;
    InferenceEngine::ICNNNetwork::Ptr graph;
    IE_CHECK_CALL(this->exec_network_ptr->GetExecGraphInfo(graph, &response));
    return IENetwork(InferenceEngine::CNNNetwork(graph));
}