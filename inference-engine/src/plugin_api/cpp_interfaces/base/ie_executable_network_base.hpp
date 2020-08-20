// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief inference engine executanle network API wrapper, to be used by particular implementors
 * \file ie_executable_network_base.hpp
 */

#pragma once

#include <cpp/ie_executable_network.hpp>
#include <cpp_interfaces/base/ie_memory_state_base.hpp>
#include <cpp_interfaces/interface/ie_imemory_state_internal.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/exception2status.hpp"

namespace InferenceEngine {

/**
 * @brief Executable network `noexcept` wrapper which accepts IExecutableNetworkInternal derived instance which can throw exceptions
 * @ingroup ie_dev_api_exec_network_api
 * @tparam T Minimal CPP implementation of IExecutableNetworkInternal (e.g. ExecutableNetworkInternal)
 */
template <class T>
class ExecutableNetworkBase : public IExecutableNetwork {
    std::shared_ptr<T> _impl;

public:
    /**
     * @brief Constructor with actual underlying implementation.
     * @param impl Underplying implementation of type IExecutableNetworkInternal
     */
    explicit ExecutableNetworkBase(std::shared_ptr<T> impl) {
        if (impl.get() == nullptr) {
            THROW_IE_EXCEPTION << "implementation not defined";
        }
        _impl = impl;
    }

    StatusCode GetOutputsInfo(ConstOutputsDataMap& outs, ResponseDesc* resp) const noexcept override {
        TO_STATUS(outs = _impl->GetOutputsInfo());
    }

    StatusCode GetInputsInfo(ConstInputsDataMap& inputs, ResponseDesc* resp) const noexcept override {
        TO_STATUS(inputs = _impl->GetInputsInfo());
    }

    StatusCode CreateInferRequest(IInferRequest::Ptr& req, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->CreateInferRequest(req));
    }

    StatusCode Export(const std::string& modelFileName, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->Export(modelFileName));
    }

    StatusCode Export(std::ostream& networkModel, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->Export(networkModel));
    }

    StatusCode GetExecGraphInfo(ICNNNetwork::Ptr& graphPtr, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->GetExecGraphInfo(graphPtr));
    }

    StatusCode QueryState(IMemoryState::Ptr& pState, size_t idx, ResponseDesc* resp) noexcept override {
        try {
            auto v = _impl->QueryState();
            if (idx >= v.size()) {
                return OUT_OF_BOUNDS;
            }
            pState = std::make_shared<MemoryStateBase<IMemoryStateInternal>>(v[idx]);
            return OK;
        } catch (const std::exception& ex) {
            return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
        } catch (...) {
            return InferenceEngine::DescriptionBuffer(UNEXPECTED);
        }
    }

    void Release() noexcept override {
        delete this;
    }

    /// @private Need for unit tests only - TODO: unit tests should test using public API, non having details
    const std::shared_ptr<T> getImpl() const {
        return _impl;
    }

    StatusCode SetConfig(const std::map<std::string, Parameter>& config, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetConfig(config, resp));
    }

    StatusCode GetConfig(const std::string& name, Parameter& result, ResponseDesc* resp) const noexcept override {
        TO_STATUS(_impl->GetConfig(name, result, resp));
    }

    StatusCode GetMetric(const std::string& name, Parameter& result, ResponseDesc* resp) const noexcept override {
        TO_STATUS(_impl->GetMetric(name, result, resp));
    }

    StatusCode GetContext(RemoteContext::Ptr& pContext, ResponseDesc* resp) const noexcept override {
        TO_STATUS(_impl->GetContext(pContext, resp));
    }

private:
    ~ExecutableNetworkBase() = default;
};

template <class T>
inline typename InferenceEngine::ExecutableNetwork make_executable_network(std::shared_ptr<T> impl) {
    typename ExecutableNetworkBase<T>::Ptr net(new ExecutableNetworkBase<T>(impl), [](IExecutableNetwork* p) {
        p->Release();
    });
    return InferenceEngine::ExecutableNetwork(net);
}

}  // namespace InferenceEngine
