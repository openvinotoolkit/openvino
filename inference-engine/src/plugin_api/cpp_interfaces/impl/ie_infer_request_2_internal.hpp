// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "cpp_interfaces/impl/ie_variable_state_2_internal.hpp"

namespace InferenceEngine {

class InferRequest2Internal : public IInferRequestInternal {
    InferRequest actual;

public:
    explicit InferRequest2Internal(const InferRequest & request) : actual(request) {
        if (!actual) {
            IE_THROW(NotAllocated);
        }
    }

    void Infer() override {
        actual.Infer();
    }

    virtual void InferImpl() {
        // should call internal methods
        IE_THROW(NotImplemented);
    }

    void Cancel() override {
        actual.Cancel();
    }

    std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const override {
        return actual.GetPerformanceCounts();
    }

    void SetBlob(const std::string& name, const Blob::Ptr& data) override {
        actual.SetBlob(name, data);
    }

    Blob::Ptr GetBlob(const std::string& name) override {
        return actual.GetBlob(name);
    }

    void SetBlob(const std::string& name, const Blob::Ptr& data, const PreProcessInfo& info) override {
        actual.SetBlob(name, data, info);
    }

    const PreProcessInfo& GetPreProcess(const std::string& name) const override {
        return actual.GetPreProcess(name);
    }

    void SetBatch(int batch) override {
        actual.SetBatch(batch);
    }

    std::vector<std::shared_ptr<IVariableStateInternal>> QueryState() override {
        std::vector<IVariableStateInternal::Ptr> states;
        for (auto & state : actual.QueryState()) {
            states.push_back(std::make_shared<VariableState2Internal>(state));
        }
        return states;
    }

    void StartAsync() override {
        actual.StartAsync();
    }

    void StartAsyncImpl() override {
        // should call internal methods
        IE_THROW(NotImplemented);
    }

    StatusCode Wait(int64_t millis_timeout) override {
        return actual.Wait(millis_timeout);
    }

    void SetCallback(Callback callback) override {
        // TODO:
    }

    void checkBlobs() override {
        // should call internal methods
        IE_THROW(NotImplemented);
    }
};

}  // namespace InferenceEngine
