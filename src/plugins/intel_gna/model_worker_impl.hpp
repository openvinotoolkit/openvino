// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-model-api.h>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "model_worker.hpp"

namespace GNAPluginNS {
class Gna2ModelWrapper;
class ModelSubrequest;

class ModelWorkerImpl : public ModelWorker {
public:
    explicit ModelWorkerImpl(std::shared_ptr<Gna2ModelWrapper> model, std::vector<ModelSubrequest> model_subrequests);

    ModelWorkerImpl(const ModelWorkerImpl&) = delete;
    ModelWorkerImpl(ModelWorkerImpl&&) = delete;
    ModelWorkerImpl& operator=(const ModelWorkerImpl&) = delete;
    ModelWorkerImpl& operator=(ModelWorkerImpl&&) = delete;

    const Gna2Model* model() const override;
    Gna2Model* model() override;

    void enqueue_request() override;

    RequestStatus wait(int64_t timeout_miliseconds) override;

    bool is_free() const override;

    uint32_t representing_index() const override;

    void set_representing_index(uint32_t index) override;

    void set_result(const InferenceEngine::BlobMap& result) override;
    void set_result(InferenceEngine::BlobMap&& result) override;

    InferenceEngine::BlobMap& result() override;

private:
    void check_if_free();

    uint32_t representing_index_{0};
    std::shared_ptr<Gna2ModelWrapper> full_model_;
    std::vector<ModelSubrequest> model_subrequests_;
    InferenceEngine::BlobMap request_result_;
};

}  // namespace GNAPluginNS
