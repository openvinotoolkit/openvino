// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "worker_factory.hpp"

#include "backend/am_intel_dnn.hpp"
#include "gna_device_interface.hpp"
#include "gna_plugin_log.hpp"
#include "request/model_wrapper.hpp"
#include "runtime/gna_float_runtime.hpp"
#include "subrequest.hpp"
#include "worker_impl.hpp"

namespace GNAPluginNS {
namespace request {

constexpr const uint32_t WorkerFactory::kFakeRequestID;

std::shared_ptr<Worker> WorkerFactory::createWorker(std::shared_ptr<ModelWrapper> model,
                                                    std::shared_ptr<GNADevice> device,
                                                    const Gna2AccelerationMode accelerationMode) {
    return std::make_shared<WorkerImpl>(model, createModelSubrequests(model, std::move(device), accelerationMode));
}

std::shared_ptr<Worker> WorkerFactory::createWorkerFP32(std::shared_ptr<ModelWrapper> model,
                                                        std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn) {
    return std::make_shared<WorkerImpl>(model, createModelSubrequestsFP32(model, std::move(dnn)));
}

std::shared_ptr<Worker> WorkerFactory::createWorkerTrivialTopology(std::shared_ptr<ModelWrapper> model) {
    return std::make_shared<WorkerImpl>(std::move(model), createModelSubrequestsTrivial());
}

std::vector<Subrequest> WorkerFactory::createModelSubrequests(std::shared_ptr<ModelWrapper> model,
                                                              std::shared_ptr<GNADevice> device,
                                                              const Gna2AccelerationMode accelerationMode) {
    std::vector<Subrequest> subrequests;
    if (!device) {
        THROW_GNA_EXCEPTION << "device is nullptr";
    }

    uint16_t layersLimit = device->maxLayersCount();
    auto submodelsNumber = model->object().NumberOfOperations / layersLimit;
    submodelsNumber += (model->object().NumberOfOperations % layersLimit) ? 1 : 0;
    auto totalOperationsNumber = model->object().NumberOfOperations;

    std::weak_ptr<GNADevice> weakDevice = device;

    auto wait = [weakDevice](uint32_t requestID, int64_t timeoutMilliseconds) -> RequestStatus {
        if (auto device = weakDevice.lock()) {
            return device->waitForRequest(requestID, timeoutMilliseconds);
        }
        THROW_GNA_EXCEPTION << "device is nullptr";
    };

    for (int i = 0; i < submodelsNumber; ++i) {
        // this models are needed only temporarily to create configurations
        Gna2Model tempModel;
        tempModel.NumberOfOperations =
            (i + 1 < submodelsNumber ? layersLimit : totalOperationsNumber - i * layersLimit);
        tempModel.Operations = &model->object().Operations[i * layersLimit];
        const auto modelID = device->createModel(tempModel);
        const auto requestConfigID = device->createRequestConfig(modelID);

        auto enqueue = [weakDevice, requestConfigID, accelerationMode]() -> uint32_t {
            if (auto device = weakDevice.lock()) {
                return device->enqueueRequest(requestConfigID, accelerationMode);
            }
            THROW_GNA_EXCEPTION << "device is nullptr";
        };

        subrequests.emplace_back(std::move(enqueue), wait);
    }
    return subrequests;
}

std::vector<Subrequest> WorkerFactory::createModelSubrequestsFP32(
    std::shared_ptr<ModelWrapper> model,
    std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn) {
    std::vector<Subrequest> subrequests;

    if (!dnn) {
        THROW_GNA_EXCEPTION << "dnn is nullptr";
    }

    std::weak_ptr<GNAPluginNS::backend::AMIntelDNN> weak_dnn = dnn;

    auto enqueFP32 = [weak_dnn]() -> uint32_t {
        if (auto dnn = weak_dnn.lock()) {
            auto runtime = runtime::FP(dnn);
            runtime.infer();
            return kFakeRequestID;
        }
        // maybe warning would be enough
        THROW_GNA_EXCEPTION << "dnn is nullptr";
    };

    auto waitSimple = [](uint32_t, int64_t) {
        return RequestStatus::kCompleted;
    };

    subrequests.emplace_back(std::move(enqueFP32), std::move(waitSimple));
    return subrequests;
}

std::vector<Subrequest> WorkerFactory::createModelSubrequestsTrivial() {
    std::vector<Subrequest> subrequests;

    auto enqueSimple = []() {
        return kFakeRequestID;
    };

    auto waitSimple = [](uint32_t, int64_t) {
        return RequestStatus::kCompleted;
    };

    subrequests.emplace_back(std::move(enqueSimple), std::move(waitSimple));
    return subrequests;
}

}  // namespace request
}  // namespace GNAPluginNS