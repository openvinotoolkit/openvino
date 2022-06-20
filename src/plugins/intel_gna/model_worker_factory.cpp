// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_worker_factory.hpp"

#include "gna_device_interface.hpp"
#include "backend/am_intel_dnn.hpp"
#include "gna2_model_wrapper.hpp"
#include "gna_plugin_log.hpp"
#include "model_subrequest.hpp"
#include "model_worker_impl.hpp"
#include "runtime/gna_float_runtime.hpp"

namespace GNAPluginNS {

constexpr const uint32_t ModelWorkerFactory::kFakeRequestID;
constexpr const uint32_t ModelWorkerFactory::kFakeRequestConfigID;

std::shared_ptr<ModelWorker> ModelWorkerFactory::create_model_worker(std::shared_ptr<Gna2ModelWrapper> model,
                                                                     std::shared_ptr<GNADevice> device,
                                                                     const Gna2AccelerationMode acceleration_mode) {
    return std::make_shared<ModelWorkerImpl>(model,
                                             create_model_subrequests(model, std::move(device), acceleration_mode));
}

std::shared_ptr<ModelWorker> ModelWorkerFactory::create_model_worker_fp32(
    std::shared_ptr<Gna2ModelWrapper> model,
    std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn) {
    return std::make_shared<ModelWorkerImpl>(model, create_model_subrequests_fp32(model, std::move(dnn)));
}

std::shared_ptr<ModelWorker> ModelWorkerFactory::create_model_worker_trivial_topology(
    std::shared_ptr<Gna2ModelWrapper> model) {
    return std::make_shared<ModelWorkerImpl>(std::move(model), create_model_subrequests_trivial());
}

std::vector<ModelSubrequest> ModelWorkerFactory::create_model_subrequests(
    std::shared_ptr<Gna2ModelWrapper> model,
    std::shared_ptr<GNADevice> device,
    const Gna2AccelerationMode acceleration_mode) {
    std::vector<ModelSubrequest> subrequests;
    if (!device) {
        THROW_GNA_EXCEPTION << "device is nullptr";
    }

    uint16_t layers_limit = device->max_layer_count();
    auto slices_number = model->object().NumberOfOperations / layers_limit;
    slices_number += (model->object().NumberOfOperations % layers_limit) ? 1 : 0;
    auto total_operations_number = model->object().NumberOfOperations;

    std::weak_ptr<GNADevice> weak_device = device;

    auto enqueue = [weak_device, acceleration_mode](uint32_t request_config_id) -> uint32_t {
        if (auto device = weak_device.lock()) {
            return device->enqueue_request(request_config_id, acceleration_mode);
        }
        THROW_GNA_EXCEPTION << "device is nullptr";
    };

    auto wait = [weak_device](uint32_t request_id, int64_t timeout_milliseconds) -> GNARequestWaitStatus {
        if (auto device = weak_device.lock()) {
            return device->wait_for_reuqest(request_id, timeout_milliseconds);
        }
        THROW_GNA_EXCEPTION << "device is nullptr";
    };

    for (int i = 0; i < slices_number; ++i) {
        // this models are needed only temporarily to create configurations
        Gna2Model tempModel;
        tempModel.NumberOfOperations =
            (i + 1 < slices_number ? layers_limit : total_operations_number - i * layers_limit);
        tempModel.Operations = &model->object().Operations[i * layers_limit];
        const auto modelId = device->create_model(tempModel);
        const auto requestConfigId = device->create_request_config(modelId);

        subrequests.emplace_back(requestConfigId, enqueue, wait);
    }
    return subrequests;
}

std::vector<ModelSubrequest> ModelWorkerFactory::create_model_subrequests_fp32(
    std::shared_ptr<Gna2ModelWrapper> model,
    std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn) {
    std::vector<ModelSubrequest> subrequests;

    if (!dnn) {
        THROW_GNA_EXCEPTION << "dnn is nullptr";
    }

    std::weak_ptr<GNAPluginNS::backend::AMIntelDNN> weak_dnn = dnn;

    auto enque_fp32 = [weak_dnn](uint32_t config_id) -> uint32_t {
        if (auto dnn = weak_dnn.lock()) {
            auto runtime = runtime::FP(dnn);
            runtime.infer();
            return kFakeRequestID;
        }
        // maybe warning would be enough
        THROW_GNA_EXCEPTION << "dnn is nullptr";
    };

    auto wait_simple = [](uint32_t, int64_t timeout_miliseconds) {
        return GNARequestWaitStatus::kCompleted;
    };

    subrequests.emplace_back(kFakeRequestConfigID, std::move(enque_fp32), std::move(wait_simple));
    return subrequests;
}

std::vector<ModelSubrequest> ModelWorkerFactory::create_model_subrequests_trivial() {
    std::vector<ModelSubrequest> subrequests;

    auto enque_simple = [](uint32_t config_id) {
        return kFakeRequestID;
    };

    auto wait_simple = [](uint32_t, int64_t timeout_miliseconds) {
        return GNARequestWaitStatus::kCompleted;
    };

    subrequests.emplace_back(kFakeRequestConfigID, std::move(enque_simple), std::move(wait_simple));
    return subrequests;
}

}  // namespace GNAPluginNS