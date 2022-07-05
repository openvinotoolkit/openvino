// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-inference-api.h>

#include <memory>

#include "worker.hpp"

namespace GNAPluginNS {
class GNADevice;

namespace backend {
class AMIntelDNN;
}

namespace request {

class ModelWrapper;
class Subrequest;

class WorkerFactory {
public:
    WorkerFactory() = delete;

    static std::shared_ptr<Worker> create_model_worker(std::shared_ptr<ModelWrapper> model,
                                                       std::shared_ptr<GNADevice> device,
                                                       const Gna2AccelerationMode acceleration_mode);
    static std::shared_ptr<Worker> create_model_worker_fp32(std::shared_ptr<ModelWrapper> model,
                                                            std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn);
    static std::shared_ptr<Worker> create_model_worker_trivial_topology(std::shared_ptr<ModelWrapper> model);

private:
    static std::vector<Subrequest> create_model_subrequests(std::shared_ptr<ModelWrapper> model,
                                                            std::shared_ptr<GNADevice> device,
                                                            const Gna2AccelerationMode acceleration_mode);
    static std::vector<Subrequest> create_model_subrequests_fp32(std::shared_ptr<ModelWrapper> model,
                                                                 std::shared_ptr<backend::AMIntelDNN> dnn);
    static std::vector<Subrequest> create_model_subrequests_trivial();

    static constexpr const uint32_t kFakeRequestID{1};
};

}  // namespace request
}  // namespace GNAPluginNS
