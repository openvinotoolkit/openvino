// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-inference-api.h>

#include <memory>

#include "model_worker.hpp"

namespace GNAPluginNS {

namespace backend {
class AMIntelDNN;
}

class Gna2ModelWrapper;
class ModelSubrequest;

class ModelWorkerFactory {
public:
    ModelWorkerFactory() = delete;

    static std::shared_ptr<ModelWorker> create_model_worker(std::shared_ptr<Gna2ModelWrapper> model,
                                                            std::shared_ptr<GNADevice> device,
                                                            const Gna2AccelerationMode acceleration_mode);
    static std::shared_ptr<ModelWorker> create_model_worker_fp32(std::shared_ptr<Gna2ModelWrapper> model,
                                                                 std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn);
    static std::shared_ptr<ModelWorker> create_model_worker_trivial_topology(std::shared_ptr<Gna2ModelWrapper> model);

private:
    static std::vector<ModelSubrequest> create_model_subrequests(std::shared_ptr<Gna2ModelWrapper> model,
                                                                 std::shared_ptr<GNADevice> device,
                                                                 const Gna2AccelerationMode acceleration_mode);
    static std::vector<ModelSubrequest> create_model_subrequests_fp32(
        std::shared_ptr<Gna2ModelWrapper> model,
        std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn);
    static std::vector<ModelSubrequest> create_model_subrequests_trivial();

    static constexpr const uint32_t kFakeRequestID{1};
    static constexpr const uint32_t kFakeRequestConfigID{0xffffffff};
};

}  // namespace GNAPluginNS
