// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vpu/pass_manager.hpp>

#include <memory>

namespace vpu {
namespace {

class PassImpl final : public Pass {
public:
    void run(const Model::Ptr& model) override {
        VPU_PROFILE(initialCheck);

        for (const auto& stage : model->getStages()) {
            stage->initialCheck();
        }
    }
};

}  // namespace

Pass::Ptr PassManager::initialCheck() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
