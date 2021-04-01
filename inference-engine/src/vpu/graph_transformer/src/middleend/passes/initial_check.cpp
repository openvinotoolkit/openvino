// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <memory>

namespace vpu {
namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override {
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
