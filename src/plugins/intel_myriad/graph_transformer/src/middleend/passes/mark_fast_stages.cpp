// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override {
        VPU_PROFILE(markFastStages);

        for (const auto& stage : model->getStages()) {
            const auto& outputs = stage->outputs();
            const auto isStageFast = std::all_of(outputs.begin(), outputs.end(), [](const Data& output) {
                return output->desc().totalDimSize() <= 100;
            });

            if (isStageFast) {
                stage->appendNamePostfix("@fast-stage");
            }
        }
    }
};

}  // namespace

Pass::Ptr PassManager::markFastStages() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
