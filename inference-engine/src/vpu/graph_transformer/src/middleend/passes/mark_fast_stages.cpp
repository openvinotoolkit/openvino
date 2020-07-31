// Copyright (C) 2020 Intel Corporation
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
            bool isStageFast = true;

            for (const auto& output : stage->outputs()) {
                if (output->desc().totalDimSize() > 100) {
                    isStageFast = false;
                    break;
                }
            }

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
