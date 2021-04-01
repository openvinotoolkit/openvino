// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/pass_manager.hpp"

#include <set>
#include <memory>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model& model) override {
        AttributesMap* currentLoopAttributes = nullptr;
        for (const auto& stage : model->getStages()) {
            if (stage->type() == StageType::LoopStart) {
                VPU_THROW_UNLESS(currentLoopAttributes == nullptr, "Nested loops are not supported yet");
                VPU_THROW_UNLESS(!stage->attrs().has("stages-count"), "The same LoopStart must not be visited twice");

                currentLoopAttributes = &stage->attrs();
                currentLoopAttributes->set<uint32_t>("stages-count", 1);
            } else if (stage->type() == StageType::LoopEnd) {
                VPU_THROW_UNLESS(currentLoopAttributes != nullptr && currentLoopAttributes->has("stages-count"), "Loop Start must be already encountered");

                auto& stagesCount = currentLoopAttributes->get<uint32_t>("stages-count");
                stagesCount++;
                currentLoopAttributes = nullptr;
            } else if (currentLoopAttributes != nullptr) {
                VPU_THROW_UNLESS(currentLoopAttributes->has("stages-count"), "Loop Start must be counted as a stage in the loop");

                if (static_cast<int>(stage->type()) >= 0) {
                    auto& stagesCount = currentLoopAttributes->get<uint32_t>("stages-count");
                    stagesCount++;
                }
            }
        }
    }
};

}  // namespace

Pass::Ptr PassManager::countStagesInLoops() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
