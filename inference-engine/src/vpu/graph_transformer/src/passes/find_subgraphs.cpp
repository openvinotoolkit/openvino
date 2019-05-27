// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <cmath>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

#include <vpu/stub_stage.hpp>
#include <vpu/sw/utility.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(findSubGraphs);

    const auto& env = CompileEnv::get();
    auto stages = model->getStages();
    int maxClasses = 0;
    int currentCount = 0;
    for (const auto& stage : stages) {
        if (currentCount >= env.config.numberOfNodesInOneSubGraph) {
            currentCount = 0;
            maxClasses++;
        }
        stage->setSubGraphNumber(maxClasses);
        currentCount++;
    }
    model->setNumberOfSubGraphs(maxClasses + 1);
}

}  // namespace

Pass::Ptr PassManager::findSubGraphs() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
