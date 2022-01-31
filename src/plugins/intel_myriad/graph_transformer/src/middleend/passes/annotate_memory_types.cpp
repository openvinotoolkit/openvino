// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/pass_manager.hpp"

namespace vpu {

namespace {

template<class DataObjects>
std::vector<vpu::MemoryType> gather(const DataObjects& dataObjects) {
    std::vector<vpu::MemoryType> types;
    types.reserve(dataObjects.size());
    std::transform(dataObjects.begin(), dataObjects.end(), std::back_inserter(types), [](const Data& data) { return data->memReqs(); });
    return types;
}

class PassImpl final : public Pass {
public:
    void run(const Model& model) override {
        for (const auto& stage : model->getStages()) {
            std::stringstream suffix;
            suffix << "@";
            printTo(suffix, gather(stage->inputs()));
            suffix << "->";
            printTo(suffix, gather(stage->outputs()));
            stage->appendNamePostfix(suffix.str());
        }
    }
};

}  // namespace

Pass::Ptr PassManager::annotateMemoryTypes() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
