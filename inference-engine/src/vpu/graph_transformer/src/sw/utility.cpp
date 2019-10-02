// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/sw/utility.hpp>

#include <memory>
#include <unordered_set>

#include <vpu/model/model.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

//
// DefaultSwWeightsContent
//

DefaultSwWeightsContent::DefaultSwWeightsContent(const DataContent::Ptr& origContent) :
        CalculatedDataContent({origContent}) {
}

void DefaultSwWeightsContent::fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const {
    VPU_PROFILE(DefaultSwWeightsContent);

    IE_ASSERT(desc().type() == DataType::FP16);
    IE_ASSERT(baseContents.size() == 1);

    kchw_to_hwck(baseContents[0]->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), desc());
}

//
// getNextStage
//

Stage getNextStage(
        const Stage& curStage,
        const std::unordered_set<StageType, EnumClassHash>& supportedTypes) {
    IE_ASSERT(curStage->numOutputs() == 1);

    auto output = curStage->output(0);

    IE_ASSERT(output->parentData() == nullptr);
    IE_ASSERT(output->numChildDatas() == 0);

    if (output->usage() != DataUsage::Intermediate) {
        return nullptr;
    }

    if (output->numConsumers() != 1) {
        return nullptr;
    }

    auto consumer = output->singleConsumer();
    if (supportedTypes.count(consumer->type()) != 0) {
        return consumer;
    }

    return nullptr;
}

}  // namespace vpu
