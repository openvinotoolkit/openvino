// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/stage.hpp>

namespace vpu {

class PostOpStage : public StageNode {
public:
    using StageNode::StageNode;

protected:
    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override;

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override;

    void finalizeDataLayoutImpl() override;

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override;

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override;

    void initialCheckImpl() const override;

    void serializeDataImpl(BlobSerializer& serializer) const override;
};

}  // namespace vpu
