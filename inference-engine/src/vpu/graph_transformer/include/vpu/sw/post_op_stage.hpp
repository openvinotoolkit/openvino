// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/stage.hpp>

namespace vpu {

class PostOpStage : public StageNode {
protected:
    void propagateDataOrderImpl() const override;

    void getDataStridesRequirementsImpl() const override;

    void finalizeDataLayoutImpl() override;

    void getBatchSupportInfoImpl() const override;

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override;

    void finalCheckImpl() const override;

    void serializeDataImpl(BlobSerializer& serializer) const override;
};

}  // namespace vpu
