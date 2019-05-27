// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/stage.hpp>

namespace vpu {

class StubStage final : public StageNode {
private:
    StagePtr cloneImpl() const override;

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>& inputScales,
            ScalePropagationStep step) override;

    DataMap<DimsOrder> propagateDataOrderImpl() const override;

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override;

    void finalizeDataLayoutImpl() override;

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override;

    void finalCheckImpl() const override;

    void serializeParamsImpl(BlobSerializer& serializer) const override;

    void serializeDataImpl(BlobSerializer& serializer) const override;
};

}  // namespace vpu
