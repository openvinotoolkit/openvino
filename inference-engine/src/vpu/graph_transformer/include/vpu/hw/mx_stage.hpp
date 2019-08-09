// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <vpu/model/stage.hpp>

namespace vpu {

class MyriadXHwStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override;

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step) override;

    void propagateDataOrderImpl() const override;

    void getDataStridesRequirementsImpl() const override;

    void finalizeDataLayoutImpl() override;

    void getBatchSupportInfoImpl() const override;

    void finalCheckImpl() const override;

    void serializeParamsImpl(BlobSerializer& serializer) const override;

    void serializeDataImpl(BlobSerializer& serializer) const override;
};

}  // namespace vpu
