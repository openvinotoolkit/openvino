// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

class HwWeightsContent final : public CalculatedDataContent {
public:
    HwWeightsContent(
            const DataContent::Ptr& origContent,
            const DataDesc& origWeightsDesc,
            const DataDesc& resDesc,
            int numInputChannels,
            int channelStartIndex = 0);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    DataDesc _origDesc;
    DataDesc _resDesc;
    int _numInputChannels = 0;
    int _channelStartIndex = 0;
};

} // namespace vpu
