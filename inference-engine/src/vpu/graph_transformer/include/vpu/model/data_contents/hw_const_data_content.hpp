// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

#include <vpu/middleend/hw/tiling.hpp>

namespace vpu {

class HwConstData final : public CalculatedDataContent {
public:
    HwConstData(
            const DataContent::Ptr& origContent,
            const DataDesc& origDesc,
            const DataDesc& resDesc,
            const std::map<Dim, Slice> dimSlices);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *outBuf) const override;

private:
    DataContent::CPtr _origContent;
    DataDesc _origDesc;
    DataDesc _resDesc;
    std::map<Dim, Slice> _dimSlices;
};

} // namespace vpu
