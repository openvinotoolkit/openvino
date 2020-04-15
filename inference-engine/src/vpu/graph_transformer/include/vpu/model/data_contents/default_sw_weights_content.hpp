// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

class DefaultSwWeightsContent final : public CalculatedDataContent {
public:
    DefaultSwWeightsContent(const DataContent::Ptr& origContent, const DataDesc& desc);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void* tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    DataDesc _desc;
};

} // namespace vpu
