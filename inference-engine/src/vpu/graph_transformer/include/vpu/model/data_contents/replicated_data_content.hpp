// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

class ReplicatedContent final : public CalculatedDataContent {
public:
    ReplicatedContent(float val, int count, const DataDesc& desc);

    ReplicatedContent(DataContent::Ptr origContent, int count, const DataDesc& desc);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataContent::CPtr _origContent = nullptr;
    DataDesc _desc;
    float _factor = 1.0f;
    int _count = 0;
};

DataContent::Ptr replicateContent(float val, int count, const DataDesc& desc);
DataContent::Ptr replicateContent(const DataContent::Ptr& origContent, int count, const DataDesc& desc);

} // namespace vpu
