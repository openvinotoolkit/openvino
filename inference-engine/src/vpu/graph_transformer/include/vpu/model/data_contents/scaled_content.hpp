// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

class ScaledContent final : public CalculatedDataContent {
public:
    ScaledContent(const DataContent::Ptr& origContent, float scale);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    float _factor = 1.0f;
};

DataContent::Ptr scaleContent(const DataContent::Ptr& origContent, float scale);

} // namespace vpu
