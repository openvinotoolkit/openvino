// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

//
// BatchNormalizationWeightsContent
//

class BatchNormalizationWeightsContent final : public CalculatedDataContent {
public:
    BatchNormalizationWeightsContent(const DataContent::Ptr& origContent, float epsilon);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void* tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    float _epsilon;
};

//
// BatchNormalizationBiasesContent
//

class BatchNormalizationBiasesContent final : public CalculatedDataContent {
public:
    BatchNormalizationBiasesContent(const DataContent::Ptr& origContent, const DataContent::Ptr& weightsContent);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void* tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    DataContent::CPtr _weightsContent;
};

} // namespace vpu
