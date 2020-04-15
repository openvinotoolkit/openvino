// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

//
// ConvIm2ColWeightsContent
//

class ConvIm2ColWeightsContent final : public CalculatedDataContent {
public:
    explicit ConvIm2ColWeightsContent(const DataContent::Ptr& origContent, DataDesc desc);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void* tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    DataDesc _desc;
};

//
// Conv3x3WeightsContent
//

class Conv3x3WeightsContent final : public CalculatedDataContent {
public:
    explicit Conv3x3WeightsContent(const DataContent::Ptr& origContent, DataDesc desc);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void* tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    DataDesc _desc;
};

//
// ConvCHWWeightsContent
//

class ConvCHWWeightsContent final : public CalculatedDataContent {
public:
    explicit ConvCHWWeightsContent(const DataContent::Ptr& origContent, DataDesc desc);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void* tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    DataDesc _desc;
};

} // namespace vpu
