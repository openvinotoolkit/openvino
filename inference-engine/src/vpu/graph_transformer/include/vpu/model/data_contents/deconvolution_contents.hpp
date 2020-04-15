// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

//
// DeconvolutionToConvolutionContent
//

class DeconvolutionToConvolutionContent final : public CalculatedDataContent {
public:
    DeconvolutionToConvolutionContent(const DataContent::Ptr& origContent, const DataDesc& desc);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    DataDesc _desc;
};

//
// DepthDeconvolutionCHWWeightsContent
//

class DepthDeconvolutionCHWWeightsContent final : public CalculatedDataContent {
public:
    DepthDeconvolutionCHWWeightsContent(
            const DataContent::Ptr& origContent,
            int KX, int KY, int channels);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    int _KX;
    int _KY;
    int _channels;
};

//
// DepthDeconvolutionHWCWeightsContent
//

class DepthDeconvolutionHWCWeightsContent final : public CalculatedDataContent {
public:
    DepthDeconvolutionHWCWeightsContent(
            const DataContent::Ptr& origContent,
            int KX, int KY, int channels);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataContent::CPtr _origContent;
    int _KX;
    int _KY;
    int _channels;
};

//
// DeconvolutionWeightsContent
//

class DeconvolutionWeightsContent final : public CalculatedDataContent {
public:
    DeconvolutionWeightsContent(
            const DataContent::Ptr& origContent,
            DataDesc desc,
            int KX, int KY,
            int IC, int OC);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataDesc _desc;
    DataContent::CPtr _origContent;
    mutable std::vector<fp16_t> _intermBuf;
    int _KX;
    int _KY;
    int _IC;
    int _OC;
};

} // namespace vpu
