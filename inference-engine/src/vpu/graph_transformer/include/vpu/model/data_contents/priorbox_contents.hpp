// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

//
// PriorBoxContent
//

class PriorBoxContent final : public CalculatedDataContent {
public:
    PriorBoxContent(
            const DataDesc& inDesc0,
            const DataDesc& inDesc1,
            const DataDesc& outDesc,
            const ie::CNNLayerPtr &layer);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataDesc _inDesc0;
    DataDesc _inDesc1;
    DataDesc _outDesc;
    ie::CNNLayerPtr _layer;
};

//
// PriorBoxClusteredContent
//

class PriorBoxClusteredContent final : public CalculatedDataContent {
public:
    PriorBoxClusteredContent(
            const DataDesc& inDesc0,
            const DataDesc& inDesc1,
            const DataDesc& outDesc,
            const ie::CNNLayerPtr& layer);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataDesc _inDesc0;
    DataDesc _inDesc1;
    DataDesc _outDesc;
    ie::CNNLayerPtr _layer;
};

} // namespace vpu
