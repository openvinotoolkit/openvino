// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>
#include <vpu/frontend/frontend.hpp>
#include <vpu/model/data_contents/calculated_data_content.hpp>
#include <vpu/model/model.hpp>
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
            const NodePtr &node);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataDesc _inDesc0;
    DataDesc _inDesc1;
    DataDesc _outDesc;
    NodePtr _node;
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
            const NodePtr& node);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataDesc _inDesc0;
    DataDesc _inDesc1;
    DataDesc _outDesc;
    NodePtr _node;
};

} // namespace vpu
