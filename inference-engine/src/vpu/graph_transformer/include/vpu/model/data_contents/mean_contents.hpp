// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

#include <ie_preprocess.hpp>

namespace vpu {

//
// MeanImageContent
//

class MeanImageContent final : public CalculatedDataContent {
public:
    MeanImageContent(const ie::PreProcessInfo& info, const DataDesc& desc);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    DataDesc _desc;
    ie::PreProcessInfo _info;
};

//
// MeanValueContent
//

class MeanValueContent final : public CalculatedDataContent {
public:
    explicit MeanValueContent(const ie::PreProcessInfo& info);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *tempBuf) const override;

private:
    ie::PreProcessInfo _info;
};

} // namespace vpu
