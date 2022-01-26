// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

class MergeFullyConnectedContentsByChannels final : public CalculatedDataContent {
public:
    MergeFullyConnectedContentsByChannels(const std::vector<DataContent::CPtr> contents,
                                          const std::vector<DataDesc> inDescs,
                                          const DataDesc& resDesc);

    size_t byteSize() const override;

protected:
    void fillTempBuf(void *temp) const override;

private:
    std::vector<DataContent::CPtr> _contents;
    std::vector<DataDesc> _inDescs;
    DataDesc _resDesc;
};

} // namespace vpu
