// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/data_content.hpp>

namespace vpu {

class MTCNNBlobContent final : public DataContent {
public:
    explicit MTCNNBlobContent(std::vector<char> blob);

    size_t byteSize() const override;

protected:
    const void* getRaw() const override;

private:
    std::vector<char> _blob;
};

} // namespace vpu
