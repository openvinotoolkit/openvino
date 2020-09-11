// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/data_content.hpp>

namespace vpu {

class KernelBinaryContent final : public DataContent {
public:
    explicit KernelBinaryContent(const std::string& blob);

    size_t byteSize() const override;

protected:
    const void* getRaw() const override;

private:
    std::string _blob;
};

} // namespace vpu
