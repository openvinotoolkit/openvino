// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/data_content.hpp>

#include <vpu/utils/small_vector.hpp>
#include <vpu/model/data_desc.hpp>

namespace vpu {

//
// Data content that is calculated on the fly, using lazy calculation:
//
//   * It performs calculation on the first call and stores it in internal buffer.
//   * Next access will return the pointer to calculated buffer.
//

class CalculatedDataContent : public DataContent {
public:
    CalculatedDataContent() = default;

private:
    const void* getRaw() const override;

    virtual void fillTempBuf(void *tempBuf) const = 0;

private:
    mutable std::vector<uint8_t> _temp;
};

} // namespace vpu
