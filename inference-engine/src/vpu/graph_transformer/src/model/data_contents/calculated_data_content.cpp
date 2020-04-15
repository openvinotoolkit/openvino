// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/calculated_data_content.hpp>

namespace vpu {

const void* CalculatedDataContent::getRaw() const {
    if (_temp.empty()) {
        _temp.resize(byteSize());
        fillTempBuf(_temp.data());
    }
    return _temp.data();
}

} // namespace vpu
