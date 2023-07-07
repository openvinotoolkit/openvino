// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "hw_accelerated_converter.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

class ConverterFactory {
public:
    std::shared_ptr<HwAcceleratedDataConverter> create_converter();
};
}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov