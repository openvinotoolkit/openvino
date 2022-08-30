// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pwl_segments_creator.hpp"

enum DnnActivationType : uint8_t;

namespace ov {
namespace intel_gna {
namespace backend {

class PWLSegmentsCreatorFactory {
public:
    static std::shared_ptr<PWLSegmentsCreator> CreateCreator(DnnActivationType activation_type);
};

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov