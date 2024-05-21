// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

#include "common.hpp"
#include "npu_private_properties.hpp"

namespace intel_npu {

//
// register
//

void registerNPUWOptions(OptionsDesc& desc);

//
// FROM_NPUW
//

struct FROM_NPUW final : OptionBase<FROM_NPUW, std::string> {
    static std::string_view key() {
        return ov::intel_npu::from_npuw.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

}  // namespace intel_npu
