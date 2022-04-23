// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/progress_reporter.hpp"
#include "openvino/frontend/extension/telemetry.hpp"

namespace ov {
namespace frontend {
struct ExtensionHolder {
    ExtensionHolder() : progress_reporter{std::make_shared<ProgressReporterExtension>()} {}
    ~ExtensionHolder() = default;
    ExtensionHolder(const ExtensionHolder&) = default;
    ExtensionHolder(ExtensionHolder&&) = default;
    ExtensionHolder& operator=(const ExtensionHolder&) = default;
    ExtensionHolder& operator=(ExtensionHolder&&) = default;
    
    std::shared_ptr<ProgressReporterExtension> progress_reporter;
    std::shared_ptr<TelemetryExtension> telemetry;
    std::vector<std::shared_ptr<ConversionExtensionBase>> conversions;
};
}  // namespace frontend
}  // namespace ov
