// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/progress_reporter.hpp"
#include "openvino/frontend/extension/telemetry.hpp"

namespace ov {
namespace frontend {
struct ExtensionHolder {
    ExtensionHolder() : progress_reporter{std::make_shared<ProgressReporterExtension>()} {}
    std::shared_ptr<ProgressReporterExtension> progress_reporter;
    std::shared_ptr<TelemetryExtension> telemetry;
};
}  // namespace frontend
}  // namespace ov
