// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/extension/progress_reporter.hpp"

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
void ProgressReporterExtension::report_progress(float progress,
                                                unsigned int total_steps,
                                                unsigned int completed_steps) const {
    FRONT_END_GENERAL_CHECK(completed_steps <= total_steps,
                            "When reporting the progress, the number of completed steps can be at most equal to the "
                            "number of total steps.");
    FRONT_END_GENERAL_CHECK(progress >= 0.0f && progress <= 1.0f,
                            "The reported progress needs to be a value between 0.0 and 1.0");
    m_callback(progress, total_steps, completed_steps);
}
}  // namespace frontend
}  // namespace ov
