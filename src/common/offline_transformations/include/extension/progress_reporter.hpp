// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/extension.hpp"

namespace ov {
namespace frontend {
struct ProgressCounter {
    explicit ProgressCounter(unsigned int steps);

    unsigned int completed_steps() const {
        return m_completed_steps;
    }

    float current_progress() const;

    float advance(unsigned int steps = 1u);

    ProgressCounter& operator++() {
        advance();
        return *this;
    }

    ProgressCounter& operator++(int) {
        return this->operator++();
    }

private:
    unsigned int m_total_steps = 0u;
    unsigned int m_completed_steps = 0u;
};

class ProgressReporterExtension : public ov::Extension {
public:
    ProgressReporterExtension();
};
}  // namespace frontend
}  // namespace ov
