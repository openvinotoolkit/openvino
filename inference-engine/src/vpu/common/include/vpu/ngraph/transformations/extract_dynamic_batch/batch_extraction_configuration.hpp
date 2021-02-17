// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

namespace vpu {

enum class SliceMode {
    Slice,
    Unchanged
};

class SliceConfiguration {
public:
    SliceConfiguration() = default;
    SliceConfiguration(std::vector<SliceMode> inputs, std::vector<SliceMode> outputs);

    bool isSliceSupported() const;
    const std::vector<SliceMode>& inputs() const;
    const std::vector<SliceMode>& outputs() const;

private:
    bool m_isSliceSupported = false;
    std::vector<SliceMode> m_inputs;
    std::vector<SliceMode> m_outputs;
};

}  // namespace vpu
