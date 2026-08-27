// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

// Records native i64 PagedSelectiveSSM metadata edges before the GPU
// precision-conversion pipeline runs.
class RecordPagedSelectiveSSMMetadataInputs final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RecordPagedSelectiveSSMMetadataInputs");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Restores only the native i64 metadata edges recorded before GPU precision
// conversion. Explicit conversions and unrelated consumers remain unchanged.
class PreservePagedSelectiveSSMMetadataWidth final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PreservePagedSelectiveSSMMetadataWidth");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
