// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include "just_sync_infer_request.hpp"

namespace ov {
namespace npuw {

class UnfoldInferRequest final : public JustInferRequest {
public:
    explicit UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

private:
    void infer() override;
};

}  // namespace npuw
}  // namespace ov
