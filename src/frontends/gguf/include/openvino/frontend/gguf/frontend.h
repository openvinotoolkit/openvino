// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace gguf {

class FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd();

    static std::shared_ptr<Model> convert(const InputModel::Ptr& model, bool naive = false);
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
