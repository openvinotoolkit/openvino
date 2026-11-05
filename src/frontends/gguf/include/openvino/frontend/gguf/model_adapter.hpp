// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/extension.hpp"
#include "openvino/core/model.hpp"

namespace ov::frontend::gguf {

using ModelMap = std::map<std::string, std::shared_ptr<ov::Model>>;

struct ModelAdapterResult {
    std::shared_ptr<ov::Model> model;
    ModelMap models;
};

// The supplied graph is an independent clone; the result identifies the graph to commit.
using ModelAdapterFn = std::function<ModelAdapterResult(const std::shared_ptr<ov::Model>&)>;

struct ModelAdapter {
    // Destroy the callback before releasing the libraries that implement it.
    std::vector<ov::Extension::Ptr> lifetime;
    ModelAdapterFn transform;
};

using ModelAdapters = std::map<std::string, std::shared_ptr<ModelAdapter>>;

}  // namespace ov::frontend::gguf
