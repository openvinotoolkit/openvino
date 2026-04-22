// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <istream>
#include <memory>
#include <unordered_map>

#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov::util {
class WeightsProvider;
}

namespace ov {
namespace frontend {
namespace ir {

class InputModel : public ov::frontend::InputModel {
    class InputModelIRImpl;
    std::shared_ptr<InputModelIRImpl> _impl;

public:
    InputModel(std::istream& stream,
               const std::shared_ptr<ov::AlignedBuffer>& weights,
               const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
               std::filesystem::path weights_path = {},
               std::shared_ptr<ov::util::WeightsProvider> weights_provider = nullptr);

    InputModel(const std::shared_ptr<ov::AlignedBuffer>& model_buf,
               const std::shared_ptr<ov::AlignedBuffer>& weights,
               const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
               std::filesystem::path weights_path = {},
               std::shared_ptr<ov::util::WeightsProvider> weights_provider = nullptr);

    std::shared_ptr<Model> convert();
};

}  // namespace ir
}  // namespace frontend
}  // namespace ov
