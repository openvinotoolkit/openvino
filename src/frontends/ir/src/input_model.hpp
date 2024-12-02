// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>

#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

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
               std::string weights_path = {});

    InputModel(const std::shared_ptr<ov::AlignedBuffer>& model_buf,
               const std::shared_ptr<ov::AlignedBuffer>& weights,
               const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
               std::string weights_path = {});

    std::shared_ptr<Model> convert();
};

}  // namespace ir
}  // namespace frontend
}  // namespace ov
