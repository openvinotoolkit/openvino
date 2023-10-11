// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {
namespace ir {

class InputModel : public ov::frontend::InputModel {
    class InputModelIRImpl;
    std::shared_ptr<InputModelIRImpl> _impl;

public:
    OPENVINO_SUPPRESS_DEPRECATED_START
    InputModel(std::istream& stream,
               const std::shared_ptr<ngraph::runtime::AlignedBuffer>& weights,
               const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions);
    OPENVINO_SUPPRESS_DEPRECATED_END

    std::shared_ptr<Model> convert();
};

}  // namespace ir
}  // namespace frontend
}  // namespace ov
