// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>

#include "frontend_manager/frontend_manager.hpp"
#include "ir_frontend/utility.hpp"

namespace ov {
namespace frontend {

class IR_API InputModelIR : public InputModel {
    friend class FrontEndIR;
    class InputModelIRImpl;
    std::shared_ptr<InputModelIRImpl> _impl;

public:
    InputModelIR(std::istream& stream,
                 const std::shared_ptr<ngraph::runtime::AlignedBuffer>& weights,
                 const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions);

    std::shared_ptr<Function> convert();
};

}  // namespace frontend
}  // namespace ov
