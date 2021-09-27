// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>
#include <ir_frontend/utility.hpp>
#include <istream>
#include <memory>
#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace frontend {
class IR_API InputModelIR : public InputModel {
    friend class FrontEndIR;
    class InputModelIRImpl;
    std::shared_ptr<InputModelIRImpl> _impl;

public:
    InputModelIR(std::istream& stream, const ov::Weights& weights, const ov::Extensions& extensions);

    std::shared_ptr<Function> convert();
};

}  // namespace frontend
}  // namespace ngraph
