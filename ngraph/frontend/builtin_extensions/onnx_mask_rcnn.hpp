// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/extension.hpp>

class ONNXMaskRCNN : public ngraph::frontend::JsonTransformationExtension {
public:

    ONNXMaskRCNN () : ngraph::frontend::JsonTransformationExtension("ONNXMaskRCNNReplacement") {}

    virtual bool transform (std::shared_ptr<ov::Function>& function, const nlohmann::json& config) const override;
};
