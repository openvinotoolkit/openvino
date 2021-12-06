// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/extensions/json_transformation_extension.hpp>
#include <nlohmann/json.hpp>

class TestExtension1 : public ov::frontend::JsonTransformationExtension {
public:
    TestExtension1();

    virtual bool transform(std::shared_ptr<ov::Function>& function, const nlohmann::json& config) const override;
};
