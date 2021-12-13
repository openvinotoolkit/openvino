// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <json_extension/json_transformation_extension.hpp>
#include <nlohmann/json.hpp>

class TestExtension1 : public ov::frontend::JsonTransformationExtension {
public:
    TestExtension1();

    bool transform(std::shared_ptr<ov::Model>& function, const nlohmann::json& config) const override;
};
