// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <extension/json_transformation.hpp>

class TestExtension1 : public ov::frontend::JsonTransformationExtension {
public:
    TestExtension1();

    bool transform(const std::shared_ptr<ov::Model>& function, const std::string& config) const override;
};

class TestExtension2 : public ov::frontend::JsonTransformationExtension {
public:
    TestExtension2();

    bool transform(const std::shared_ptr<ov::Model>& function, const std::string& config) const override;
};
