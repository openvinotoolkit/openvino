// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_extensions.hpp"

#include <openvino/core/core.hpp>

bool TestExtension1::transform(const std::shared_ptr<ov::Model>& function, const std::string& config) const {
    function->set_friendly_name("TestFunction");
    return true;
}

TestExtension1::TestExtension1() : ov::frontend::JsonTransformationExtension("buildin_extensions_2::TestExtension1") {}

bool TestExtension2::transform(const std::shared_ptr<ov::Model>& function, const std::string& config) const {
    function->set_friendly_name("TestFunction");
    return true;
}

TestExtension2::TestExtension2() : ov::frontend::JsonTransformationExtension("buildin_extensions_2::TestExtension2") {}
