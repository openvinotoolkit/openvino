// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>

#include "test_extensions.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({std::make_shared<TestExtension1>(),
                                                            std::make_shared<TestExtension2>()}));
