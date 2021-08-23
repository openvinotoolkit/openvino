// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <iostream>

using namespace ov;

ov::Extension::~Extension() {
    std::cout << "AAAA " << std::endl;
}

void ov::setExtensionSharedObject(const Extension::Ptr& extension, const SOLoader& so) {
    extension->so = so;
}
