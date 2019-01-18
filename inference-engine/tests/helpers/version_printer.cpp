// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>

#include <ie_version.hpp>

class PrintVersion {
public:
    PrintVersion() {
        printf("BuildVersion: %s\n",
               InferenceEngine::GetInferenceEngineVersion()->buildNumber);
    }
};

static PrintVersion cons;
