// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include <gmock/gmock.h>

class Listener : public InferenceEngine::IErrorListener {
public:
    MOCK_QUALIFIED_METHOD1(onError, noexcept, void (const char * err));
};
