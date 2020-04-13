// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>
#include <ie_error.hpp>

IE_SUPPRESS_DEPRECATED_START

class Listener : public InferenceEngine::IErrorListener {
public:
    MOCK_QUALIFIED_METHOD1(onError, noexcept, void(const char * err));
};

IE_SUPPRESS_DEPRECATED_END
