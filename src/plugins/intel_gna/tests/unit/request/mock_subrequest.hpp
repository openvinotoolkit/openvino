// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "request/subrequest.hpp"

namespace GNAPluginNS {
namespace request {

class MockSubrequest : public Subrequest {
public:
    MOCK_METHOD(RequestStatus, wait, (int64_t), (override));
    MOCK_METHOD(void, enqueue, (), (override));
    MOCK_METHOD(bool, isPending, (), (const, override));
    MOCK_METHOD(bool, isAborted, (), (const, override));
    MOCK_METHOD(bool, isCompleted, (), (const, override));
};

}  // namespace request
}  // namespace GNAPluginNS
