// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "request/worker.hpp"

namespace GNAPluginNS {
namespace request {

class MockWorker : public Worker {
public:
    MOCK_METHOD(Gna2Model*, model, (), (override));
    MOCK_METHOD(const Gna2Model*, model, (), (const, override));
    MOCK_METHOD(void, enqueueRequest, (), (override));
    MOCK_METHOD(RequestStatus, wait, (int64_t), (override));
    MOCK_METHOD(bool, isFree, (), (const, override));
    MOCK_METHOD(uint32_t, representingIndex, (), (const, override));
    MOCK_METHOD(void, setRepresentingIndex, (uint32_t), (override));
    MOCK_METHOD(InferenceEngine::BlobMap&, result, (), (override));
    MOCK_METHOD(void, setResult, (const InferenceEngine::BlobMap&), (override));
    MOCK_METHOD(void, setResult, (InferenceEngine::BlobMap &&), (override));
};

}  // namespace request
}  // namespace GNAPluginNS
