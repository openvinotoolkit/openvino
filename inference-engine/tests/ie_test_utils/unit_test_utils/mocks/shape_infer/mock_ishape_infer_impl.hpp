// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <gmock/gmock.h>

#include <ie_api.h>

IE_SUPPRESS_DEPRECATED_START

#include <ie_iextension.h>

using namespace InferenceEngine;

class MockIShapeInferImpl : public IShapeInferImpl {
public:
    using Ptr = std::shared_ptr<MockIShapeInferImpl>;

    MOCK_QUALIFIED_METHOD5(inferShapes, noexcept, StatusCode(
            const std::vector<Blob::CPtr> &,
            const std::map<std::string, std::string>&,
            const std::map<std::string, Blob::Ptr>&,
            std::vector<SizeVector> &,
            ResponseDesc *));
};

IE_SUPPRESS_DEPRECATED_END
