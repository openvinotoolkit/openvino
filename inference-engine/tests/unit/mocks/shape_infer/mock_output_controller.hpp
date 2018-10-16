// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <inference_engine/shape_infer/ie_reshape_io_controllers.hpp>

using namespace InferenceEngine;

class MockOutputController : public OutputController {
public:
    MockOutputController(const std::vector<DataPtr>& dataVec) : OutputController(dataVec, {}, {}, std::make_shared<EmptyChecker>()) {}

    MOCK_METHOD2(setShapeByName, void(const SizeVector&, const std::string&));

    MOCK_METHOD2(setShapeByIndex, void(const SizeVector&, size_t index));

    MOCK_METHOD1(getIRShapeByName, SizeVector(const std::string&));

    MOCK_METHOD1(getShapes, std::vector<SizeVector>(bool));

    MOCK_METHOD0(getIRShapes, std::vector<SizeVector>());

    MOCK_METHOD0(applyChanges, void());

    MOCK_METHOD0(reset, void());

    MOCK_METHOD1(propagateShapes, void(const std::set<ReshapeLauncher::Ptr>&));

    MOCK_METHOD1(setShapes, void(const std::vector<SizeVector>&));

    std::vector<SizeVector> realGetShapes() {
        return OutputController::getShapes(false);
    }
};

