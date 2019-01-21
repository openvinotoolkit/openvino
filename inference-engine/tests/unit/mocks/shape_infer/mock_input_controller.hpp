// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <inference_engine/shape_infer/ie_reshape_io_controllers.hpp>

using namespace InferenceEngine;
using namespace ShapeInfer;

class MockInputController : public InputController {
public:
    MockInputController(const std::vector<DataPtr>& dataVec) : InputController(dataVec, {}, std::make_shared<EmptyChecker>()) {}

    MOCK_METHOD2(setShapeByName, void(
            const SizeVector&, const std::string&));

    MOCK_METHOD2(setShapeByIndex, void(
            const SizeVector&, size_t index));

    MOCK_METHOD1(getShapes, std::vector<SizeVector>(bool));

    MOCK_METHOD0(getIRShapes, std::vector<SizeVector>());

    MOCK_METHOD1(getIRShapeByName, SizeVector(
            const std::string&));

    MOCK_METHOD0(applyChanges, void());

    MOCK_METHOD0(reset, void());

    SizeVector realGetIRShapeByName(const std::string& name) {
        return InputController::getIRShapeByName(name);
    }
};

