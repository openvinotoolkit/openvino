// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dsr_tests_common.hpp"

namespace LayerTestsUtils {
namespace vpu {

struct GatherTestCase {
    DataShapeWithUpperBound inputShapes;
    DataShapeWithUpperBound indexShape;
    int64_t axis;
};

using GatherParameters = std::tuple<
    DataType,
    DataType,
    GatherTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_GatherBase : public testing::WithParamInterface<GatherParameters>,
                       public DSR_TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherParameters> obj) {
        DataType dataType, idxType;
        GatherTestCase gatherTestCase;
        LayerTestsUtils::TargetDevice targetDevice;
        std::tie(dataType, idxType, gatherTestCase, targetDevice) = obj.param;

        std::ostringstream result;
        result << "DT=" << dataType << "_";
        result << "IT=" << idxType << "_";
        result << "DataRealShape=" << CommonTestUtils::vec2str(gatherTestCase.inputShapes.shape) << "_";
        result << "DataUBShape=" << CommonTestUtils::vec2str(gatherTestCase.inputShapes.upperBoundShape) << "_";
        result << "IdxRealShape=" << CommonTestUtils::vec2str(gatherTestCase.inputShapes.shape) << "_";
        result << "IdxUBShape=" << CommonTestUtils::vec2str(gatherTestCase.inputShapes.upperBoundShape) << "_";
        result << "Axis=" << gatherTestCase.axis << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

protected:
    std::set<std::string> m_indicesInputNames;

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        const auto& name = info.name();
        if (m_indicesInputNames.count(name)) {
            const auto& parameters = GetParam();
            const auto& gatherSetup = std::get<2>(parameters);
            const auto& inputRank = gatherSetup.inputShapes.shape.size();
            const auto axis = gatherSetup.axis < 0 ? gatherSetup.axis + inputRank : gatherSetup.axis;

            const auto endValue = gatherSetup.inputShapes.shape[axis] - 1;

            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), endValue, 0);
        }
        return DSR_TestsCommon::GenerateInput(info);
    }
};

}  // namespace vpu
}  // namespace LayerTestsUtils
