// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "base_reference_test.hpp"

using namespace ngraph;

namespace {

struct AcoshParams {
    template <class IT, class OT>
    AcoshParams(const PartialShape& shape, const element::Type& type, const std::vector<IT>& iValues, const std::vector<OT>& oValues, size_t size = 0)
        : pshape(shape), type(type), input_data(CreateBlob(type, iValues, size)), expected_output(CreateBlob(type, oValues, size)) {}
    PartialShape pshape;
    element::Type type;
    InferenceEngine::Blob::Ptr input_data;
    InferenceEngine::Blob::Ptr expected_output;
};

class ReferenceAcoshLayerTest : public testing::TestWithParam<AcoshParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.type);
        inputData = {params.input_data};
        refOutData = {params.expected_output};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AcoshParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "type=" << param.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& pshape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, pshape);
        const auto acosh = std::make_shared<op::Acosh>(in);
        return std::make_shared<Function>(NodeVector {acosh}, ParameterVector {in});
    }
};

TEST_P(ReferenceAcoshLayerTest, AcoshWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_Acosh_With_Hardcoded_Refs, ReferenceAcoshLayerTest,
                         ::testing::Values(AcoshParams {PartialShape {8}, element::f32, std::vector<float> {1.f, 2.f, 3.f, 4.f, 5.f, 10.f, 100.f, 1000.f},
                                                        std::vector<float> {0.f, 1.316958, 1.762747, 2.063437, 2.292432, 2.993223, 5.298292, 7.600902}},
                                           AcoshParams {PartialShape {8}, element::i32, std::vector<int32_t> {1, 2, 3, 4, 5, 10, 100, 1000},
                                                        std::vector<int32_t> {0, 1, 2, 2, 2, 3, 5, 8}},
                                           AcoshParams {PartialShape {8}, element::u32, std::vector<uint32_t> {1, 2, 3, 4, 5, 10, 100, 1000},
                                                        std::vector<uint32_t> {0, 1, 2, 2, 2, 3, 5, 8}}),
                         ReferenceAcoshLayerTest::getTestCaseName);
