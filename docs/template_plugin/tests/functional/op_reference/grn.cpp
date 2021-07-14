// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace ngraph;
using namespace InferenceEngine;

struct GrnParams {
    template <class IT, class OT>
    GrnParams(const float bias, const ngraph::PartialShape& shape, const ngraph::element::Type& iType, const ngraph::element::Type& oType,
              const std::vector<IT>& iValues, const std::vector<OT>& oValues, size_t iSize = 0, size_t oSize = 0)
        : bias(bias), pshape(shape), inType(iType), outType(oType), inputData(CreateBlob(iType, iValues, iSize)), refData(CreateBlob(oType, oValues, oSize)) {}
    float bias;
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceGrnLayerTest : public testing::TestWithParam<GrnParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.bias, params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GrnParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bias=" << param.bias << "_";
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(float bias, const PartialShape& input_shape, const element::Type& input_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto grn = std::make_shared<op::v0::GRN>(in, bias);
        return std::make_shared<Function>(NodeVector {grn}, ParameterVector {in});
    }
};

TEST_P(ReferenceGrnLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_GRN_With_Hardcoded_Refs, ReferenceGrnLayerTest,
                         ::testing::Values(GrnParams(1e-6f, ngraph::PartialShape {1, 2, 3, 4}, ngraph::element::f32, ngraph::element::f32,
                                                     std::vector<float> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                                                     std::vector<float> {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                                                                         0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.4314555f,  0.4472136f,
                                                                         0.9970545f,  0.98994946f, 0.9805807f,  0.97014254f, 0.9593655f,  0.9486833f,
                                                                         0.9383431f,  0.9284767f,  0.91914505f, 0.9103665f,  0.9021342f,  0.8944272f})),
                         ReferenceGrnLayerTest::getTestCaseName);
