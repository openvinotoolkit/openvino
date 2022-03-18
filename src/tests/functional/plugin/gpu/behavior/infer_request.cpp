// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/runtime/core.hpp"

#include <common_test_utils/test_common.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "transformations/utils/utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace ::testing;

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U16,
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U32,
        InferenceEngine::Precision::U64,
        InferenceEngine::Precision::I64,
        // Interpreter backend doesn't implement evaluate method for OP
        // InferenceEngine::Precision::FP64,
};

typedef std::tuple<
        InferenceEngine::Precision,    // Input/Output Precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::Layout,       // Output layout
        std::vector<size_t>,           // Input Shape
        std::string> newtworkParams;

class InferRequestIOPrecision : public testing::WithParamInterface<newtworkParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<newtworkParams> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

std::string InferRequestIOPrecision::getTestCaseName(const testing::TestParamInfo<newtworkParams> &obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Layout inLayout, outLayout;
    std::vector<size_t> shape;
    std::string targetDevice;
    std::tie(netPrecision, inLayout, outLayout, shape, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "netPRC=" << netPrecision.name() << separator;
    result << "inL=" << inLayout << separator;
    result << "outL=" << outLayout << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void InferRequestIOPrecision::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> shape;
    std::tie(netPrecision, inLayout, outLayout.front(), shape, targetDevice) = GetParam();
    inPrc.front() = netPrecision;
    outPrc.front() = netPrecision;

    float clamp_min = netPrecision.isSigned() ? -5.f : 0.0f;
    float clamp_max = 5.0f;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {shape});
    params[0]->set_friendly_name("Input");

    auto activation = ngraph::builder::makeActivation(params[0],
                                                      ngPrc,
                                                      ngraph::helpers::ActivationTypes::Clamp,
                                                      {},
                                                      {clamp_min, clamp_max});

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params);
}

InferenceEngine::Blob::Ptr InferRequestIOPrecision::GenerateInput(const InferenceEngine::InputInfo &info) const {
    bool inPrcSigned = function->get_parameters()[0]->get_element_type().is_signed();
    bool inPrcReal = function->get_parameters()[0]->get_element_type().is_real();

    int32_t data_start_from = inPrcSigned ? -10 : 0;
    uint32_t data_range = 20;
    int32_t resolution = inPrcReal ? 32768 : 1;

    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_range,
                                            data_start_from,
                                            resolution);
}

TEST_P(InferRequestIOPrecision, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_GPU_BehaviorTests, InferRequestIOPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputPrecisions),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t>{1, 50}),
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         InferRequestIOPrecision::getTestCaseName);

TEST(TensorTest, smoke_canSetShapeForPreallocatedTensor) {
    auto ie = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ngraph::builder::subgraph::makeSplitMultiConvConcat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    auto exec_net = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    // Check set_shape call for pre-allocated input/output tensors
    auto input_tensor = inf_req.get_input_tensor(0);
    ASSERT_NO_THROW(input_tensor.set_shape({1, 4, 20, 20}));
    ASSERT_NO_THROW(input_tensor.set_shape({1, 3, 20, 20}));
    ASSERT_NO_THROW(input_tensor.set_shape({2, 3, 20, 20}));
    auto output_tensor = inf_req.get_output_tensor(0);
    ASSERT_NO_THROW(output_tensor.set_shape({1, 10, 12, 12}));
    ASSERT_NO_THROW(output_tensor.set_shape({1, 10, 10, 10}));
    ASSERT_NO_THROW(output_tensor.set_shape({2, 10, 20, 20}));
}
