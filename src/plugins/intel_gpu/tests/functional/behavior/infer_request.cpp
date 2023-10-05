// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/runtime/core.hpp"

#include <common_test_utils/test_common.hpp>
#include "ov_models/subgraph_builders.hpp"
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
    std::tie(netPrecision, inLayout, outLayout, shape, targetDevice) = GetParam();
    inPrc = netPrecision;
    outPrc = netPrecision;

    float clamp_min = netPrecision.isSigned() ? -5.f : 0.0f;
    float clamp_max = 5.0f;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape))};
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
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         InferRequestIOPrecision::getTestCaseName);

TEST(TensorTest, smoke_canSetShapeForPreallocatedTensor) {
    auto ie = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ngraph::builder::subgraph::makeSplitMultiConvConcat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    auto exec_net = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
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

TEST(TensorTest, smoke_canSetScalarTensor) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{})};
    params.front()->set_friendly_name("Scalar_1");
    params.front()->output(0).get_tensor().set_names({"scalar1"});

    std::vector<size_t> const_shape = {1};
    auto const1 = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, const_shape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});
    const1->fill_data(ov::element::i64, 0);

    auto unsqueeze1 = std::make_shared<ngraph::opset1::Unsqueeze>(params.front(), const1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(unsqueeze1)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);

    auto ie = ov::Core();
    auto compiled_model = ie.compile_model(fnPtr, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();
    double real_data = 1.0;
    ov::Tensor input_data(ngraph::element::f64, {}, &real_data);
    request.set_tensor("scalar1", input_data);
    ASSERT_NO_THROW(request.infer());
}

TEST(TensorTest, smoke_canSetTensorForDynamicInput) {
    auto ie = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ngraph::builder::subgraph::makeSplitMultiConvConcat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    std::map<size_t, ov::PartialShape> shapes = { {0, ov::PartialShape{-1, -1, -1, -1}} };
    function->reshape(shapes);
    auto exec_net = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    ov::Tensor t1(ov::element::i8, {1, 4, 20, 20});
    ov::Tensor t2(ov::element::i8, {1, 4, 30, 30});
    ov::Tensor t3(ov::element::i8, {1, 4, 40, 40});

    // Check set_shape call for pre-allocated input/output tensors
    ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    ASSERT_NO_THROW(inf_req.infer());

    ASSERT_NO_THROW(inf_req.set_input_tensor(t2));
    ASSERT_NO_THROW(inf_req.infer());

    ASSERT_NO_THROW(inf_req.set_input_tensor(t3));
    ASSERT_NO_THROW(inf_req.infer());

    ASSERT_NO_THROW(inf_req.set_input_tensor(t3));
    ASSERT_NO_THROW(inf_req.infer());

    ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    ASSERT_NO_THROW(inf_req.infer());

    ASSERT_NO_THROW(inf_req.set_input_tensor(t2));
    ASSERT_NO_THROW(inf_req.infer());
}
