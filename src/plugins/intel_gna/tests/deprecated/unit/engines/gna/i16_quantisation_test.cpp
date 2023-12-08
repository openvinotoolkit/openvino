// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <legacy/layer_transform.hpp>
#include <vector>

#include "backend/gna_limitations.hpp"
#include "backend/gna_types.hpp"
#include "frontend/layer_quantizer.hpp"
#include "frontend/model_quantizer.hpp"
#include "gna_matcher.hpp"
#include "ov_models/builders.hpp"

using namespace InferenceEngine;
using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::frontend;
using namespace GNATestIRs;

class I16QuantisationTest : public GNATest<> {
protected:
    InferenceEngine::CNNLayerPtr quantize(InferenceEngine::CNNLayerPtr lp) {
        auto newLayer = InferenceEngine::injectData<QuantizedLayerParams>(lp);
        Config gna_config;
        gna_config.gnaPrecision = InferenceEngine::Precision::I16;
        gna_config.gnaFlags.input_low_precision = false;
        LayerQuantizer lq(gna_config);
        lq.quantize(*newLayer);
        return newLayer;
    };

    InferenceEngine::CNNNetwork quantize_single_input_model(const InferenceEngine::CNNNetwork& model,
                                                            float scale_factor) const {
        auto scale_factors = std::vector<float>({scale_factor});

        GnaInputs inputs;
        InferenceEngine::InputsDataMap inputs_map = model.getInputsInfo();

        auto input_layer = getCreatorLayer(inputs_map.begin()->second->getInputData()).lock();
        inputs[input_layer->name].scale_factor = scale_factor;

        Config gna_config;
        gna_config.gnaPrecision = InferenceEngine::Precision::I16;
        gna_config.gnaFlags.input_low_precision = false;

        auto transformer = ov::intel_gna::TransformationsPipeline(gna_config);

        return ModelQuantizer(transformer).quantize(model, inputs);
    }

    void SetUp() override {
        Limitations::init(target::DeviceVersion::Default);
    }

    void TearDown() override {
        Limitations::deinit();
    }
};

template <class T>
T setWeights(T blob) {
    blob->allocate();
    // actual quantisation algorithm is involved - we need to provide weights that will be quantized with scale factor
    // of 1
    for (auto&& w : *blob) {
        w = MAX_VAL_2B_WEIGHT;
    }
    return blob;
}

template <>
TBlob<uint8_t>::Ptr setWeights(TBlob<uint8_t>::Ptr blob) {
    blob->allocate();
    auto buf = blob->buffer();
    auto ptr = buf.as<float*>();

    for (int i = 0; i != blob->byteSize() / 4; i++) {
        ptr[i] = MAX_VAL_2B_WEIGHT;
    }
    return blob;
}

// TODO: add test for FC weights after quantization
TEST_F(I16QuantisationTest, canQuantizeFCLayer) {
    auto fc = std::make_shared<FullyConnectedLayer>(LayerParams{"name", "type", Precision::FP32});
    fc->_out_num = 9;
    fc->_weights = setWeights(make_shared_blob<float>({Precision::FP32, {1, 1}, Layout::NC}));
    fillWeights(fc->_weights);
    fc->_biases = make_shared_blob<float>({Precision::FP32, {1, 1}, Layout::NC});
    fc->_biases->allocate();
    fillWeights(fc->_biases);

    std::shared_ptr<Data> outData =
        std::make_shared<Data>("data", TensorDesc(Precision::FP32, SizeVector({1, 1}), Layout::NC));
    fc->outData.push_back(outData);
    fc->insData.push_back(outData);

    ASSERT_NO_THROW(quantize(fc));
}

TEST_F(I16QuantisationTest, canQuantizeActivation) {
    auto sigmoid = std::make_shared<GenericLayer>(LayerParams{"name", "type", Precision::FP32});
    sigmoid->params["value"] = 2;
    sigmoid->type = "Activation";

    ASSERT_NO_THROW(quantize(sigmoid));
}

TEST_F(I16QuantisationTest, outputAffinePrecisionIs32Bits) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {440}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(Fc2DOutputModel(), weights);

    auto newNet = quantize_single_input_model(network, 1000);
    InputsDataMap inputs = newNet.getInputsInfo();
    auto affineDataPtr = getInputTo(inputs.begin()->second->getInputData()).begin()->second->outData.front();

    ASSERT_EQ(affineDataPtr->getTensorDesc().getPrecision(), Precision::I32);
}

TEST_F(I16QuantisationTest, canQuantizeLstmLikeTopology) {
    auto weights = setWeights(make_shared_blob<uint8_t>({Precision::U8, {440}, C}));
    // std::fill_n(weights->buffer().as<float*>(), weights->byteSize()/sizeof(float), 0);

    Core ie;
    auto network = ie.ReadNetwork(affineToMemoryModel(), weights);

    ASSERT_NO_THROW(quantize_single_input_model(network, 1000));
}

TEST_F(I16QuantisationTest, DISABLED_outputScaleFactorForAffineIsCorrect) {
    const float inputScaleFactorTest = 1000;
    const float weightValueTest = 100;

    auto weights = make_shared_blob<uint8_t>({Precision::U8, {440}, C});
    weights->allocate();
    fillWeights(weights, {weightValueTest});

    Core ie;
    auto network = ie.ReadNetwork(Fc2DOutputModel(), weights);

    auto newNet = quantize_single_input_model(network, inputScaleFactorTest);
    InputsDataMap inputs = newNet.getInputsInfo();
    auto affineLayerPtr = getInputTo(inputs.begin()->second->getInputData()).begin()->second;

    auto quantParams = getInjectedData<QuantizedLayerParams>(affineLayerPtr);

    ASSERT_FLOAT_EQ(quantParams->_dst_quant.GetScale(), MAX_VAL_2B_WEIGHT / weightValueTest * inputScaleFactorTest);
    ASSERT_FLOAT_EQ(quantParams->_weights_quant.GetScale(), MAX_VAL_2B_WEIGHT / weightValueTest);
}

TEST_F(I16QuantisationTest, OnlyAffine_NoActivationInsertion) {
    assert_that()
        .onInferModel(Fc2DOutputModel())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_without()
        .pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, OnlyAffine_NoActivationInsertion_ProfilingEnabled) {
    assert_that()
        .onInferModel(Fc2DOutputModel())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_without()
        .pwl_inserted_into_nnet()
        .profiling_counters();
}

TEST_F(I16QuantisationTest, OnlyAffineWithNanScaleFactorFails) {
    gna().onInferModel(Fc2DOutputModel()).withNanScaleFactor().propagate_forward().throws();
}

TEST_F(I16QuantisationTest, OnlyAffineWithInfScaleFactorFails) {
    gna().onInferModel(Fc2DOutputModel()).withInfScaleFactor().propagate_forward().throws();
}

TEST_F(I16QuantisationTest, AffineToMemoryWillResultInActivationInsertion) {
    assert_that()
        .onInferModel(affineToMemoryModel())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseToMemoryWithNoOutputActivationInsertion) {
    assert_that()
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .onInferModel(eltwiseToMemoryModelNoOutput(),
                      [](CNNNetwork& net) {
                          net.addOutput("Eltwise_8");
                      })
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseToMemory_ActivationInsertion) {
    assert_that()
        .onInferModel(eltwiseToMemoryModel())
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, SplitFollowedByActivation_DummyDiagonalAffineInsertion) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 20});
    const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
    auto split = std::make_shared<ngraph::opset8::Split>(input_params, axis_node, 2);
    auto tanh = std::make_shared<ngraph::opset8::Tanh>(split->outputs()[0]);
    auto add = std::make_shared<ngraph::opset8::Add>(split->outputs()[1], tanh);
    auto result = std::make_shared<ngraph::opset8::Result>(add);
    auto function =
        std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    assert_that()
        .onInferNgraphModel(function)
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, SliceFollowedBy2FCsAnd2Eltwises_AlignedFilterInsertion) {
    assert_that()
        .onInferModel(twoFCWithPaddingAfterSliceModel())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .diagonal_inserted_into_nnet();
}

// ToDo requires implementation of aligning filter for concat inputs and improvement of
// qunatization/scaling algorithm for concat
TEST_F(I16QuantisationTest, DISABLED_DoubleConcatPropageteForwardWithSuccess_AlignedFilterInsertion) {
    assert_that()
        .onInferModel(doubleConcatModel())
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, EltwiseSumm_onlyOneIdentityInsertion) {
    assert_that()
        .onInferModel(eltwiseSummModel())
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet()
        .once();
}

TEST_F(I16QuantisationTest, EltwiseMull_willInsertTwoIdentities) {
    assert_that()
        .onInferModel(eltwiseMulModel())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet()
        .twice();
}

TEST_F(I16QuantisationTest, multiple_inputs_supported) {
    std::string configKey = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_");
    assert_that()
        .onInferModel(two_inputs_to_affine())
        .inNotCompactMode()
        .withGNAConfig(configKey + std::to_string(0), 1.0f)
        .withGNAConfig(configKey + std::to_string(1), 2.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet()
        .once();
}

TEST_F(I16QuantisationTest, DISABLED_multiple_inputs_into_concat_supported) {
    assert_that()
        .onInferModel(two_inputs_to_concat())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet()
        .once();
}

TEST_F(I16QuantisationTest, ScaleShift_Affine_WillResultInIdentityInsertion) {
    assert_that()
        .onInferModel(scaleShiftAffineModel())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet()
        .once();
}

TEST_F(I16QuantisationTest, ClampFollowedByTanh_ResultInDiagonalInsertion) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 10});
    auto clamp = std::make_shared<ngraph::opset8::Clamp>(input_params, -50, 50);
    auto tanh = std::make_shared<ngraph::opset8::Tanh>(clamp);
    auto result = std::make_shared<ngraph::opset8::Result>(tanh);
    auto function =
        std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    assert_that()
        .onInferNgraphModel(function)
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .diagonal_inserted_into_nnet()
        .twice();
}

TEST_F(I16QuantisationTest, EltwiseWithMemoryAndActivationInput_ResultInTwoDiagonalsInsertion) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 10});
    const auto constant = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{10, 10}, {1});
    auto matmul = std::make_shared<ngraph::opset8::MatMul>(input_params, constant);
    auto mem_i = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f32, ngraph::Shape{1, 10}, 0);
    auto mem_r = std::make_shared<ngraph::op::v3::ReadValue>(mem_i, "r_27-28");
    auto tanh = std::make_shared<ngraph::opset8::Tanh>(matmul);
    auto add = std::make_shared<ngraph::opset8::Add>(tanh, mem_r);
    tanh->add_control_dependency(mem_r);
    auto mem_w = std::make_shared<ngraph::op::v3::Assign>(tanh, "r_27-28");
    auto result = std::make_shared<ngraph::opset8::Result>(add);
    mem_w->add_control_dependency(mem_r);
    result->add_control_dependency(mem_w);
    auto function =
        std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    assert_that()
        .onInferNgraphModel(function)
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .diagonal_inserted_into_nnet()
        .twice();
}

TEST_F(I16QuantisationTest, AffineWith2AffineOutputs_ResultInOnlyOneIdentityInsertion) {
    // one Identity activation from first FC, and one Identity activation for eltwise
    assert_that()
        .onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwl_inserted_into_nnet()
        .twice();
}

TEST_F(I16QuantisationTest, ScaleShiftWithBroadcast_ResultInDiagonalInsertion) {
    auto& affineWeights = storage<std::vector<uint16_t>>();

    affineWeights = {
        2048,  4096,  6144,  8192,  10240, 12288, 14336, 16384, 2048,  4096,  6144,  8192,  10240, 12288,
        14336, 16384, 2048,  4096,  6144,  8192,  10240, 12288, 14336, 16384, 2048,  4096,  6144,  8192,
        10240, 12288, 14336, 16384, 2048,  4096,  6144,  8192,  10240, 12288, 14336, 16384,
    };

    assert_that()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .onInferModel(ScaleShift3DModel())
        .withWeigthsPattern({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f})
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .called_with()
        .affine_weights_eq(affineWeights);
}

TEST_F(I16QuantisationTest, MemoryAfterConcat_ResultInCopyInsertion) {
    assert_that()
        .onInferModel(MemoryAfterConcatModel())
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .copy_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, MemoryAndConcatAfterOneNode_ResultInCopyInsertion) {
    assert_that()
        .onInferModel(MemoryAndConcatAfterOneNode())
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .copy_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, DISABLED_permutationOfWeightsBetweenConvAndAffine) {
    auto& affineWeights = storage<std::vector<uint16_t>>();

    // least likely that width and height both are multiple of 7
    auto weigthsPattern = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};

    // here weights are transpozed
    save()
        .onInferModel(affineAfterConvNoPermute())
        .withWeigthsPattern(weigthsPattern)
        .inNotCompactMode()
        .from()
        .propagate_forward()
        .affine_weights_transpozed({128, 61})
        .to(affineWeights);

    // here weights shouldn't be transposed
    assert_that()
        .onInferModel(affineAfterConvWithPermute())
        .withWeigthsPattern(weigthsPattern)
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .affine_weights_eq(affineWeights);
}

TEST_F(I16QuantisationTest, DISABLED_noPermutationOfWeightsBetweenConvAndAffineIfPermuteLayerWithCorrectArgs) {
    auto& affineWeights = storage<std::vector<uint16_t>>();

    // least likely that width and height both are multiple of 7
    auto weigthsPattern = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};

    save()
        .onInferModel(affineAfterConvWithPermute())
        .withWeigthsPattern(weigthsPattern)
        .inNotCompactMode()
        .from()
        .propagate_forward()
        .affine_weights()
        .to(affineWeights);

    assert_that()
        .onInferModel(affineAfterConvNoPermute())
        .withWeigthsPattern(weigthsPattern)
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .affine_weights_transposed(affineWeights, {128, 61});
}

TEST_F(I16QuantisationTest, fp16tofp32_on_fullyConnected_model) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {220}, Layout::C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(FCOnlyModelFP16(), weights);

    quantize_single_input_model(network, 1000);
}

TEST_F(I16QuantisationTest,
       MultipleActivationsAfterAffineWithIdentityActivation_MultipleDiagonalLayersWithActivaitons) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 10});
    const auto constant = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{10, 10}, {1});
    auto matmul1 = std::make_shared<ngraph::opset8::MatMul>(input_params, constant);
    auto matmul2 = std::make_shared<ngraph::opset8::MatMul>(input_params, constant);
    auto add = std::make_shared<ngraph::opset8::Add>(matmul2, matmul1);
    auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(matmul2);
    auto relu = std::make_shared<ngraph::opset8::Relu>(matmul2);
    auto mul = std::make_shared<ngraph::opset8::Multiply>(sigmoid, relu);
    auto add2 = std::make_shared<ngraph::opset8::Add>(add, mul);
    auto result = std::make_shared<ngraph::opset8::Result>(add);
    auto function =
        std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    // identiy came from automatic insertion due to
    assert_that()
        .onInferNgraphModel(function)
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwls_inserted_into_nnet({kActSigmoid, kActRelu, kActIdentity, kActIdentity});
}

TEST_F(I16QuantisationTest, MultipleActivationsAfterAffine_ResultInMultipleDiagonalLayersWithActivaitons) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 10});
    const auto constant = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{10, 10}, {1});
    auto matmul = std::make_shared<ngraph::opset8::MatMul>(input_params, constant);
    auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(matmul);
    auto relu = std::make_shared<ngraph::opset8::Relu>(matmul);
    auto mul = std::make_shared<ngraph::opset8::Multiply>(sigmoid, relu);
    auto result = std::make_shared<ngraph::opset8::Result>(mul);
    auto function =
        std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    // extra identity inserted for affine
    assert_that()
        .onInferNgraphModel(function)
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        // 1 diag for second activation, 1 for eltwise
        .pwls_inserted_into_nnet({kActRelu, kActSigmoid})
        .diagonal_inserted_into_nnet()
        .times(3);
}

// TODO: build a regression test on top of it using real quantisation accuracy checking
TEST_F(I16QuantisationTest, ConcatWithConstInputPropagatedForward) {
    assert_that()
        .onInferModel(concatModelWithConstLayer())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwls_inserted_into_nnet({kActIdentity});
}

TEST_F(I16QuantisationTest, LSTMCell_quantize) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {33664}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(LSTMCellOnlyModel(), weights);

    ASSERT_NO_THROW(quantize_single_input_model(network, 1000));
}

TEST_F(I16QuantisationTest, LSTMCell_unaligned_quantize) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {3480}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(LSTMCellOnlyModelUnaligned(), weights);

    ASSERT_NO_THROW(quantize_single_input_model(network, 1000));
}

TEST_F(I16QuantisationTest, EltwisetWithConstInputPropagatedForward) {
    assert_that()
        .onInferModel(eltwiseSumModelWithConstLayer())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, PowerWithScaleFactorPropagateForward) {
    assert_that()
        .onInferModel(PowerWithScaleFactor1())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwls_inserted_into_nnet({kActIdentity})
        .And()
        .diagonal_inserted_into_nnet();
}

TEST_F(I16QuantisationTest, ConcatWithDifferentInputScaleFactorsPropagateForward) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 20});
    const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
    auto split = std::make_shared<ngraph::opset8::Split>(input_params, axis_node, 2);
    auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(split->outputs()[0]);
    auto tanh = std::make_shared<ngraph::opset8::Tanh>(split->outputs()[1]);
    auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{sigmoid, tanh}, 1);
    auto result = std::make_shared<ngraph::opset8::Result>(concat);
    auto function =
        std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    assert_that()
        .onInferNgraphModel(function)
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called_with()
        .pwls_inserted_into_nnet({kActIdentity});
}

TEST_F(I16QuantisationTest, TI_quantize) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {249748}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(TIModelWithLSTMCell2(), weights);

    ASSERT_NO_THROW(quantize_single_input_model(network, 1000));
}

TEST_F(I16QuantisationTest, TI_PropagateForward) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 10});
    auto mul = std::make_shared<ngraph::opset8::Multiply>(
        input_params,
        std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{1, 10}));
    auto add = std::make_shared<ngraph::opset8::Add>(
        mul,
        std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{1, 10}));
    auto reshape = std::make_shared<ngraph::opset8::Reshape>(
        add,
        std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<size_t>{1, 1, 10}),
        false);

    auto reshape_shape = reshape->output(0).get_shape();
    const size_t batch_size = 1;
    const size_t hiddenSize = 10;

    auto H_init = ngraph::builder::makeConstant<float>(ngraph::element::f32, {batch_size, hiddenSize}, {}, true);
    auto C_init = ngraph::builder::makeConstant<float>(ngraph::element::f32, {batch_size, hiddenSize}, {}, true);

    auto H_t = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, hiddenSize});
    auto C_t = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, hiddenSize});

    // Body
    auto X = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32,
                                                         ngraph::Shape{batch_size, 1, reshape_shape[2]});
    auto weightsNode =
        ngraph::builder::makeConstant<float>(ngraph::element::f32, {4 * hiddenSize, reshape_shape[2]}, {}, true);
    auto reccurrenceWeightsNode =
        ngraph::builder::makeConstant<float>(ngraph::element::f32, {4 * hiddenSize, hiddenSize}, {}, true);

    // lstm
    auto constantX =
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {batch_size, reshape_shape[2]});
    auto lstm1 =
        std::make_shared<ngraph::opset8::LSTMCell>(std::make_shared<ngraph::opset8::Reshape>(X, constantX, false),
                                                   H_t,
                                                   C_t,
                                                   weightsNode,
                                                   reccurrenceWeightsNode,
                                                   hiddenSize);

    auto H_o = lstm1->output(0);
    auto C_o = lstm1->output(1);

    auto body =
        std::make_shared<ngraph::Function>(ngraph::OutputVector{H_o, C_o}, ngraph::ParameterVector{X, H_t, C_t});

    auto tensor_iterator = std::make_shared<ngraph::opset8::TensorIterator>();
    tensor_iterator->set_body(body);

    tensor_iterator->set_sliced_input(X, reshape, 0, 1, 1, -1, 1);
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);

    auto out0 = tensor_iterator->get_iter_value(H_o, -1);

    const size_t output_size = 12;
    auto fc = ngraph::builder::makeFullyConnected(out0,
                                                  ngraph::element::f32,
                                                  output_size,
                                                  true,
                                                  {hiddenSize, output_size},
                                                  {1},
                                                  {1});
    auto result = std::make_shared<ngraph::opset8::Result>(fc);
    auto function =
        std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    assert_that()
        .onInferNgraphModel(function)
        .withWeigthsPattern({0.1f})
        .inNotCompactMode()
        .gna()
        .propagate_forward()
        .called_with()
        .pwls_inserted_into_nnet({kActIdentity});
}

TEST_F(I16QuantisationTest, SplitToConcatWith2Inputs1360NotAlignedNoFC) {
    assert_that()
        .onInferModel(SplitToConcatWith2Inputs1360NotAlignedNoFC())
        .inNotCompactMode()
        .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .gna()
        .propagate_forward()
        .called();
}
