// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"
#include <math.h>

using namespace InferenceEngine;

struct region_test_params {
    tensor_test_params in;
    int coords;
    int classes;
    int num;
    int maskSize;
    int doSoftMax;
    std::string customLayers;
    friend std::ostream& operator<<(std::ostream& os, region_test_params const& tst)
    {
        return os << "tensor (" << tst.in
                  << "),coords=" << tst.coords
                  << ", classes=" << tst.classes
                  << ", num=" << tst.num
                  << ", maskSize=" << tst.maskSize
                  << ", doSoftMax=" << tst.doSoftMax
                  << ", by using custom layer=" << (tst.customLayers.empty() ? "no" : "yes");
    };
};

class myriadLayerRegionYolo_nightly: public myriadLayersTests_nightly,
                             public testing::WithParamInterface<region_test_params> {
};

TEST_P(myriadLayerRegionYolo_nightly, BaseTestsRegion) {
    region_test_params p = ::testing::WithParamInterface<region_test_params>::GetParam();

    // TODO: M2 mode is not working for OpenCL compiler
    if(!p.customLayers.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }

    std::map<std::string, std::string> params;

    params["coords"] = std::to_string(p.coords);
    params["classes"] = std::to_string(p.classes);
    params["num"] = std::to_string(p.num);
    params["mask"] = "0,1,2";
    params["do_softmax"] = std::to_string(p.doSoftMax);

    InferenceEngine::SizeVector tensor;
    tensor.resize(4);
    tensor[3] = p.in.w;
    tensor[2] = p.in.h;
    tensor[1] = p.in.c;
    tensor[0] = 1;
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = p.customLayers;
    _testNet.addLayer(LayerInitParams("RegionYolo")
             .params(params)
             .in({tensor})
             .out({tensor}),
             ref_RegionYolo_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().layoutPreference(vpu::LayoutPreference::ChannelMinor)));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), 0.0025);
}

static std::vector<region_test_params> s_regionData = {
    region_test_params{{1, (4+20+1)*5, 13, 13}, 4, 20, 5, 3, 1, ""},
    region_test_params{{1, (4+80+1)*5, 13, 13}, 4, 80, 5, 3, 1, ""},
    region_test_params{{1, (4+20+1)*3, 13, 13}, 4, 20, 9, 3, 0, ""},
    region_test_params{{1, (4+80+1)*3, 13, 13}, 4, 80, 9, 3, 0, ""},

#ifdef VPU_HAS_CUSTOM_KERNELS
   region_test_params{{1, (4+20+1)*5, 13, 13}, 4, 20, 5, 3, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
   region_test_params{{1, (4+80+1)*5, 13, 13}, 4, 80, 5, 3, 1, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
   region_test_params{{1, (4+20+1)*3, 13, 13}, 4, 20, 9, 3, 0, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
   region_test_params{{1, (4+80+1)*3, 13, 13}, 4, 80, 9, 3, 0, getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"},
#endif
};

/* HW network needs to be created to test strides influence to RegionYolo input */
/* so convolution layer added as the first layer to this test                   */
class myriadLayersTestsRegion_CHW_HW_nightly: public ConvolutionTest<>{
};

/*80 input classes */
class myriadLayersTestsRegion_CHW_HW_80cl_nightly: public ConvolutionTest<>{
};

/* to passthrough "original" data */
template<size_t width>
void constWeightsRange(uint16_t* ptr, size_t weightsSize) {
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(weightsSize, width * width);
    std::memset(ptr, 0, sizeof(uint16_t) * (weightsSize));
    for (int i = 0; i < weightsSize/width; ++i) {
        ptr[i * width + i] = PrecisionUtils::f32tof16(1.0f);
    }
}

void constBiasesRange(uint16_t* ptr, size_t weightsSize) {
    std::memset(ptr, 0, sizeof(uint16_t) * (weightsSize));
}

void loadData(InferenceEngine::Blob::Ptr blob) {
    /* input blob has predefined size and CHW layout */
    ASSERT_NE(blob, nullptr);
    auto inDims = blob->getTensorDesc().getDims();
    InferenceEngine::Blob::Ptr inputBlobRef =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, inDims, InferenceEngine::NCHW});
    inputBlobRef->allocate();
    const float* ref_values = inputBlobRef->buffer();

    std::string inputTensorBinary = TestDataHelpers::get_data_path();
    inputTensorBinary += "/vpu/InputYoLoV2Tiny.bin";
    ASSERT_TRUE(fromBinaryFile(inputTensorBinary, inputBlobRef));
    uint16_t *inputBlobRawDataFp16 = static_cast<uint16_t *>(blob->buffer());
    ASSERT_NE(inputBlobRawDataFp16, nullptr);

    switch(blob->getTensorDesc().getLayout()) {
    case InferenceEngine::NCHW:
        for (int indx = 0; indx < blob->size(); indx++) {
            inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(ref_values[indx]);
        }
        break;
    case InferenceEngine::NHWC:
        for (int h = 0 ; h < inDims[2]; ++h) {
            for (int w = 0 ; w < inDims[3]; ++w) {
                for (int c = 0 ; c < inDims[1]; ++c) {
                    int src_i = w + inDims[3] * h + inDims[3] * inDims[2] * c;
                    int dst_i = c + inDims[1] * w + inDims[3] * inDims[1] * h;
                    inputBlobRawDataFp16[dst_i] = PrecisionUtils::f32tof16(ref_values[src_i]);
                }
            }
        }
        break;
    default:
        FAIL() << "unsupported layout: " << blob->getTensorDesc().getLayout();
    }
}

void loadData_80cl(InferenceEngine::Blob::Ptr blob) {
    /* input blob has predefined size and CHW layout */
    ASSERT_NE(blob, nullptr);
    auto inDims = blob->getTensorDesc().getDims();
    InferenceEngine::Blob::Ptr inputBlobRef =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, inDims, InferenceEngine::NCHW});
    inputBlobRef->allocate();
    const float* ref_values = inputBlobRef->buffer();

    std::string inputTensorBinary = TestDataHelpers::get_data_path();
    inputTensorBinary += "/vpu/InputYoLoV2_80cl.bin";
    ASSERT_TRUE(fromBinaryFile(inputTensorBinary, inputBlobRef));
    uint16_t *inputBlobRawDataFp16 = static_cast<uint16_t *>(blob->buffer());
    ASSERT_NE(inputBlobRawDataFp16, nullptr);

    switch(blob->getTensorDesc().getLayout()) {
    case InferenceEngine::NCHW:
        for (int indx = 0; indx < blob->size(); indx++) {
            inputBlobRawDataFp16[indx] = PrecisionUtils::f32tof16(ref_values[indx]);
        }
        break;
    case InferenceEngine::NHWC:
        for (int h = 0 ; h < inDims[2]; ++h) {
            for (int w = 0 ; w < inDims[3]; ++w) {
                for (int c = 0 ; c < inDims[1]; ++c) {
                    int src_i = w + inDims[3] * h + inDims[3] * inDims[2] * c;
                    int dst_i = c + inDims[1] * w + inDims[3] * inDims[1] * h;
                    inputBlobRawDataFp16[dst_i] = PrecisionUtils::f32tof16(ref_values[src_i]);
                }
            }
        }
        break;
    default:
        FAIL() << "unsupported layout: " << blob->getTensorDesc().getLayout();
     }
}

TEST_P(myriadLayersTestsRegion_CHW_HW_nightly, RegionYolo) {
    std::map<std::string, std::string> params;
    params["coords"] = "4";
    params["classes"] = "20";
    params["num"] = "5";
    params["mask"] = std::string("0,1,2");
    params["do_softmax"] = "1";
    _testNet.addLayer(LayerInitParams("RegionYolo")
             .params(params)
             .in({_output_tensor})
             .out({{1, _output_tensor[0] * _output_tensor[1] * _output_tensor[2] * _output_tensor[3]}}),
             ref_RegionYolo_wrap);
    _testNet.setWeightsCallbackForLayer(0, constWeightsRange<125>);
    _testNet.setBiasesCallbackForLayer(0, constBiasesRange);
    _genDataCallback = loadData;
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt(true)));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), 0.0035);
}

TEST_P(myriadLayersTestsRegion_CHW_HW_80cl_nightly, RegionYolol) {
    std::map<std::string, std::string> params;
    params["coords"] = "4";
    params["classes"] = "80";
    params["num"] = "5";
    params["mask"] = std::string("0,1,2");
    params["do_softmax"] = "1";
    _testNet.addLayer(LayerInitParams("RegionYolo")
             .params(params)
             .in({_output_tensor})
             .out({{1, _output_tensor[0] * _output_tensor[1] * _output_tensor[2] * _output_tensor[3]}}),
             ref_RegionYolo_wrap);
    _testNet.setWeightsCallbackForLayer(0, constWeightsRange<425>);
    _testNet.setBiasesCallbackForLayer(0, constBiasesRange);
    _genDataCallback = loadData_80cl;
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt(true)));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), 0.0060);
}

class myriadLayerRegionYolo_CHW_nightly: public myriadLayersTests_nightly,
                             public testing::WithParamInterface<int> {
};

TEST_P(myriadLayerRegionYolo_CHW_nightly, TestsRegion) {
    auto classes = GetParam();
    InferenceEngine::SizeVector input_dims = {1, 125, 13, 13};
    if (classes == 80) {
        input_dims[1] = 425;
    }
    IN_OUT_desc input_tensor;
    input_tensor.push_back(input_dims);

    std::map<std::string, std::string> params;
    params["coords"] = "4";
    params["classes"] = std::to_string(classes);
    params["num"] = "5";
    params["mask"] = std::string("0,1,2");
    params["do_softmax"] = "1";
    _testNet.addLayer(LayerInitParams("RegionYolo")
             .params(params)
             .in(input_tensor)
             .out({{1, input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]}}),
             ref_RegionYolo_wrap);
    _genDataCallback = loadData;
    if (classes == 80) {
        _genDataCallback = loadData_80cl;
    }
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    /* bound is too high , set for M2 tests */
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), 0.006);
}

TEST_P(myriadLayerRegionYolo_CHW_nightly, Test_CHW_HWC_Compare) {
    auto classes = GetParam();
    IN_OUT_desc input_tensor;
    InferenceEngine::SizeVector input_dims = {1, 125, 13, 13};
    if (classes == 80) {
        input_dims[1] = 425;
    }

    input_tensor.push_back(input_dims);

    std::map<std::string, std::string> params;
    params["coords"] = "4";
    params["classes"] = std::to_string(classes);
    params["num"] = "5";
    params["mask"] = std::string("0,1,2");
    params["do_softmax"] = "1";
    _testNet.addLayer(LayerInitParams("RegionYolo")
             .params(params)
             .in(input_tensor)
             .out({{1, input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]}}),
             ref_RegionYolo_wrap);
    if (classes == 80) {
        _genDataCallback = loadData_80cl;
    }
    _config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)] = CONFIG_VALUE(NO);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt(false).runRefGraph(false)));
    /* create  NHWC version                                */
    /* we cannot use the same generateNetAndInfer call due */
    /* to IE bug.                                          */
    InferenceEngine::InputsDataMap           inputsInfo;
    InferenceEngine::BlobMap                 outputMap;
    InferenceEngine::OutputsDataMap          outputsInfo;
    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    InferenceEngine::IInferRequest::Ptr      inferRequest;

    _inputsInfo.begin()->second->setLayout(NHWC);
    _outputsInfo.begin()->second->setLayout(NC);

    InferenceEngine::StatusCode st = InferenceEngine::StatusCode::GENERAL_ERROR;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetwork, _cnnNetwork, _config, &_resp));
    ASSERT_NE(exeNetwork, nullptr) << _resp.msg;
    ASSERT_NO_THROW(exeNetwork->CreateInferRequest(inferRequest, &_resp)) << _resp.msg;
    ASSERT_NE(inferRequest, nullptr) << _resp.msg;
    ASSERT_NO_THROW(inputsInfo = _cnnNetwork.getInputsInfo());
    auto inIt = _inputsInfo.begin();
    for (auto in = _inputsInfo.begin(); in != _inputsInfo.end(); in++) {
        Blob::Ptr inpt;
        ASSERT_NO_THROW(_inferRequest->GetBlob(inIt->first.c_str(), inpt, &_resp));
        ASSERT_NO_THROW(inferRequest->SetBlob(inIt->first.c_str(), inpt, &_resp));
        ++inIt;
    }
    ASSERT_NO_THROW(outputsInfo = _cnnNetwork.getOutputsInfo());
    auto outIt = _outputsInfo.begin();
    for (auto outputInfo : outputsInfo) {
        outputInfo.second->setPrecision(outIt->second->getTensorDesc().getPrecision());
        InferenceEngine::SizeVector outputDims = outputInfo.second->getTensorDesc().getDims();
        Blob::Ptr outputBlob = nullptr;
        Layout layout = outIt->second->getTensorDesc().getLayout();
        // work only with NHWC layout if size of the input dimensions == NHWC
        switch (outputInfo.second->getPrecision()) {
        case Precision::FP16:
            outputBlob = InferenceEngine::make_shared_blob<ie_fp16>({Precision::FP16, outputDims, layout});
            break;
        case Precision::FP32:
            outputBlob = InferenceEngine::make_shared_blob<float>({Precision::FP32, outputDims, layout});
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported precision for output. Supported FP16, FP32";
        }
        outputBlob->allocate();
        st = inferRequest->SetBlob(outputInfo.first.c_str(), outputBlob, &_resp);
        outputMap[outputInfo.first] = outputBlob;
        ASSERT_EQ((int) InferenceEngine::StatusCode::OK, st) << _resp.msg;
        ++outIt;
    }
    ASSERT_EQ(inferRequest->Infer(&_resp), InferenceEngine::OK);
    /* bound is too high !!!! investigation TBD */
    CompareCommonAbsolute(_outputMap.begin()->second, outputMap.begin()->second, 0.001);
}

const std::vector<int> s_classes = {20, 80};
