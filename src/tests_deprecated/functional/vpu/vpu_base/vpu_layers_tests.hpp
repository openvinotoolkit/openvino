// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <tuple>

#include <ie_version.hpp>
#include <precision_utils.h>

#include <vpu/private_plugin_config.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/ie_helpers.hpp>

#include "ie_core_adapter.hpp"
#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "vpu_layer_tests_utils.hpp"
#include "vpu_test_net.hpp"

class vpuLayersTests : public TestsCommon {
protected:
    class NetworkParams {
    public:
        vpu::LayoutPreference      _layoutPreference = vpu::LayoutPreference::ChannelMajor;
        InferenceEngine::Precision _outputPrecision  = InferenceEngine::Precision::FP16;
        InferenceEngine::Precision _inputPrecision   = InferenceEngine::Precision::FP16;

        bool _useHWOpt = false;
        // For historical reasons, createInferRequest() function use to 'hack' blob layout:
        // replace NCHW with NHWC even if you explicitly setup layout preference
        // be channel-major. To disable this hack, please set lockLayout = true
        bool _lockLayout = false;

        bool _runRefGraph = true;
        bool _createInference = true;
    };

    class NetworkInitParams : public NetworkParams {
    public:
        NetworkInitParams& layoutPreference(const vpu::LayoutPreference layoutPreference)
            { _layoutPreference = layoutPreference; return *this;}
        NetworkInitParams& outputPrecision(InferenceEngine::Precision outputPrecision)
            { _outputPrecision = outputPrecision; return *this;}
        NetworkInitParams& inputPrecision(InferenceEngine::Precision inputPrecision)
            { _inputPrecision = inputPrecision; return *this;}

        NetworkInitParams& useHWOpt(const bool useHWOpt)
            { _useHWOpt = useHWOpt; return *this;}
        NetworkInitParams& lockLayout(const bool lockLayout)
            { _lockLayout = lockLayout; return *this;}
        NetworkInitParams& runRefGraph(const bool runRefGraph)
            { _runRefGraph = runRefGraph; return *this;}
        NetworkInitParams& createInference(const bool createInference)
            { _createInference = createInference; return *this;}
    };
    using DataGenerator   = void (*)(InferenceEngine::Blob::Ptr blob);
    using LayerParams     = VpuTestNet::LayerParams;
    using LayerInitParams = VpuTestNet::LayerInitParams;

protected:
    void SetUp() override;
    void TearDown() override;
    bool CheckMyriadX();
    void dumpPerformance();
    bool wasCustomLayerInferred() const;

    // For historical reasons, gen-blob functions use to 'hack' blob layout:
    // replace NCHW with NHWC even if you explicitly setup layout preference
    // be channel-major. To disable this hack, please set lockLayout = true
    void genInputBlobs(bool lockLayout = false);
    void genOutputBlobs(bool lockLayout = false);
    void genRefBlob(bool lockLayout = false);

    void ReferenceGraph();
    bool Infer();
    bool generateNetAndInfer(const NetworkParams& params);
    void ResetGeneratedNet();
    void ResetReferenceLayers();

    void SetInputReshape();
    void SetInputTensor(const tensor_test_params& tensor);
    void SetInputTensor(const tensor_test_params_3d& tensor);
    void SetOutputTensor(const tensor_test_params& tensor);
    void SetOutputTensor(const tensor_test_params_3d& tensor);
    void SetInputTensors(const IN_OUT_desc& in_tensors);
    void SetOutputTensors(const IN_OUT_desc& out_tensors);

    void SetFirstInputToRange(float start,
                              float finish);

    void SetInputInOrder();
    void SetInputInOrderReverse();
    void SetSeed(uint32_t seed);

    InferenceEngine::Blob::Ptr getReferenceOutput();

    void genNetwork();
    void makeSingleLayerNetworkImpl(const LayerParams& layerParams,
                       const NetworkParams& networkParams,
                       const WeightsBlob::Ptr& weights = nullptr);

    void readNetwork(const std::string& model, const WeightsBlob::Ptr& modelWeights = nullptr);
    void readNetwork(const std::string& modelFilename, const std::string& weightsFilename);
    void createInferRequest(const NetworkParams& params);

protected:
    IECoreAdapter::Ptr                             _vpuPluginPtr;

    std::map<std::string, std::string>             _config;

    IRVersion                                      _irVersion = IRVersion::v7;

    InferenceEngine::CNNNetwork                    _cnnNetwork;
    InferenceEngine::InputsDataMap                 _inputsInfo;
    InferenceEngine::BlobMap                       _inputMap;
    InferenceEngine::BlobMap                       _outputMap;
    InferenceEngine::OutputsDataMap                _outputsInfo;
    InferenceEngine::ExecutableNetwork             _exeNetwork;
    InferenceEngine::InferRequest                  _inferRequest;

    InferenceEngine::Blob::Ptr                     _refBlob;
    VpuTestNet                                     _testNet;

    DataGenerator                                  _genDataCallback0 = nullptr;
    DataGenerator                                  _genDataCallback = GenRandomData;

private:
    IN_OUT_desc                                    _inputTensors;
    IN_OUT_desc                                    _outputTensors;
    bool                                           _doReshape = false;  // reshape 4D input to layer input Tensor
};
