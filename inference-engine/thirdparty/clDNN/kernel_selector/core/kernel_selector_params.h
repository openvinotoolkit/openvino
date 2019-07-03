/*
// Copyright (c) 2016-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include <string>
#include <memory>
#include <cstddef>
#include "common_types.h"
#include "tensor_type.h"
#include "document.h"

namespace kernel_selector
{
    using DataTensor = Tensor::DataTensor;
    using WeightsTensor = Tensor::WeightsTensor;
    using DataLayout = Tensor::DataLayout;
    using WeightsLayout = Tensor::WeightsLayout;
    using MultiDataTensor = std::vector<DataTensor>;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ParamsKey
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ParamsKey
    {
    public:
        ParamsKey()
        {
            key.restrict.raw = 0;
            key.enableTuning = 1;
            key.machineInfo.raw = 0;
            key.inputType.raw = 0;
            key.outputType.raw = 0;
            key.inputWeightsType.raw = 0;
            key.outputWeightsType.raw = 0;
            key.inputLayout = 0;
            key.outputLayout = 0;
            key.weightsInputLayout = 0;
            key.weightsOutputLayout = 0;
        }

        struct Key
        {
            union restrict_t
            {
                struct val_t
                {
                    uint32_t different_types : 1;
                    uint32_t different_input_weights_types : 1;
                    uint32_t offset : 1;
                    uint32_t pitches : 1;
                    uint32_t batching : 1;
                    uint32_t biasPerFeatureMap : 1;
                    uint32_t biasPerOutput : 1;
                    uint32_t nonBias : 1;
                    uint32_t activationAdditionalParamsAsInput : 1;
                    uint32_t FP16Emulation : 1;
                    uint32_t gradient : 1;
                    uint32_t momentum : 1;

                    union dedicated_t
                    {
                        struct lookt_t
                        {
                            uint32_t axisX : 1;
                            uint32_t axisY : 1;
                            uint32_t axisFeature : 1;
                            uint32_t axisBatch : 1;
                            uint32_t axisXYF : 1;
                            uint32_t indicesF32 : 1;
                            uint32_t indicesOther : 1;
                        } lookt;
						struct argm_t
						{
							uint32_t axisX : 1;
							uint32_t axisY : 1;
							uint32_t axisFeature : 1;
							uint32_t axisBatch : 1;
							uint32_t axisXYF : 1;
						} argm;
                        struct idxsel_t
                        {
                            uint32_t axisX : 1;
                            uint32_t axisY : 1;
                            uint32_t axisFeature : 1;
                            uint32_t axisBatch : 1;
                        } idxsel;
                        struct norm_t
                        {
                            uint32_t across : 1;
                            uint32_t within : 1;
                            uint32_t fixedKenrelDivider : 1;
                            uint32_t dynamicKenrelDivider : 1;
                        } norm;
                        struct mvn_t
                        {
                            uint32_t across : 1;
                            uint32_t within : 1;
                            uint32_t normalize_variance : 1;
                        } mvn;
                        struct pooling_t
                        {
                            uint32_t max : 1;
                            uint32_t avg : 1;
                            uint32_t floor : 1;
                            uint32_t max_with_argmax : 1;
                            uint32_t ceil : 1;
                            uint32_t bilinear : 1;
                            uint32_t fixedKenrelDivider : 1;
                            uint32_t dynamicKenrelDivider : 1;
                            uint32_t dynamicKenrelDividerWithPadding : 1;
                            uint32_t position_sensitive : 1;
                        } pooling;
                        struct conv_t
                        {
                            uint32_t split : 1;
                            uint32_t dilation : 1;
                            uint32_t depthwise_separable_opt : 1;
                            uint32_t transposed : 1;
                            uint32_t quantization : 1;
                            uint32_t calibration : 1;
                            uint32_t local : 1;
                            uint32_t grouped : 1;
                        } conv;
                        struct fc_t {} fc;
                        struct softmax_t
                        {
                            uint32_t dimX : 1;
                            uint32_t dimY : 1;
                            uint32_t dimFeature : 1;
                        } softmax;
                        struct region_yolo_t
                        {
                            uint32_t dimX : 1;
                            uint32_t dimY : 1;
                            uint32_t dimFeature : 1;
                            uint32_t coords : 1;
                            uint32_t classes : 1;
                            uint32_t num : 1;
                        } region_yolo;
                        struct reorg_yolo_t
                        {
                            uint32_t dimX : 1;
                            uint32_t dimY : 1;
                            uint32_t dimFeature : 1;
                            uint32_t stride : 1;
                        } reorg_yolo;
                        struct concat_t
                        {
                            uint32_t axisX : 1;
                            uint32_t axisY : 1;
                            uint32_t axisFeature : 1;
                            uint32_t axisBatch : 1;
                            uint32_t kernelPerInput : 1;
                            uint32_t oneKernel : 1;
                        } concat;
                        struct upsample_t
                        {
                            uint32_t nearest : 1;
                            uint32_t bilinear : 1;
                        } upsample;
                        struct reorder_t
                        {
                            uint32_t winograd : 1;
                        } reorder;
                        struct eltwise_t
                        {
                            uint32_t stride : 1;
                            uint32_t broadcast : 1;
                        } eltwise;
                        struct lstm_gemm_t {
                            uint32_t bias : 1;
                            uint32_t hidden : 1;
                        } lstm_gemm;
                        struct lstm_elt_t {
                            uint32_t cell : 1;
                        } lstm_elt;
                        struct fused_conv_eltw_t {
                            // conv
                            uint32_t split : 1;
                            uint32_t dilation : 1;
                            uint32_t depthwise_separable_opt : 1;
                            uint32_t transposed : 1;
                            uint32_t quantization : 1;
                            uint32_t calibration : 1;
                            uint32_t local : 1;
                            uint32_t grouped : 1;
                            // eltw
                            uint32_t stride : 1;
                            // fused conv eltw
                            uint32_t rw_out_opt : 1;
                        } fused_conv_eltw;
                    } dedicated;
                } val;
                uint64_t raw;
            } restrict;

            union machine_info_t
            {
                struct val_t
                {
                    uint32_t subgroup : 1;
                    uint32_t subgroupShort : 1;
                } val;
                uint32_t raw;
            } machineInfo;

            static_assert(sizeof(restrict_t) == sizeof(uint64_t), "problem with union");

            typedef union DataTypesKey_t
            {
                struct val_t
                {
                    uint32_t int8 : 1;
                    uint32_t uint8 : 1;
                    uint32_t int16 : 1;
                    uint32_t uint16 : 1;
                    uint32_t int32 : 1;
                    uint32_t uint32 : 1;
                    uint32_t int64 : 1;
                    uint32_t F16 : 1;
                    uint32_t F32 : 1;
                } val;
                uint32_t raw;
            } DataTypesKey;

            uint32_t enableTuning;
            DataTypesKey inputType;
            DataTypesKey outputType;
            DataTypesKey inputWeightsType;
            DataTypesKey outputWeightsType;
            uint32_t inputLayout;
            uint32_t outputLayout;
            uint32_t weightsInputLayout;
            uint32_t weightsOutputLayout;
        };

        void EnableInputDataType(Datatype dt);
        void EnableAllInputDataType();
        void EnableOutputDataType(Datatype dt);
        void EnableAllOutputDataType();
        void EnableInputWeightsType(WeightsType wt);
        void EnableAllInputWeightsType();
        void EnableOutputWeightsType(WeightsType wt);
        void EnableAllOutputWeightsType();
        void EnableFP16Emulation() { key.restrict.val.FP16Emulation = 1; }
        void EnableDifferentTypes() { key.restrict.val.different_types = 1; }
        void EnableDifferentInputWeightsTypes() {
            key.restrict.val.different_input_weights_types = 1; }
        void EnableInputLayout(DataLayout l) { key.inputLayout |= (1 << l); }
        void EnableAllInputLayout() { key.inputLayout = 0xffffffff; }
        void EnableOutputLayout(DataLayout l) { key.outputLayout |= (1 << l); }
        void EnableAllOutputLayout() { key.outputLayout = 0xffffffff; }
        void EnableInputWeightsLayout(WeightsLayout l) { key.weightsInputLayout |= (1 << l); }
        void EnableAllInputWeightsLayout() { key.weightsInputLayout = 0xffffffff; }
        void EnableOutputWeightsLayout(WeightsLayout l) { key.weightsOutputLayout |= (1 << l); }
        void EnableAllOutputWeightsLayout() { key.weightsOutputLayout = 0xffffffff; }
        void EnableTensorOffset() { key.restrict.val.offset = 1; }
        void EnableTensorPitches() { key.restrict.val.pitches = 1; }
        void EnableBatching() { key.restrict.val.batching = 1; }
        void EnableGradient() { key.restrict.val.gradient = 1; }
        void EnableSubGroup() { key.machineInfo.val.subgroup = 1; }
        void EnableSubGroupShort() { key.machineInfo.val.subgroupShort = 1; }
        void EnableNonBiasTerm() { key.restrict.val.nonBias = 1; }
        void EnableBiasPerFeature() { key.restrict.val.biasPerFeatureMap = 1; }
        void EnableBiasPerOutput() { key.restrict.val.biasPerOutput = 1; }
        void EnableActivationAdditionalParamsAsInput() { key.restrict.val.activationAdditionalParamsAsInput = 1; }
        void EnableMomentum() { key.restrict.val.momentum = 1; }
        void EnableLRNMode(LRNMode m);
        void EnableLookUpTableAxis(LookUpTableAxis m);
        void EnableNormalizeMode(NormalizeMode m);
        void EnableMVNMode(MVNMode m);
        void EnableMVNNormalizeVariance();
        void EnableLRNKernelDividerMode(KernelDividerMode m);
        void EnablePoolKernelDividerMode(KernelDividerMode m);
        void EnablePoolType(PoolType t);
        void EnablePoolRemainder(PoolRemainder r);
        void EnablePositionSensitivePooling() { key.restrict.val.dedicated.pooling.position_sensitive = 1; }
        void EnableSplitSupport() { key.restrict.val.dedicated.conv.split = 1; }
        void EnableDilation() { key.restrict.val.dedicated.conv.dilation = 1; }
        void EnableDepthwiseSeparableOpt() { key.restrict.val.dedicated.conv.depthwise_separable_opt = 1; }
        void EnableLocalConvolution() { key.restrict.val.dedicated.conv.local = 1; }
        void EnableGroupedConvolution() { key.restrict.val.dedicated.conv.grouped = 1; }
        void EnableTranspose() { key.restrict.val.dedicated.conv.transposed = 1; }
        void EnableInt8Quantization() { key.restrict.val.dedicated.conv.quantization = 1; }
        void EnableOutputCalibration() { key.restrict.val.dedicated.conv.calibration = 1; }

        void EnableFusedConvEltwSplitSupport() { key.restrict.val.dedicated.fused_conv_eltw.split = 1; }
        void EnableFusedConvEltwDilation() { key.restrict.val.dedicated.fused_conv_eltw.dilation = 1; }
        void EnableFusedConvEltwDepthwiseSeparableOpt() { key.restrict.val.dedicated.fused_conv_eltw.depthwise_separable_opt = 1; }
        void EnableFusedConvEltwLocalConvolution() { key.restrict.val.dedicated.fused_conv_eltw.local = 1; }
        void EnableFusedConvEltwGroupedConvolution() { key.restrict.val.dedicated.fused_conv_eltw.grouped = 1; }
        void EnableFusedConvEltwTranspose() { key.restrict.val.dedicated.fused_conv_eltw.transposed = 1; }
        void EnableFusedConvEltwInt8Quantization() { key.restrict.val.dedicated.fused_conv_eltw.quantization = 1; }
        void EnableFusedConvEltwOutputCalibration() { key.restrict.val.dedicated.fused_conv_eltw.calibration = 1; }
        void EnableFusedConvEltwEltwiseStride();

        void EnableWinogradReorder() { key.restrict.val.dedicated.reorder.winograd = 1; }
        void EnableSoftmaxDim(SoftmaxDim d);
        void EnableConcatAxis(ConcatAxis a);
        void EnableUpSamplingSampleType(SampleType a);
        void EnableEltwiseStride();
        void EnableEltwiseBroadcast() { key.restrict.val.dedicated.eltwise.broadcast = 1; }
        void EnableLSTMGEMMBias() { key.restrict.val.dedicated.lstm_gemm.bias = 1; }
        void EnableLSTMGEMMHidden() { key.restrict.val.dedicated.lstm_gemm.hidden = 1; }
        void EnableLSTMEltCell() { key.restrict.val.dedicated.lstm_elt.cell = 1; }
        void EnableConcatKernelPerInput() { key.restrict.val.dedicated.concat.kernelPerInput = 1; }
        void DisableTuning() { key.enableTuning = 0; }
        void EnableConcatOneKernel() { key.restrict.val.dedicated.concat.oneKernel = 1; }
        void EnableArgMaxMinAxis(ArgMaxMinAxis a);
        void EnableLookUpTableIndicesFormat(Datatype a);
        void EnableIndexSelectAxis(IndexSelectAxis a);
        void EnableFusedConvEltwiseRWOutOpt();
        bool Support(const ParamsKey& k) const;
        bool TuningSupport() const
        {
            if (key.enableTuning == 1)
                return true;
            return false;
        }
        bool isEnabledDifferentInputWeightsTypes() const {
            return key.restrict.val.different_input_weights_types ? true : false;
        }
        ParamsKey Merge(const ParamsKey& k) const;

    private:
        Key key;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EngineInfo
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EngineInfo
    {
        bool bSubGroupSupport = false;
        bool bSubGroupShortSupport = false;
        bool bFP16Support = false;
        bool bFP64Support = false;
        bool bImageSupport = false;
        bool bIMADSupport = false;
        bool bIMMADSupport = false;
        uint32_t computeUnitsCount = 0;
        uint64_t maxWorkGroupSize = 0;
        uint64_t maxLocalMemSize = 0;
        uint64_t maxImage2dWidth = 0;
        uint64_t maxImage2dHeight = 0;
        std::string deviceId = "";
        std::string driverVersion = "";
        std::string hostVersion = "";
        std::shared_ptr<rapidjson::Document> deviceCache;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Params
    {
        virtual ~Params() {}

        KernelType GetType() const { return kType; }
        virtual ParamsKey GetParamsKey() const;

    protected:
        Params(KernelType kt, const std::string& id) : kType(kt), layerID(id) {}
        KernelType kType;

    public:
        std::string layerID;
        EngineInfo engineInfo;

        virtual std::string to_string() const;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // base_activation_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct base_activation_params
    {
        ActivationFunction  function = ActivationFunction::NONE;
        float m = 1.f;
        float n = 0.f;

        base_activation_params() = default;
        base_activation_params(const float m, const float n) : m(m), n(n) {}

        virtual std::string to_string() const;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // base_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct base_params : public Params
    {
        virtual ~base_params() {}

        base_activation_params activation;
        MultiDataTensor        inputs;
        DataTensor             output;
        bool                   gradient = false;

        virtual std::string to_string() const;
        virtual ParamsKey GetParamsKey() const;
    protected:

        base_params(KernelType kt) : Params(kt, ""), inputs(1){}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Auto tuner parameters
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class KernelRunnerInterface;
    struct TuningParams
    {
        TuningMode mode;
        std::string cacheFilePath;
        std::shared_ptr<KernelRunnerInterface> runner;

        TuningParams() : mode(TuningMode::TUNING_DISABLED), cacheFilePath(""), runner(nullptr) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct optional_params
    {
        virtual ~optional_params() {}

        KernelType GetType() const { return kType; }

        std::vector<DataLayout> inputLayouts;
        std::vector<DataLayout> outputLayouts;

        bool meaningfulKernelsNames     = false;    // use layer name instead of internal kernel name
        bool allowStaticInputReordering = true;     // allow kernel to provide a kernel which reorder static data like weights/bias/tables...
        bool allowInputReordering       = false;    // allow kernel to ask graph compiler to reorder the input data before executing its
        bool allowOutputReordering      = false;    // allow kernel to ask graph compiler to reorder the output data before executing the next kernel

        TuningParams tuningParams;

        virtual ParamsKey GetSupportedKey() const;
    protected:
        optional_params(KernelType kt) : kType(kt) {}
        KernelType kType;
    };
}
