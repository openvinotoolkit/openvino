// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include <algorithm>
#include <fstream>
#include <regex>
#include <string>

#include "common_test_utils/data_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

struct conv_input {
    size_t w;
    size_t h;
    size_t c;
};

conv_input conv_input_cases[] = {
//    {16, 32, 1},
//    {16, 32, 2},
    {16, 32, 3},
};

struct conv_param_kernel {
    size_t krn_n;
    size_t BiasesCoeff() const {
        return krn_n;
    }
};

struct conv_pool_param {
    size_t poolStrideW;
    size_t poolStrideH;
    size_t poolWinW;
    size_t poolWinH;
};

conv_pool_param conv_pool_param_cases[]{
    //{1,1,1,1},
    {2, 2, 2, 2}
};

conv_param_kernel conv_param_kernel_cases[] = {
//    {1},
    {2},
//    {3},
//    {4},
//    {5},
//    {6},
//    {7},
//    {8},
//    {9},
//    {10},
//    {11},
//    {12},
//    {13},
//    {14},
//    {15},
//    {16},
//    {17},
};

struct conv_ir_params {
    size_t krn_w;
    size_t krn_h;
    size_t str_w;
    size_t str_h;

    std::string GetDeviceName() {
        return "GNA";
    }

    size_t KernelsCoeff(const conv_input& in, size_t krn_n) const {
        return krn_w * krn_h * krn_n * in.c;
    }

    SizeVector OutNCHW(const conv_input& in, size_t krn_n) const {
        SizeVector out = { 1,
            krn_n,
            (in.h - krn_h) / str_h + 1,
            (in.w - krn_w) / str_w + 1 };
        return out;
    }

    SizeVector OutPoolNCHW(const conv_input& in, size_t krn_n, const conv_pool_param& pool) const {
        auto o = OutNCHW(in, krn_n);
        o.at(2) = (o.at(2) - pool.poolWinH) / pool.poolStrideH + 1;
        o.at(3) = (o.at(3) - pool.poolWinW) / pool.poolStrideW + 1;;
        return o;
    }
};

struct conv_fill {
    int inFillPat;
    int filtersFillPat;

    static void FillData(float* data, size_t elements, int pattern) {
        std::mt19937 random_engine{};
        std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

        for (size_t i = 0; i < elements; i++) {
            switch (pattern) {
            case 0:
                data[i] = distribution(random_engine);
                break;
            case 1:
                data[i] = distribution(random_engine) / 100;
                break;
            case 2:
                data[i] = static_cast<float>(i % 5 - 2.0f);
                break;
            case 3:
                data[i] = i % 13 == 0 ? i % 3 - 2.0 : 0.0;
                break;
            default:
                data[i] = 0.0f;
            }
        }
    }
};
conv_fill conv_ir_fill_cases[] = {
    { 0, 1 },
//    { 2, 3 },
//    // { 0, 3 }, //TODO: check why failing
//    { 2, 1 },
};

typedef std::tuple< conv_ir_params, conv_fill, conv_input, conv_param_kernel, conv_pool_param> conv_ir_test_params;

conv_ir_params conv_ir_test_cases[] = {
//    conv_ir_params{1, 1, 1, 1},
//    conv_ir_params{2, 1, 1, 1},
//    conv_ir_params{2, 2, 1, 1},
//    conv_ir_params{2, 2, 2, 2},
//    conv_ir_params{2, 2, 1, 2},
//    conv_ir_params{2, 3, 1, 1},
//    conv_ir_params{2, 3, 2, 1},
//    conv_ir_params{2, 3, 2, 3},
    conv_ir_params{3, 1, 1, 1},
//    conv_ir_params{3, 3, 1, 1},
//    conv_ir_params{4, 1, 1, 1},
//    conv_ir_params{4, 2, 1, 1},
//    conv_ir_params{4, 4, 1, 1},
//    conv_ir_params{5, 1, 1, 1},
//    conv_ir_params{6, 1, 1, 1},
};

template <typename data_t>
void ref_conv_relu(const TBlob<data_t>& src, const data_t* weights, const size_t weightsSize,
    TBlob<data_t>& dst, conv_ir_params prm, const SizeVector& outNCHW,
    uint32_t poolWinH = 1, uint32_t poolWinW = 1,
    uint32_t poolStrideH = 1, uint32_t poolStrideW = 1) {
    size_t KW = prm.krn_w;
    size_t KH = prm.krn_h;
    const size_t GC = 1;

    size_t IW = src.getTensorDesc().getDims()[3];
    size_t IH = src.getTensorDesc().getDims()[2];
    size_t IC = src.getTensorDesc().getDims()[1];

    size_t OW = outNCHW.at(3);
    size_t OH = outNCHW.at(2);
    size_t OC = outNCHW.at(1);

    const data_t* src_data = src.readOnly();
    const data_t* weights_data = weights;
    const data_t* bias_data = weights_data + KW * KH * OC * IC / GC;
    data_t* dst_data = dst.data();
    bool poolingActive = poolStrideH != 1 || poolStrideW != 1 || poolWinH != 1 || poolWinW != 1;
    if (poolingActive) {
        dst_data = new float[OC * OH * OW];
    }

    IE_ASSERT(KW * KH * OC * IC / GC + OC == weightsSize);
    IE_ASSERT(OW == dst.getTensorDesc().getDims()[3]);
    IE_ASSERT(OH == dst.getTensorDesc().getDims()[2]);

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t oh = 0; oh < OH; oh++) {
                for (uint32_t ow = 0; ow < OW; ow++) {
                    size_t oidx = g * OC / GC * OH * OW
                        + oc * OH * OW + oh * OW + ow;
                    dst_data[oidx] = bias_data[g * OC / GC + oc];

                    for (size_t ic = 0; ic < IC / GC; ic++) {
                        for (size_t kh = 0; kh < KH; kh++) {
                            for (size_t kw = 0; kw < KW; kw++) {
                                int32_t iw = ow * prm.str_w + kw;
                                int32_t ih = oh * prm.str_h + kh;
                                if (iw < 0 || iw >= (int32_t)IW || ih < 0
                                    || ih >= (int32_t)IH)
                                    continue;
                                size_t iidx = g * IC / GC * IH * IW
                                    + ic * IH * IW + ih * IW + iw;
                                size_t widx = g * OC / GC * IC / GC * KH * KW
                                    + oc * IC / GC * KH * KW
                                    + ic * KH * KW + kh * KW + kw;

                                dst_data[oidx] += src_data[iidx] * weights_data[widx];
                            }
                        }
                    }
                    // Applying ReLU
                    if (dst_data[oidx] < 0) dst_data[oidx] = 0;
                }
            }
        }
    }

    if (poolingActive) {
        float *f = dst.data();
        uint32_t outPoolH = (OH - poolWinH) / poolStrideH + 1;
        uint32_t outPoolW = (OW - poolWinW) / poolStrideW + 1;
        for (uint32_t oc = 0; oc < OC; oc++) {
            for (uint32_t oh = 0; oh < outPoolH; oh++) {
                for (uint32_t ow = 0; ow < outPoolW; ow++) {
                    auto idx = oc * outPoolH * outPoolW + oh * outPoolW + ow;
                    f[idx] = 0;
                    for (uint32_t ph = 0; ph < poolWinH; ph++) {
                        for (uint32_t pw = 0; pw < poolWinW; pw++) {
                            auto idxp = oc * OH * OW + (oh * poolStrideH + ph) * OH + (ow * poolStrideW + pw);
                            f[idx] = (std::max)(f[idx], dst_data[idxp]);
                        }
                    }
                }
            }
        }
        delete[] dst_data;
    }
}

class smoke_ConvolutionIRTest : public testing::Test, public WithParamInterface<conv_ir_test_params> {
    const std::string modelIRTemplate = R"V0G0N(
<Net Name="Convolution_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>__IN_C__</dim>
                    <dim>__IN_H__</dim>
                    <dim>__IN_W__</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="Convolution" precision="FP32">
            <convolution stride-x="__STR_W__" stride-y="__STR_H__"
                         pad-x="0"    pad-y="0"
                         kernel-x="__KRN_W__" kernel-y="__KRN_H__"
                         output="__KRN_N__"  group="1"/>

            <weights offset="0" size="__KRN_BS__" />
            <biases offset="__KRN_BS__" size="__BIAS_BS__" />

            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>__IN_C__</dim>
                    <dim>__IN_H__</dim>
                    <dim>__IN_W__</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>__KRN_N__</dim>
                    <dim>__OUT_H__</dim>
                    <dim>__OUT_W__</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="conv1_relu" type="ReLU" precision="FP32">
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>__KRN_N__</dim>
                    <dim>__OUT_H__</dim>
                    <dim>__OUT_W__</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>__KRN_N__</dim>
                    <dim>__OUT_H__</dim>
                    <dim>__OUT_W__</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="conv1_pooling" type="Pooling" precision="FP32">
            <data exclude-pad="false" kernel="__POOL_WIN_W__,__POOL_WIN_H__" pads_begin="0,0" pads_end="0,0"
                  pool-method="max" rounding_type="ceil" strides="__POOL_STR_W__,__POOL_STR_H__" />
            <input>
                <port id="5">
                    <dim>1</dim>
                    <dim>__KRN_N__</dim>
                    <dim>__OUT_H__</dim>
                    <dim>__OUT_W__</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>1</dim>
                    <dim>__KRN_N__</dim>
                    <dim>__OUT_POOL_H__</dim>
                    <dim>__OUT_POOL_W__</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1" />
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3" />
        <edge from-layer="2" from-port="4" to-layer="3" to-port="5" />
    </edges>
</Net>
)V0G0N";

    void SubstringReplce(std::string& inPlace, std::string pattern, int value) {
        inPlace = std::regex_replace(inPlace, std::regex(pattern), std::to_string(value));
    }

    std::string GetModelIR(const conv_ir_params& p, const conv_input& in, const conv_param_kernel& ker, const conv_pool_param & pool) {
        auto model = modelIRTemplate;
        SubstringReplce(model, "__IN_C__", in.c);
        SubstringReplce(model, "__IN_H__", in.h);
        SubstringReplce(model, "__IN_W__", in.w);
        SubstringReplce(model, "__KRN_H__", p.krn_h);
        SubstringReplce(model, "__KRN_W__", p.krn_w);
        SubstringReplce(model, "__KRN_N__", ker.krn_n);
        SubstringReplce(model, "__KRN_BS__", sizeof(float) * p.KernelsCoeff(in, ker.krn_n));
        SubstringReplce(model, "__BIAS_BS__", sizeof(float) * ker.BiasesCoeff());
        SubstringReplce(model, "__OUT_H__", p.OutNCHW(in, ker.krn_n).at(2));
        SubstringReplce(model, "__OUT_W__", p.OutNCHW(in, ker.krn_n).at(3));
        SubstringReplce(model, "__STR_H__", p.str_h);
        SubstringReplce(model, "__STR_W__", p.str_w);

        SubstringReplce(model, "__OUT_POOL_H__", p.OutPoolNCHW(in, ker.krn_n, pool).at(2));
        SubstringReplce(model, "__OUT_POOL_W__", p.OutPoolNCHW(in, ker.krn_n, pool).at(3));

        SubstringReplce(model, "__POOL_WIN_W__", p.OutPoolNCHW(in, ker.krn_n, pool).at(2));
        SubstringReplce(model, "__POOL_WIN_H__", p.OutPoolNCHW(in, ker.krn_n, pool).at(3));
        SubstringReplce(model, "__POOL_STR_W__", p.OutPoolNCHW(in, ker.krn_n, pool).at(2));
        SubstringReplce(model, "__POOL_STR_H__", p.OutPoolNCHW(in, ker.krn_n, pool).at(3));
        return model;
    }
    TBlob<uint8_t>::Ptr readfile(std::string name) const {
        std::ifstream file{ name, std::ios::binary | std::ios::ate };
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        auto weights = new TBlob<uint8_t>{ TensorDesc{Precision::U8, { (size_t)size }, Layout::C} };
        weights->allocate();
        auto wPtr = TBlob<uint8_t>::Ptr(weights);
        if (file.read(weights->buffer().as<char*>(), size)) {
            return wPtr;
        }
        throw - 1;
    }

protected:
    static float GetNRMSD(const float* res_ptr, const float* ref_ptr, size_t size) {
        float sum = 0;
        const auto refStat = std::minmax_element(ref_ptr, ref_ptr + size);
        const auto range = (std::max)(*refStat.second - *refStat.first, 0.00001f);

        for (size_t i = 0; i < size; i++) {
            float sqr = (ref_ptr[i] - res_ptr[i]);
            sqr *= sqr;
            sum += sqr;
        }
        const auto MSD = sum / size;
        const auto RMSD = pow(MSD, 0.5f);

        const auto NRMSD = RMSD / range;
        return NRMSD;
    }

    virtual void SetUp() {
        try {
            auto allparams = ::testing::WithParamInterface<conv_ir_test_params>::GetParam();
            auto p = std::get<0>(allparams);
            auto fillParam = std::get<1>(allparams);
            auto inputParam = std::get<2>(allparams);
            auto kernelParam = std::get<3>(allparams);
            auto poolParam = std::get<4>(allparams);
            auto outNCHW = p.OutNCHW(inputParam, kernelParam.krn_n);
            // Prepare filters and biases
            const auto filtersCoefficients = p.KernelsCoeff(inputParam, kernelParam.krn_n);
            const auto BiasesCoefficients = kernelParam.BiasesCoeff();
            auto weights = new TBlob<uint8_t>{ TensorDesc{Precision::U8, { (filtersCoefficients + BiasesCoefficients) * sizeof(float) }, Layout::C} };
            weights->allocate();
            auto wTab = weights->buffer().as<float*>();

            conv_fill::FillData(wTab, filtersCoefficients + BiasesCoefficients, fillParam.filtersFillPat);

            TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);

            Core ie;
            auto model = GetModelIR(p, inputParam, kernelParam, poolParam);
            auto network = ie.ReadNetwork(model, weights_ptr);
            // Setting the statistics data

            CNNNetwork myNetwork = ie.ReadNetwork(model, weights_ptr);
            /** Taking information about all topology inputs **/
            InputsDataMap inputInfo(myNetwork.getInputsInfo());

            if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
            auto inputInfoItem = *inputInfo.begin();

            // SizeVector dims_src = { inputParam.w,
            //                        inputParam.h,
            //                        inputParam.c,
            //                        1 };          // 1 is a batch size
            const auto inputDimensions = inputInfoItem.second->getTensorDesc().getDims();
            Blob::Ptr src = make_shared_blob<float>(TensorDesc(Precision::FP32, inputDimensions, NCHW));
            src->allocate();
            std::fill_n(src->buffer().as<float*>(), src->size(), 0.0);

            /** Specifying the precision of input data provided by the user.
             * This should be called before load of the network to the plugin **/
            inputInfoItem.second->setPrecision(Precision::FP32);
            inputInfoItem.second->setLayout(Layout::NCHW);

            SizeVector outpoolNCHW = { 1, 1, 1, 1 };
            OutputsDataMap outputInfo(myNetwork.getOutputsInfo());
            for (auto itOut : outputInfo) {
                itOut.second->setPrecision(Precision::FP32);
                outpoolNCHW = itOut.second->getDims();
            }
            //auto outpoolNCHW = p.OutPoolNCHW(inputParam, kernelParam.krn_n, poolParam);
            Blob::Ptr dst = make_shared_blob<float>(TensorDesc(Precision::FP32, outpoolNCHW, NCHW));
            dst->allocate();

            size_t num_chanels = src->getTensorDesc().getDims()[1];
            size_t image_size = src->getTensorDesc().getDims()[2] * src->getTensorDesc().getDims()[3];

            float* data = src->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

            conv_fill::FillData(data, num_chanels * image_size, fillParam.inFillPat);

            // Inferring the converted network and comparing the result with the reference
            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.GetDeviceName());
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            OutputsDataMap outInfo;
            outInfo = network.getOutputsInfo();
            ASSERT_EQ(outInfo.size(), 1);
            ASSERT_NE(outInfo.begin()->second, nullptr);
            inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            inferRequest.SetBlob(outInfo.begin()->first, dst);

            inferRequest.Infer();

            // Calculating FP32 reference
            TBlob<float> dst_ref(TensorDesc(Precision::FP32, outNCHW, NCHW));
            dst_ref.allocate();
            auto* srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            ref_conv_relu<float>(*srcPtr, (const float*)weights->buffer(), weights->size() / sizeof(float), dst_ref, p, outNCHW,
                poolParam.poolWinH, poolParam.poolWinW, poolParam.poolStrideH, poolParam.poolStrideW);

            // Comparing the result with the reference
            const auto results = dst->buffer().as<const float*>();
            const auto reference = dst_ref.buffer().as<const float*>();
            EXPECT_EQ(dst->size(), dst_ref.size());
            const auto NRMSD = GetNRMSD(results, reference, dst->size());
            EXPECT_LE(NRMSD, 0.0218);
        }
        catch (const details::InferenceEngineException& e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_ConvolutionIRTest, TestsConvolution) {
}

static std::string  getTestCaseName(testing::TestParamInfo<conv_ir_test_params> obj) {
    auto p = std::get<0>(obj.param);
    auto pf = std::get<1>(obj.param);
    auto in = std::get<2>(obj.param);
    auto ker = std::get<3>(obj.param);
    auto pool = std::get<4>(obj.param);
    return  p.GetDeviceName() +
        "_w" + std::to_string(in.w) +
        "_h" + std::to_string(in.h) +
        "_c" + std::to_string(in.c) +
        "_krnw" + std::to_string(p.krn_w) +
        "_krnh" + std::to_string(p.krn_h) +
        "_krnn" + std::to_string(ker.krn_n) +
        "_strw" + std::to_string(p.str_w) +
        "_strh" + std::to_string(p.str_h) +
        "_fillInputs" + std::to_string(pf.inFillPat) +
        "_fillFilters" + std::to_string(pf.filtersFillPat) +
        "_poolSW" + std::to_string(pool.poolStrideW) +
        "_poolSH" + std::to_string(pool.poolStrideH) +
        "_poolWW" + std::to_string(pool.poolWinW) +
        "_poolWH" + std::to_string(pool.poolWinH);
}

INSTANTIATE_TEST_CASE_P(
    TestConvolution, smoke_ConvolutionIRTest, ::testing::Combine(
        ::testing::ValuesIn(conv_ir_test_cases),
        ::testing::ValuesIn(conv_ir_fill_cases),
        ::testing::ValuesIn(conv_input_cases),
        ::testing::ValuesIn(conv_param_kernel_cases),
        ::testing::ValuesIn(conv_pool_param_cases))
        , getTestCaseName);
