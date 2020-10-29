// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include <single_layer_common.hpp>
#include <string>

using namespace ::testing;
using namespace InferenceEngine;
using std::vector;

struct bin_conv_base_params {
    vector<size_t> in_dims;
    vector<size_t> kernel;
    vector<size_t> strides;
    vector<size_t> pads_begin;
    vector<size_t> pads_end;
    vector<size_t> dilations;

    size_t out_c;
    size_t grp_c;

    vector<size_t> out_dims;

    float pad_value;
};

struct bin_conv_test_params : bin_conv_base_params {
    std::string device_name;

    bin_conv_test_params(std::string name, bin_conv_base_params params) :
            bin_conv_base_params(params), device_name(name) {}

};

class BinaryConvolutionOnlyTest : public TestsCommon,
                            public WithParamInterface<bin_conv_test_params> {

    std::string model_t_4D = R"V0G0N(
<net name="BinaryConvolution_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="BinaryConvolution" precision="FP32">
            <data strides="_KS_"
                         pads_begin="_PB_" pads_end="_PE_"
                         kernel="_K_"
                         dilations="_DL_"
                         input="_IC_" output="_OC_" group="_GC_"
                         pad_value="_PV_" mode="_M_"/>

            <weights offset="0" size="_S1_" />

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

protected:

    static void fill_data_bin(float *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = sinf((float)i) > 0.f ? 1.f : -1.f;
        }
    }

    static void fill_data_bin_packed(int8_t *data, size_t size) {
        int nbits = 8;
        for (size_t i = 0; i < div_up(size, nbits); i++) {
            data[i] = static_cast<int8_t>(i % 255);
        }
    }

    size_t calculateOutDim(size_t in_dim, size_t kernel, size_t stride, size_t pad_begin) {
        return (in_dim + 2lu * pad_begin - kernel) / stride + 1lu;
    }

    void createBlobs(const bin_conv_test_params &p, TBlob<float>::Ptr &src, TBlob<float>::Ptr &dst, TBlob<float>::Ptr &dst_ref) {
        auto in_size = p.in_dims.size();
        auto out_size = p.out_dims.size();
        SizeVector dims_src;
        for (int i = in_size; i > 0; i--) {
            dims_src.insert(dims_src.begin(), p.in_dims[i - 1]);
        }

        SizeVector dims_dst = {
            1lu,
            p.out_c,
            p.out_dims[out_size - 2] == 0 ? calculateOutDim(p.in_dims[in_size - 2], p.kernel[Y_AXIS], p.strides[Y_AXIS], p.pads_begin[Y_AXIS]) : p.out_dims[out_size - 2],
            p.out_dims[out_size - 1] == 0 ? calculateOutDim(p.in_dims[in_size - 1], p.kernel[X_AXIS], p.strides[X_AXIS], p.pads_begin[X_AXIS]) : p.out_dims[out_size - 1]
        };

        Layout layout = NCHW;
        if (in_size == 5) {
            layout = NCDHW;

            dims_dst.insert(dims_dst.begin() + 3,
                p.out_dims.size() > 2 ?
                (p.out_dims[out_size - 3] == 0 ?
                    calculateOutDim(p.in_dims[in_size - 3], p.kernel[Z_AXIS], p.strides[Z_AXIS], p.pads_begin[Z_AXIS]) : p.out_dims[out_size - 3]) : 1lu);
        }

        src = make_shared_blob<float>({Precision::FP32, dims_src, layout});
        src->allocate();

        dst = make_shared_blob<float>({Precision::FP32, dims_dst, layout});
        dst->allocate();

        dst_ref = make_shared_blob<float>({Precision::FP32, dims_dst, layout});
        dst_ref->allocate();
    }

    TBlob<uint8_t>::Ptr fillWeights(const bin_conv_test_params &p) {
        auto KZ = p.kernel.size() > Z_AXIS ? p.kernel[Z_AXIS] : 1lu;
        TBlob<uint8_t> *weights_ptr = new TBlob<uint8_t>({Precision::BIN,
                    {(p.kernel[X_AXIS] * p.kernel[Y_AXIS] * KZ * p.out_c * p.in_dims[1] / p.grp_c + p.out_c)},
                    Layout::C});
        weights_ptr->allocate();
        fill_data_bin_packed(weights_ptr->buffer(), weights_ptr->size());
        return TBlob<uint8_t>::Ptr(weights_ptr);
    }


    struct bin_conv_common_params {
        InferenceEngine::PropertyVector<unsigned int> stride;
        InferenceEngine::PropertyVector<unsigned int> kernel;
        InferenceEngine::PropertyVector<unsigned int> pads_begin;
        InferenceEngine::PropertyVector<unsigned int> pads_end;
        InferenceEngine::PropertyVector<unsigned int> dilation;
        std::string auto_pad;
        size_t group;
        size_t out_c;
        float pad_value;
    };

    void ref_bin_conv_common(const Blob& src,
                         Blob& dst,
                         const uint8_t* weights_data,
                         const bin_conv_common_params& prm) {
        if (src.getTensorDesc().getLayout() != Layout::NCHW &&
            dst.getTensorDesc().getLayout() != Layout::NCDHW)
            THROW_IE_EXCEPTION << "Reference FP32 convolution supports NCHW and NCDHW layouts only";
        size_t KW = prm.kernel[X_AXIS];
        size_t KH = prm.kernel[Y_AXIS];
        size_t KD = prm.kernel.size() > Z_AXIS ? prm.kernel[Z_AXIS] : 1lu;

        size_t SW = prm.stride[X_AXIS];
        size_t SH = prm.stride[Y_AXIS];
        size_t SD = prm.stride.size() > Z_AXIS ? prm.stride[Z_AXIS] : 0lu;

        size_t DW = prm.dilation[X_AXIS];
        size_t DH = prm.dilation[Y_AXIS];
        size_t DD = prm.dilation.size() > Z_AXIS ? prm.dilation[Z_AXIS] : 0lu;

        size_t PW = prm.pads_begin[X_AXIS];
        size_t PH = prm.pads_begin[Y_AXIS];
        size_t PD = prm.pads_begin.size() > Z_AXIS ? prm.pads_begin[Z_AXIS] : 0lu;

        size_t GC = prm.group;

        auto src_dims = src.getTensorDesc().getDims();
        size_t IW, IH, ID, IC = src_dims[1];

        if (src_dims.size() == 5lu) {
            IW = src_dims[4];
            IH = src_dims[3];
            ID = src_dims[2];
        } else {
            IW = src_dims[3];
            IH = src_dims[2];
            ID = 1lu;
        }

        auto dst_dims = dst.getTensorDesc().getDims();
        size_t OW, OH, OD;
        size_t OC = prm.out_c;

        if (dst_dims.size() == 5lu) {
            OW = dst_dims[4];
            OH = dst_dims[3];
            OD = dst_dims[2];
        }
        else {
            OW = dst_dims[3];
            OH = dst_dims[2];
            OD = 1lu;
        }

        const auto* src_data = src.cbuffer().as<const float*>();
        auto* dst_data = dst.buffer().as<float*>();

        int nbits = 8;

        auto extract_weights = [](uint8_t val, uint8_t bit) -> float {
            return (uint8_t)((val >> bit) & 0x0001) == 1 ? 1.f : -1.f;
        };

        for (uint32_t g = 0; g < GC; g++) {
            for (uint32_t oc = 0; oc < OC / GC; oc++) {
                for (uint32_t od = 0; od < OD; od++) {
                    for (uint32_t oh = 0; oh < OH; oh++) {
                        for (uint32_t ow = 0; ow < OW; ow++) {
                            size_t oidx = g * OC / GC * OD * OH * OW
                                          + oc * OD * OH * OW
                                          + od * OH * OW
                                          + oh * OW
                                          + ow;

                            dst_data[oidx] = 0.f;

                            for (size_t ic = 0; ic < IC / GC; ic++) {
                                for (size_t kd = 0; kd < KD; kd++) {
                                    for (size_t kh = 0; kh < KH; kh++) {
                                        for (size_t kw = 0; kw < KW; kw++) {
                                            size_t widx = g * OC / GC * IC / GC * KD * KH * KW
                                                          + oc * IC / GC * KD * KH * KW
                                                          + ic * KD * KH * KW
                                                          + kd * KH * KW
                                                          + kh * KW
                                                          + kw;
                                            float w = extract_weights(weights_data[widx/nbits], (uint8_t)(widx % nbits));

                                            float s;

                                            int32_t iw = ow * SW - PW + kw * DW;
                                            int32_t ih = oh * SH - PH + kh * DH;
                                            int32_t id = od * SD - PD + kd * DD;
                                            if (iw < 0 || iw >= (int32_t) IW ||
                                                ih < 0 || ih >= (int32_t) IH ||
                                                id < 0 || id >= (int32_t) ID) {
                                                s = prm.pad_value;
                                            } else {
                                                size_t iidx = g * IC / GC * ID * IH * IW
                                                              + ic * ID * IH * IW
                                                              + id * IH * IW
                                                              + ih * IW
                                                              + iw;
                                                s = src_data[iidx];
                                            }

                                            dst_data[oidx] += s * w;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void calculateRef(const TBlob<uint8_t>::Ptr &weights, const bin_conv_test_params &p, const TBlob<float>::Ptr &src,
                      TBlob<float>::Ptr &dst_ref) {
        const uint8_t *weights_data = (const uint8_t *)weights->buffer();
        size_t bias_size = p.out_c;
        bin_conv_common_params params;
        for (int i = 0; i < p.kernel.size(); i++)
            params.kernel.insert(i, p.kernel[i]);
        for (int i = 0; i < p.strides.size(); i++)
            params.stride.insert(i, p.strides[i]);
        for (int i = 0; i < p.pads_begin.size(); i++)
            params.pads_begin.insert(i, p.pads_begin[i]);
        for (int i = 0; i < p.dilations.size(); i++)
            params.dilation.insert(i, p.dilations[i]);
        params.group = p.grp_c;
        params.out_c = p.out_c;
        params.pad_value = p.pad_value;

        ref_bin_conv_common(*src.get(), *dst_ref.get(), weights_data, params);
    }

    CNNNetwork getNetwork(const TBlob<uint8_t>::Ptr &weights, const bin_conv_test_params &p) {
        Core ie;
        return ie.ReadNetwork(getModel(p), weights);
    }

    virtual void
    infer(CNNNetwork &network, const bin_conv_test_params &p, TBlob<float>::Ptr &src, TBlob<float>::Ptr &dst) {
        Core ie;
        ExecutableNetwork executable_network = ie.LoadNetwork(network, p.device_name);
        InferRequest inferRequest = executable_network.CreateInferRequest();

        InputsDataMap inputInfo(network.getInputsInfo());
        inferRequest.SetBlob(inputInfo.begin()->first, src);

        OutputsDataMap outputInfo(network.getOutputsInfo());
        inferRequest.SetBlob(outputInfo.begin()->first, dst);

        inferRequest.Infer();
    }

    void SetUp() override {
        try {
            auto p = ::testing::WithParamInterface<bin_conv_test_params>::GetParam();
            TBlob<float>::Ptr src, dst, dst_ref;

            createBlobs(p, src, dst, dst_ref);
            fill_data_bin(src->data(), src->size());

            auto weights = fillWeights(p);
            calculateRef(weights, p, src, dst_ref);

            CNNNetwork network = getNetwork(weights, p);
            infer(network, p, src, dst);

            compare(*dst, *dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }

    virtual std::string getModel(bin_conv_test_params p) {
        std::string model;
        auto in_dims_size = p.in_dims.size();
        model = model_t_4D;

        REPLACE_WITH_NUM(model, "_IW_", p.in_dims[in_dims_size - 1]);
        REPLACE_WITH_NUM(model, "_IH_", p.in_dims[in_dims_size - 2]);
        REPLACE_WITH_NUM(model, "_ID_", p.in_dims[in_dims_size - 3]);
        REPLACE_WITH_NUM(model, "_IC_", p.in_dims[1]);
        REPLACE_WITH_NUM(model, "_IN_", p.in_dims[0]);

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.strides);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.pads_end);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_DL_", p.dilations);

        auto out_dims_size = p.out_dims.size();
        REPLACE_WITH_NUM(model, "_GC_", p.grp_c);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);
        REPLACE_WITH_NUM(model, "_OD_", out_dims_size > 2 ?
                (p.out_dims[out_dims_size - 3] == 0 ?
                    calculateOutDim(p.in_dims[in_dims_size - 3], p.kernel[Z_AXIS], p.strides[Z_AXIS], p.pads_begin[Z_AXIS]) : p.out_dims[out_dims_size - 3]) :
                        1lu);
        REPLACE_WITH_NUM(model, "_OH_", p.out_dims[out_dims_size - 2] == 0 ?
                calculateOutDim(p.in_dims[in_dims_size - 2], p.kernel[Y_AXIS], p.strides[Y_AXIS], p.pads_begin[Y_AXIS]) : p.out_dims[out_dims_size - 2]);
        REPLACE_WITH_NUM(model, "_OW_", p.out_dims[out_dims_size - 1] == 0 ?
                calculateOutDim(p.in_dims[in_dims_size - 1], p.kernel[X_AXIS], p.strides[X_AXIS], p.pads_begin[X_AXIS]) : p.out_dims[out_dims_size - 1]);

        size_t KD = p.kernel.size() > Z_AXIS ? p.kernel[Z_AXIS] : 1lu;

        int nbits = 8;
        size_t w_data_size = div_up(p.kernel[X_AXIS] * p.kernel[Y_AXIS] * KD * p.out_c * p.in_dims[1] / p.grp_c, nbits);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);

        REPLACE_WITH_NUM(model, "_PV_", p.pad_value);
        REPLACE_WITH_STR(model, "_M_", "xnor-popcount");

        return model;
    }
};

#define case_1  bin_conv_base_params({{1lu, 9lu, 32lu, 16lu},  {2lu, 4lu}, {1lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}, -1.f})
#define case_2  bin_conv_base_params({{1lu, 9lu, 32lu, 16lu},  {2lu, 4lu}, {1lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}, 0.f})
#define case_3  bin_conv_base_params({{1lu, 9lu, 32lu, 16lu},  {2lu, 4lu}, {2lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}, -1.f})
#define case_4  bin_conv_base_params({{1lu, 9lu, 32lu, 16lu},  {2lu, 4lu}, {2lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}, 0.f})
#define case_5  bin_conv_base_params({{1lu, 9lu, 32lu, 16lu},  {2lu, 4lu}, {2lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}, 1.f})
#define case_6  bin_conv_base_params({{1lu, 3lu, 40lu, 40lu},  {3lu, 3lu}, {1lu, 2lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 20lu, 1lu, {0lu, 0lu}, 0.f})
#define case_7  bin_conv_base_params({{1lu, 9lu, 16lu, 32lu},  {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}, -1.f})
#define case_8  bin_conv_base_params({{1lu, 9lu, 16lu, 32lu},  {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}, 0.f})
#define case_9  bin_conv_base_params({{1lu, 9lu, 16lu, 32lu},  {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}, 1.f})
#define case_10 bin_conv_base_params({{1lu, 16lu, 40lu, 40lu}, {3lu, 3lu}, {1lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 16lu, 16lu, {0lu, 0lu}, 0.f})
#define case_11 bin_conv_base_params({{1lu, 32lu, 16lu, 32lu}, {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {1lu, 1lu}, 32lu, 32lu, {0lu, 0lu}, 0.f})
#define case_12 bin_conv_base_params({{1lu, 16lu, 40lu, 40lu}, {3lu, 3lu}, {1lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {9lu, 9lu}, 16lu, 16lu, {0lu, 0lu}, 0.f})
#define case_13 bin_conv_base_params({{1lu, 32lu, 16lu, 32lu}, {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {9lu, 9lu}, 32lu, 32lu, {0lu, 0lu}, 0.f})
#define case_14 bin_conv_base_params({{1lu, 19lu, 16lu, 32lu}, {3lu, 3lu}, {1lu, 1lu}, {1lu, 1lu}, {1lu, 1lu}, {1lu, 1lu}, 21lu, 1lu, {0lu, 0lu}, -1.f})
#define case_15 bin_conv_base_params({{1lu, 17lu, 16lu, 32lu}, {3lu, 3lu}, {1lu, 1lu}, {1lu, 1lu}, {1lu, 1lu}, {1lu, 1lu}, 19lu, 1lu, {0lu, 0lu}, 0.f})
#define case_16 bin_conv_base_params({{1lu, 21lu, 16lu, 32lu}, {3lu, 3lu}, {1lu, 1lu}, {1lu, 1lu}, {1lu, 1lu}, {1lu, 1lu}, 33lu, 1lu, {0lu, 0lu}, 1.f})

TEST_P(BinaryConvolutionOnlyTest, TestsBinaryConvolution) {
}

std::string getTestCaseName(testing::TestParamInfo<bin_conv_test_params> obj) {
    auto in_dims_size = obj.param.in_dims.size();
    return obj.param.device_name +
           "_w" + std::to_string(obj.param.in_dims[in_dims_size - 1]) +
           "_h" + std::to_string(obj.param.in_dims[in_dims_size - 2]) +
           (obj.param.in_dims.size() > 4 ? "_d" + std::to_string(obj.param.in_dims[in_dims_size - 3]) : "") +
           "_c" + std::to_string(obj.param.in_dims[1]) +
           "_kw" + std::to_string(obj.param.kernel[X_AXIS]) +
           "_kh" + std::to_string(obj.param.kernel[Y_AXIS]) +
           (obj.param.kernel.size() > Z_AXIS ? "_kd" + std::to_string(obj.param.kernel[Z_AXIS]) : "") +
           "_sw" + std::to_string(obj.param.strides[X_AXIS]) +
           "_sh" + std::to_string(obj.param.strides[Y_AXIS]) +
           (obj.param.strides.size() > Z_AXIS ? "_sd" + std::to_string(obj.param.strides[Z_AXIS]) : "") +
           "_dilw" + std::to_string(obj.param.dilations[X_AXIS]) +
           "_dilh" + std::to_string(obj.param.dilations[Y_AXIS]) +
           (obj.param.dilations.size() > Z_AXIS ? "_dild" + std::to_string(obj.param.dilations[Z_AXIS]) : "") +
           "_grpc" + std::to_string(obj.param.grp_c) +
           "_pad_v" + std::to_string(obj.param.pad_value);
}
