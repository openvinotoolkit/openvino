// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "../common_single_layer_tests/conv_ref.hpp"
#include <string>
#include <algorithm>

#include "common_test_utils/common_layers_params.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using std::vector;

struct conv_base_params {
    vector<size_t> in_dims;
    vector<size_t> kernel;
    vector<size_t> strides;
    vector<size_t> pads_begin;
    vector<size_t> pads_end;
    vector<size_t> dilations;

    size_t out_c;
    size_t grp_c;

    vector<size_t> out_dims;
};

struct conv_test_params : conv_base_params {
    std::string device_name;

    conv_test_params(std::string name, conv_base_params params) :
            conv_base_params(params), device_name(name) {}
};

template <typename data_t>
static void fill_int_data_even(data_t *data, size_t size, bool is_signed) {
    for (size_t i = 0 ; i < size; i++) {
        data[i] = (i * 13 % 21 - 10 * is_signed) * 2;
    }
}

template <typename data_t>
static void fill_int_data(data_t *data, size_t size, bool is_signed) {
    for (size_t i = 0 ; i < size; i++) {
        data[i] = i * 13 % 21 - 10 * is_signed;
    }
}

template <typename src_data_t>
class smoke_ConvolutionInt8OnlyTest : public TestsCommon,
                                  public WithParamInterface<conv_test_params> {

    std::string model_t = (std::string)R"V0G0N(
<net name="Convolution_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="_IP_" id="0">
            <output>
                <port id="0">
                    _INPUT_DIMS_
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="Convolution" precision="I8">
            <convolution strides="_KS_"
                         pads_begin="_PB_"  pads_end="_PE_"
                         kernel="_K_"
                         dilations="_DL_"
                         output="_OC_"  group="_GC_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">
                    _INPUT_DIMS_
                </port>
            </input>
            <output>
                <port id="2">
                    _OUTPUT_DIMS_
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

    size_t calculateOutDim(size_t in_dim, size_t kernel, size_t stride, size_t pad_begin) {
        return (in_dim + 2lu * pad_begin - kernel) / stride + 1lu;
    }

    void createBlobs(const conv_test_params &p, typename TBlob<src_data_t>::Ptr &src, TBlob<float>::Ptr &dst, TBlob<float>::Ptr &dst_ref) {
        auto in_size = p.in_dims.size();
        auto out_size = p.out_dims.size();
        SizeVector dims_dst = {
                p.out_dims[out_size - 1] == 0 ?
                calculateOutDim(p.in_dims[in_size - 1], p.kernel[X_AXIS], p.strides[X_AXIS], p.pads_begin[X_AXIS]) : p.out_dims[out_size - 1],
                p.out_dims[out_size - 2] == 0 ?
                calculateOutDim(p.in_dims[in_size - 2], p.kernel[Y_AXIS], p.strides[Y_AXIS], p.pads_begin[Y_AXIS]) : p.out_dims[out_size - 2],
                p.out_c,
                1lu};
        SizeVector dims_src;
        for (int i = in_size; i > 0; i--) {
            dims_src.push_back(p.in_dims[i - 1]);
        }

        Layout layout = NCHW;
        if (in_size == 5) {
            layout = NCDHW;
            dims_dst.insert(dims_dst.begin() + 2, p.out_dims.size() > 2 ?
                                                  (p.out_dims[out_size - 3] == 0 ?
                                                   calculateOutDim(p.in_dims[in_size - 3], p.kernel[Z_AXIS], p.strides[Z_AXIS], p.pads_begin[Z_AXIS]) : p.out_dims[out_size - 3]) : 1lu);
        }

        std::reverse(dims_src.begin(), dims_src.end());
        std::reverse(dims_dst.begin(), dims_dst.end());

        Precision src_precision = (typeid(src_data_t) == typeid(int8_t)) ? Precision::I8 : Precision::U8;
        src = make_shared_blob<src_data_t>(TensorDesc({src_precision, dims_src, layout}));
        src->allocate();

        dst = make_shared_blob<float>(TensorDesc({Precision::FP32, dims_dst, layout}));
        dst->allocate();

        dst_ref = make_shared_blob<float>(TensorDesc({Precision::FP32, dims_dst, layout}));
        dst_ref->allocate();
    }

    TBlob<uint8_t>::Ptr fillWeights(const conv_test_params &p) {
        auto KZ = p.kernel.size() > Z_AXIS ? p.kernel[Z_AXIS] : 1lu;
        TBlob<uint8_t> *weights_ptr = new TBlob<uint8_t>(TensorDesc({Precision::U8,
                                                         {p.kernel[X_AXIS] * p.kernel[Y_AXIS] * KZ * p.out_c * p.in_dims[1] / p.grp_c * sizeof(uint8_t)
                                                         + p.out_c * sizeof(int32_t)}, C}));
        weights_ptr->allocate();
        size_t bias_size = p.out_c;
        size_t weights_size = (weights_ptr->size() - bias_size * sizeof(int32_t)) / sizeof(uint8_t);
        int8_t *weights_data = (int8_t *) weights_ptr->buffer();
        auto *bias_data = (int32_t *)(weights_data + weights_size);

        if (typeid(src_data_t) == typeid(int8_t)) {
            // If input data is signed, weight data is divided by 2 due to the specifics of implementation in mkl-dnn
            fill_int_data_even(weights_data, weights_size, true);
        } else {
            fill_int_data(weights_data, weights_size, true);
        }
        fill_int_data(bias_data, bias_size, true);

        return TBlob<uint8_t>::Ptr(weights_ptr);
    }

    void calculateRef(const TBlob<uint8_t>::Ptr &weights, const conv_test_params &p, const typename TBlob<src_data_t>::Ptr &src,
                      TBlob<float>::Ptr &dst_ref) {
        const int8_t *weights_data = (const int8_t *) weights->buffer();
        size_t bias_size = p.out_c;
        size_t weights_size = (weights->size() - bias_size * sizeof(int32_t)) / sizeof(uint8_t);
        auto *bias_data = (const int32_t *)(weights_data + weights_size);
        CommonTestUtils::conv_common_params params;
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
        ref_conv_common<>({ src }, *dst_ref.get(), weights_data, weights_size, bias_data, bias_size, params);
    }

    void SetUp() override {
        try {
            conv_test_params p = ::testing::WithParamInterface<conv_test_params>::GetParam();
            std::string model = getModel(p);

            typename TBlob<src_data_t>::Ptr src;
            TBlob<float>::Ptr dst, dst_ref;
            createBlobs(p, src, dst, dst_ref);
            auto *src_data = src->cbuffer().template as<src_data_t*>();
            size_t src_size = src->size() / sizeof(src_data_t);
            if (typeid(src_data_t) == typeid(int8_t)) {
                fill_int_data(src_data, src_size, true);
            } else {
                fill_int_data(src_data, src_size, false);
            }

            auto weights = fillWeights(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, weights);

            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("in1", src));

            OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            outputBlobs[item.first] = dst;

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.device_name);
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(srcs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            calculateRef(weights, p, src, dst_ref);
            compare(*dst, *dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }

    virtual std::string getModel(conv_test_params p) {
        std::string model = model_t;

        auto in_dims_size = p.in_dims.size();
        std::string input_dims = "<dim>" + std::to_string(p.in_dims[0]) + "</dim>";
        for (int i = 1; i < in_dims_size; i++) {
            input_dims += "\n                    <dim>" + std::to_string(p.in_dims[i]) + "</dim>";
        }
        REPLACE_WITH_STR(model, "_INPUT_DIMS_", input_dims);

        auto out_dims_size = p.out_dims.size();
        std::string output_dims = "<dim>" + std::to_string(p.in_dims[0]) + "</dim>";
        output_dims += "\n                    <dim>" + std::to_string(p.out_c) + "</dim>";
        if (out_dims_size > 2) {
            size_t od = (p.out_dims[out_dims_size - 3] == 0 ?
                         calculateOutDim(p.in_dims[in_dims_size - 3], p.kernel[Z_AXIS], p.strides[Z_AXIS], p.pads_begin[Z_AXIS]) : p.out_dims[out_dims_size - 3]);
            output_dims += "\n                    <dim>" + std::to_string(od) + "</dim>";
        }
        size_t oh = p.out_dims[out_dims_size - 2] == 0 ?
                    calculateOutDim(p.in_dims[in_dims_size - 2], p.kernel[Y_AXIS], p.strides[Y_AXIS], p.pads_begin[Y_AXIS]) : p.out_dims[out_dims_size - 2];
        output_dims += "\n                    <dim>" + std::to_string(oh) + "</dim>";
        size_t ow = p.out_dims[out_dims_size - 1] == 0 ?
                    calculateOutDim(p.in_dims[in_dims_size - 1], p.kernel[X_AXIS], p.strides[X_AXIS], p.pads_begin[X_AXIS]) : p.out_dims[out_dims_size - 1];
        output_dims += "\n                    <dim>" + std::to_string(ow) + "</dim>";
        REPLACE_WITH_STR(model, "_OUTPUT_DIMS_", output_dims);

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.strides);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_DL_", p.dilations);

        REPLACE_WITH_NUM(model, "_GC_", p.grp_c);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);

        std::string ip = (typeid(src_data_t) == typeid(int8_t)) ? "I8" : "U8";
        REPLACE_WITH_STR(model, "_IP_", ip);

        size_t KD = p.kernel.size() > Z_AXIS ? p.kernel[Z_AXIS] : 1lu;
        size_t w_data_size = (p.kernel[X_AXIS] * p.kernel[Y_AXIS] * KD * p.out_c * p.in_dims[1] / p.grp_c) * sizeof(uint8_t);
        size_t b_data_size = p.out_c;
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);

        return model;
    }
};

// conv_base_params ({in_dims, kernel, strides, pads_begin, pads_end, dilations, out_c, grp_c, out_dims})
// If out_dims are zero, they are calculated automatically.
// 2D
#define case_1  conv_base_params({{1, 9, 16, 32},  {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 17, 1, {0, 0}})
#define case_2  conv_base_params({{1, 9, 32, 16},  {2, 4}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 17, 1, {0, 0}})
#define case_3  conv_base_params({{1, 9, 32, 16},  {2, 4}, {2, 1}, {0, 0}, {0, 0}, {1, 1}, 17, 1, {0, 0}})
#define case_4  conv_base_params({{1, 3, 40, 40},  {3, 3}, {1, 2}, {0, 0}, {0, 0}, {1, 1}, 20, 1, {0, 0}})
#define case_5  conv_base_params({{1, 9, 16, 32},  {7, 7}, {2, 2}, {3, 3}, {0, 0}, {1, 1}, 17, 1, {0, 0}})
#define case_6  conv_base_params({{1, 3, 224, 224}, {7, 7}, {2, 2}, {2, 2}, {0, 0}, {1, 1}, 64, 1, {111, 111}})
#define case_7  conv_base_params({{1, 16, 40, 40}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 16, 16, {0, 0}})
#define case_8  conv_base_params({{1, 32, 16, 32}, {7, 7}, {2, 2}, {3, 3}, {0, 0}, {1, 1}, 32, 32, {0, 0}})
#define case_9  conv_base_params({{1, 16, 40, 40}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {9, 9}, 16, 16, {0, 0}})
#define case_10 conv_base_params({{1, 32, 16, 32}, {7, 7}, {2, 2}, {3, 3}, {0, 0}, {3, 3}, 32, 32, {2, 10}})
#define case_11 conv_base_params({{1, 4, 16, 32},  {7, 7}, {2, 2}, {3, 3}, {0, 0}, {2, 2}, 4, 4, {5, 13}})
#define case_12 conv_base_params({{1, 3, 224, 224}, {10, 10}, {1, 1}, {4, 4}, {0, 0}, {1, 1}, 4, 1, {223, 223}})
#define case_13 conv_base_params({{1, 32, 1, 15000}, {11, 1}, {1, 1}, {20, 0}, {0, 0}, {4, 1}, 32, 1, {1, 15000}})
#define case_14 conv_base_params({{1, 16, 40, 40}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 16, 8, {0, 0}})
#define case_15 conv_base_params({{1, 16, 40, 40}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 8, 2, {0, 0}})
#define case_16 conv_base_params({{1, 3, 40, 40}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 9, 3, {0, 0}})
// 3D
#define case_3d_0 conv_base_params({{1, 3, 16, 32, 32},  {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, 17, 1, {0, 0, 0}})
#define case_3d_1 conv_base_params({{1, 3, 16, 32, 32},  {3, 3, 3}, {2, 2, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, 64, 1, {0, 0, 0}})
#define case_3d_2 conv_base_params({{1, 32, 8, 8, 8},  {3, 3, 3}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, 32, 32, {0, 0, 0}})
#define case_3d_3 conv_base_params({{1, 32, 10, 10, 10},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 32, 32, {0, 0, 0}})
#define case_3d_4 conv_base_params({{1, 32, 8, 8, 8},  {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, 32, 32, {0, 0, 0}})
#define case_3d_5 conv_base_params({{1, 32, 8, 8, 8},  {3, 3, 3}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, 16, 16, {0, 0, 0}})
#define case_3d_6 conv_base_params({{1, 32, 10, 10, 10},  {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 16, 8, {0, 0, 0}})
#define case_3d_7 conv_base_params({{1, 4, 8, 8, 8},  {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, 16, 4, {0, 0, 0}})

using smoke_conv_u8s32 = smoke_ConvolutionInt8OnlyTest<uint8_t>;

TEST_P(smoke_conv_u8s32, TestsConvolution) {
}

std::string getTestCaseInt8Name(testing::TestParamInfo<conv_test_params> obj) {
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
           "_grpc" + std::to_string(obj.param.grp_c);
}

conv_test_params conv_only_int8_test_cases[] = {
        conv_test_params("CPU", case_1),
        conv_test_params("CPU", case_2),
        conv_test_params("CPU", case_3),
        conv_test_params("CPU", case_4),
        conv_test_params("CPU", case_5),
        conv_test_params("CPU", case_6),
//// todo: it does not work on AVX-512
//        conv_test_params("CPU", case_7),
//        conv_test_params("CPU", case_8),
//        conv_test_params("CPU", case_9),
//        conv_test_params("CPU", case_10),
//        conv_test_params("CPU", case_11),
        conv_test_params("CPU", case_12),
        conv_test_params("CPU", case_13),
        conv_test_params("CPU", case_14),
        conv_test_params("CPU", case_15),
        conv_test_params("CPU", case_16),
};

conv_test_params conv_only_int8_3d_test_cases[] = {
        conv_test_params("CPU", case_3d_0),
        conv_test_params("CPU", case_3d_1),
        conv_test_params("CPU", case_3d_2),
        conv_test_params("CPU", case_3d_3),
        conv_test_params("CPU", case_3d_4),
        conv_test_params("CPU", case_3d_5),
        conv_test_params("CPU", case_3d_6),
        conv_test_params("CPU", case_3d_7),
};

INSTANTIATE_TEST_CASE_P(
        TestConvolution, smoke_conv_u8s32, ::testing::ValuesIn(conv_only_int8_test_cases), getTestCaseInt8Name);

INSTANTIATE_TEST_CASE_P(
        TestConvolution_3d, smoke_conv_u8s32, ::testing::ValuesIn(conv_only_int8_3d_test_cases), getTestCaseInt8Name);
