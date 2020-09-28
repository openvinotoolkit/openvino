// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

#include <string>

#include <format_reader/format_reader_ptr.h>
#include "common_test_utils/data_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

struct conv_base_params {
    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    size_t krn_w;
    size_t krn_h;
    size_t str_w;
    size_t str_h;
    size_t pad_w;
    size_t pad_h;
    size_t dil_w;
    size_t dil_h;

    size_t out_c;
    size_t grp_c;

    struct {
        size_t w;
        size_t h;
    } out;
};

struct conv_test_int8_params : conv_base_params {
    std::string device_name;

    conv_test_int8_params(std::string name, conv_base_params params) :
            conv_base_params(params), device_name(name) {}
};

template <typename data_t>
void ref_conv_relu(const TBlob<data_t> &src, const data_t *weights, const size_t weightsSize,
                   TBlob<data_t> &dst, conv_test_int8_params prm) {
    size_t KW = prm.krn_w;
    size_t KH = prm.krn_h;
    size_t GC = prm.grp_c;

    size_t IW = src.getTensorDesc().getDims()[3];
    size_t IH = src.getTensorDesc().getDims()[2];
    size_t IC = src.getTensorDesc().getDims()[1];

    size_t OW = prm.out.w == 0 ? (IW + 2 * prm.pad_w - prm.krn_w) / prm.str_w + 1 : prm.out.w;
    size_t OH = prm.out.h == 0 ? (IH + 2 * prm.pad_h - prm.krn_h) / prm.str_h + 1 : prm.out.h;
    size_t OC = prm.out_c;

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    const data_t *bias_data = weights_data + KW * KH * OC * IC / GC;
    data_t *dst_data = dst.data();

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
                                int32_t iw = ow * prm.str_w - prm.pad_w + kw * (1 + prm.dil_w);
                                int32_t ih = oh * prm.str_h - prm.pad_h + kh * (1 + prm.dil_h);
                                if (iw < 0 || iw >= (int32_t)IW || ih < 0
                                    || ih >= (int32_t)IH)
                                    continue;
                                size_t iidx = g * IC / GC * IH * IW
                                              + ic * IH * IW + ih * IW + iw;
                                size_t widx = g * OC / GC * IC / GC * KH * KW
                                              + oc * IC / GC * KH * KW
                                              + ic * KH * KW + kh * KW + kw;

                                dst_data[ oidx] += src_data[iidx] * weights_data[widx];
                            }
                        }
                    }

                    // Applying ReLU
                    if (dst_data[oidx] < 0) dst_data[oidx] = 0;

                }
            }
        }
    }
}

class smoke_ConvolutionInt8Test: public TestsCommon,
                                 public WithParamInterface<conv_test_int8_params> {

    std::string model_t = R"V0G0N(
<Net Name="Convolution_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="Convolution" precision="FP32">
            <convolution stride-x="_SW_" stride-y="_SH_"
                         pad-x="_PW_"    pad-y="_PH_"
                         kernel-x="_KW_" kernel-y="_KH_"
                         output="_OC_"   group="_GC_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="conv1_relu" type="ReLU" precision="FP32">
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1" />
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3" />
    </edges>
</Net>
)V0G0N";

    std::string getModel(conv_test_int8_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);

        REPLACE_WITH_NUM(model, "_KW_", p.krn_w);
        REPLACE_WITH_NUM(model, "_KH_", p.krn_h);
        REPLACE_WITH_NUM(model, "_SW_", p.str_w);
        REPLACE_WITH_NUM(model, "_SH_", p.str_h);
        REPLACE_WITH_NUM(model, "_PW_", p.pad_w);
        REPLACE_WITH_NUM(model, "_PH_", p.pad_h);

        REPLACE_WITH_NUM(model, "_GC_", p.grp_c);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);
        REPLACE_WITH_NUM(model, "_OH_", p.out.h == 0 ? (p.in.h + 2 * p.pad_h - p.krn_h) / p.str_h + 1 : p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w == 0 ? (p.in.w + 2 * p.pad_w - p.krn_w) / p.str_w + 1 : p.out.w);

        size_t w_data_size = (p.krn_w * p.krn_h * p.out_c * p.in.c / p.grp_c )* sizeof(float);
        size_t b_data_size = p.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);
        return model;
    }

protected:
    const char* DEFAULT_PATH_P = "./lib";

    static void compare_NRMSD(InferenceEngine::Blob &res, InferenceEngine::Blob &ref, float max_nrmsd = 0.01f) {
        float *res_ptr = res.buffer().as<float*>();
        size_t res_size = res.size();

        float *ref_ptr = ref.buffer().as<float*>();
        size_t ref_size = ref.size();

        ASSERT_EQ(res_size, ref_size);

        float sum = 0;

        float mmin = ref_ptr[0], mmax = ref_ptr[0];

        for (size_t i = 0; i < ref_size; i++) {
            float sqr = (ref_ptr[i] - res_ptr[i]);
            sqr *= sqr;
            sum += sqr;

            mmin = (std::min)(mmin, ref_ptr[i]);
            mmax = (std::max)(mmax, ref_ptr[i]);

            if (i % 10007 == 0) {
                std::cout << i << ": " << res_ptr[i] << "\t" << ref_ptr[i] << "\t" << "\tdiv: " << ref_ptr[i] / res_ptr[i] << std::endl;
            }

        }
        sum /= ref_size;

        sum = pow(sum, 0.5f);

        sum /= mmax - mmin;

        ASSERT_LE(sum, max_nrmsd);
    }

    virtual void SetUp() {
        try {
            conv_test_int8_params p = ::testing::WithParamInterface<conv_test_int8_params>::GetParam();
            std::string model = getModel(p);

            TBlob<uint8_t> *weights = new TBlob<uint8_t>(TensorDesc(Precision::U8, {(p.krn_w * p.krn_h * p.out_c * p.in.c / p.grp_c + p.out_c)
                                                                                    * sizeof(float)}, C));
            weights->allocate();

            //fill_data_sine((float *) weights->buffer(), weights->size() / sizeof(float), 0.00, 0.005, 0.1);
            CommonTestUtils::fill_data_sine((float *) weights->buffer(), weights->size() / sizeof(float), 1, 4, 0.3);
            //fill_data_dbgval((float *) weights->buffer(), weights->size() / sizeof(float));
            //size_t bias_start = p.krn_w * p.krn_h * p.out_c * p.in.c / p.grp_c;
            //fill_data_const((float *) weights->buffer() + bias_start, p.out_c, 0.00);

            // Set biases to 0
            /*for (int i = weights->size() / sizeof(float) - C - 1; i < weights->size() / sizeof(float); i++) {
                ((float *) weights->buffer())[i] = 0;
            }*/


            TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);

            // Collecting statistics

            // TODO Load nodes stats from file
            std::string imageFilename = TestDataHelpers::get_data_path() + "/validation_set/224x224/dog.bmp";
            std::cout << "Using image file: " << imageFilename << std::endl;

            Core ie;
            auto network = ie.ReadNetwork(model, weights_ptr);

            SizeVector dims_dst = {p.out.w == 0 ? (p.in.w + 2 * p.pad_w - p.krn_w) / p.str_w + 1 : p.out.w,
                                   p.out.h == 0 ? (p.in.h + 2 * p.pad_h - p.krn_h) / p.str_h + 1 : p.out.h,
                                   p.out_c,
                                   1};
            Blob::Ptr dst = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst->allocate();

            // Setting the statistics data

            CNNNetwork myNetwork = ie.ReadNetwork(model, weights_ptr);

            SizeVector dims_src = {p.in.w,
                                   p.in.h,
                                   p.in.c,
                                   1};          // 1 is a batch size
            Blob::Ptr src = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), NCHW));
            src->allocate();
            fill_data(src->buffer().as<float *>(), src->size());






            std::vector<std::string> imageNames = { imageFilename };

            /** Taking information about all topology inputs **/
            InputsDataMap inputInfo(myNetwork.getInputsInfo());

            if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
            auto inputInfoItem = *inputInfo.begin();

            /** Specifying the precision of input data provided by the user.
             * This should be called before load of the network to the plugin **/
            inputInfoItem.second->setPrecision(Precision::FP32);
            inputInfoItem.second->setLayout(Layout::NCHW);


            std::vector<std::shared_ptr<unsigned char>> imagesData;
            for (auto & i : imageNames) {
                FormatReader::ReaderPtr reader(i.c_str());
                if (reader.get() == nullptr) {
                    std::cout << "Image " + i + " cannot be read!" << std::endl;
                    continue;
                }
                /** Store image data **/
                SizeVector dims = inputInfoItem.second->getTensorDesc().getDims();
                std::shared_ptr<unsigned char> data(reader->getData(dims.back(), dims.at(dims.size() - 2)));
                if (data.get() != nullptr) {
                    imagesData.push_back(data);
                }
            }
            if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

            OutputsDataMap outputInfo(myNetwork.getOutputsInfo());
            for (auto itOut : outputInfo) {
                itOut.second->setPrecision(Precision::FP32);
            }

            /** Filling input tensor with images. First b channel, then g and r channels **/
            size_t num_chanels = src->getTensorDesc().getDims()[1];
            size_t image_size = src->getTensorDesc().getDims()[2] * src->getTensorDesc().getDims()[3];

            float* data = src->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_chanels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_chanels + ch * image_size + pid ] = (float)(imagesData.at(image_id).get()[pid*num_chanels + ch]);
                    }
                }
            }

            // Inferring the converted network and comparing the result with the reference
            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.device_name);
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            OutputsDataMap outInfo;
            outInfo = network.getOutputsInfo();
            ASSERT_EQ(outInfo.size(), 1);
            ASSERT_NE(outInfo.begin()->second, nullptr);
            inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            inferRequest.SetBlob(outInfo.begin()->first, dst);

            std::cout << "Inferring int8" << std::endl;
            inferRequest.Infer();

            // Calculating FP32 reference
            TBlob<float> dst_ref(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst_ref.allocate();
            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            ref_conv_relu<float>(*srcPtr, (const float *)weights->buffer(), weights->size() / sizeof(float), dst_ref, p);

            // Comparing the result with the reference
            compare_NRMSD(*dst, dst_ref, 0.17);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

/*
    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    size_t krn_w;
    size_t krn_h;
    size_t str_w;
    size_t str_h;
    size_t pad_w;
    size_t pad_h;
    size_t dil_w;
    size_t dil_h;

    size_t out_c;
    size_t grp_c;

    struct {
        size_t w;
        size_t h;
    } out;
*/
// Wo=(Wi−F+2P)/S+1

#define case_1 conv_base_params({{4, 4, 3}, 1, 1, 1, 1, 0, 0, 0, 0, 3, 1})
#define case_2 conv_base_params({{16, 32, 3}, 2, 4, 1, 1, 0, 0, 0, 0, 17, 1})
#define case_3 conv_base_params({{16, 32, 3}, 2, 4, 2, 1, 0, 0, 0, 0, 17, 1})
#define case_4 conv_base_params({{40, 40, 3}, 3, 3, 1, 2, 0, 0, 0, 0, 20, 1})
#define case_5 conv_base_params({{32, 16, 3}, 7, 7, 2, 2, 3, 3, 0, 0, 17, 1})
#define case_6 conv_base_params({{224, 224, 3}, 7, 7, 2, 2, 2, 2, 0, 0, 64, 1, {112, 112}})
/*#define case_7 conv_base_params({{40, 40, 16}, 3, 3, 1, 1, 0, 0, 0, 0, 16, 16})
#define case_8 conv_base_params({{32, 16, 32}, 7, 7, 2, 2, 3, 3, 0, 0, 32, 32})*/

// These tests use dilated convolution and don't work yet
/*#define case_9 conv_base_params({{40, 40, 16}, 3, 3, 1, 1, 0, 0, 8, 8, 16, 16})
#define case_10 conv_base_params({{32, 16, 32}, 7, 7, 2, 2, 3, 3, 8, 8, 32, 32})
#define case_11 conv_base_params({{32, 16, 4}, 7, 7, 2, 2, 3, 3, 8, 8, 4, 4})*/

TEST_P(smoke_ConvolutionInt8Test, TestsConvolution) {
}

std::string  getTestCaseName(testing::TestParamInfo<conv_test_int8_params> obj) {
    return  obj.param.device_name +
        "_w" + std::to_string(obj.param.in.w) +
        "_h" + std::to_string(obj.param.in.h) +
        "_c" + std::to_string(obj.param.in.c) +
        "_krnw" + std::to_string(obj.param.krn_w) +
        "_krnh" + std::to_string(obj.param.krn_h) +
        "_strw" + std::to_string(obj.param.str_w) +
        "_strh" + std::to_string(obj.param.str_h) +
        "_dilw" + std::to_string(obj.param.dil_w) +
        "_dilh" + std::to_string(obj.param.dil_h) +
        "_grpc" + std::to_string(obj.param.grp_c);
}

conv_test_int8_params conv_int8_test_cases[] = {
    conv_test_int8_params("CPU", case_1),
    conv_test_int8_params("CPU", case_2),
    conv_test_int8_params("CPU", case_3),
    conv_test_int8_params("CPU", case_4),
    conv_test_int8_params("CPU", case_5),
    // conv_test_int8_params("CPU", case_6),
    //conv_test_int8_params("CPU", case_7),
    //conv_test_int8_params("CPU", case_8),
    //conv_test_int8_params("CPU", case_9),
    //conv_test_int8_params("CPU", case_10),
    //conv_test_int8_params("CPU", case_11),
};

INSTANTIATE_TEST_CASE_P(
        TestConvolution, smoke_ConvolutionInt8Test, ::testing::ValuesIn(conv_int8_test_cases), getTestCaseName);