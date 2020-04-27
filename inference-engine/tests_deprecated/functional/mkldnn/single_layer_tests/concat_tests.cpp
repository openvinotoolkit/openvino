// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

using namespace ::testing;
using namespace InferenceEngine;


struct concat_base_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in1;

    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in2;

    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } out;

    size_t axis;
};

struct concat_test_params : concat_base_params {
    std::string device_name;

    concat_test_params(std::string name, concat_base_params params)
            : concat_base_params(params), device_name(name) {}
};

template <typename data_t>
void check_concat_fwd(const TBlob<data_t> &src, concat_test_params prm)
{
}

class smoke_CPU_ConcatOnlyTest: public TestsCommon,
                    public WithParamInterface<concat_test_params> {

    std::string model_t = R"V0G0N(
<net name="ConcatOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>_IN1_</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    <dim>_IN2_</dim>
                    <dim>_IC2_</dim>
                    <dim>_IH2_</dim>
                    <dim>_IW2_</dim>
                </port>
            </output>
        </layer>
        <layer name="con" id="3" type="Concat" precision="FP32">
            <concat_data axis="_AXIS_"/>
            <input>
                <port id="1">
                    <dim>_IN1_</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
                <port id="2">
                    <dim>_IN2_</dim>
                    <dim>_IC2_</dim>
                    <dim>_IH2_</dim>
                    <dim>_IW2_</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>_ON_</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(concat_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IN1_", p.in1.n);
        REPLACE_WITH_NUM(model, "_IC1_", p.in1.c);
        REPLACE_WITH_NUM(model, "_IW1_", p.in1.w);
        REPLACE_WITH_NUM(model, "_IH1_", p.in1.h);

        REPLACE_WITH_NUM(model, "_IN2_", p.in2.n);
        REPLACE_WITH_NUM(model, "_IC2_", p.in2.c);
        REPLACE_WITH_NUM(model, "_IW2_", p.in2.w);
        REPLACE_WITH_NUM(model, "_IH2_", p.in2.h);

        REPLACE_WITH_NUM(model, "_ON_", p.out.n);
        REPLACE_WITH_NUM(model, "_OC_", p.out.c);
        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        return model;
    }

protected:

    static void fill_data_ints(float *data, size_t size, int start) {
        for (size_t i = 0; i < size; i++) {
            data[i] = (float) (start + i);
        }
    }

    virtual void SetUp() {

        try {
            concat_test_params p = ::testing::WithParamInterface<concat_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());

            SizeVector dims_src1 = {p.in1.n,
                                    p.in1.c,
                                    p.in1.h,
                                    p.in1.w
                                    };

            SizeVector dims_src2 = {p.in2.n,
                                    p.in2.c,
                                    p.in2.h,
                                    p.in2.w};

            SizeVector dims_dst = {p.out.n,
                                   p.out.c,
                                   p.out.h,
                                   p.out.w};

            Blob::Ptr src1 = make_shared_blob<float>({Precision::FP32, dims_src1, Layout::NCHW});
            src1->allocate();
            fill_data_ints(src1->buffer(), src1->size(), 0);
            Blob::Ptr src2 =  make_shared_blob<float>({Precision::FP32, dims_src2, Layout::NCHW});
            src2->allocate();
            fill_data_ints(src2->buffer(), src2->size(), 10000);
            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, Blob::Ptr>("in2", src2));

            OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.device_name);
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(srcs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            //compare(src, dst);

            float *src1_ptr = src1->buffer();
            float *src2_ptr = src2->buffer();
            float *dst_ptr = output->buffer();

            int len1 = 1, len2 = 1, cycles;
            for (int dim = p.axis; dim < output->getTensorDesc().getDims().size(); dim++) {
                len1 *= src1->getTensorDesc().getDims()[dim];
                len2 *= src2->getTensorDesc().getDims()[dim];
            }
            cycles = p.axis;


            int index1 = 0, index2 = 0, index = 0;
            for (int cycle = 0; cycle < cycles; cycle ++) {
                for (int i1 = 0; i1 < len1; i1++) {
                    if (src1_ptr[index1] != dst_ptr[index])
                    {
                        FAIL() << "index: " << index << " src: " << src1_ptr[index1] << ", dst: " << dst_ptr[index];
                    }
                    index1++; index++;
                }
                for (int i2 = 0; i2 < len2; i2++) {
                    if (src2_ptr[index2] != dst_ptr[index])
                    {
                        FAIL() << "index: " << index << " src: " << src2_ptr[index2] << ", dst: " << dst_ptr[index];
                    }
                    index2++; index++;
                }
            }


        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_1 concat_base_params({\
	{1, 7, 2, 5},\
	{1, 7, 2, 5},\
	{2, 7, 2, 5},\
	0})
#define case_2 concat_base_params({\
	{1, 7, 2, 5},\
	{1, 7, 2, 5},\
	{1, 7, 4, 5},\
	2})
#define case_3 concat_base_params({\
	{1, 7, 2, 5},\
	{1, 13, 2, 5},\
	{1, 20, 2, 5},\
	1})
#define case_4 concat_base_params({\
	{1, 7, 2, 13},\
	{1, 7, 2, 17},\
	{1, 7, 2, 30},\
	3})
#define case_5 concat_base_params({\
	{1, 8, 8, 16},\
	{1, 16, 8, 16},\
	{1, 24, 8, 16},\
	1})

TEST_P(smoke_CPU_ConcatOnlyTest, TestsConcat) {
}

std::string  getTestCaseName(testing::TestParamInfo<concat_test_params> obj) {
    return  obj.param.device_name +
        "_out_w" + std::to_string(obj.param.out.w) +
        "_out_h" + std::to_string(obj.param.out.h) +
        "_out_c" + std::to_string(obj.param.out.c) +
        "_out_n" + std::to_string(obj.param.out.n);
}

concat_test_params concat_only_test_cases[] = {
        concat_test_params("CPU", case_1),
        concat_test_params("CPU", case_2),
        concat_test_params("CPU", case_3),
        concat_test_params("CPU", case_4),
        concat_test_params("CPU", case_5),
};

INSTANTIATE_TEST_CASE_P(TestConcat, smoke_CPU_ConcatOnlyTest, ::testing::ValuesIn(concat_only_test_cases), getTestCaseName);
