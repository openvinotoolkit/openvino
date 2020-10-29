// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "ir_gen_helper.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace single_layer_tests;

struct crop_base_params {
    std::vector<size_t> in_dims;
    std::vector<size_t> out_dims;
    std::vector<size_t> offsets;
};

#ifdef IN
#undef IN
#endif

struct crop_test_params : crop_base_params {
    std::string device_name;

    crop_test_params(std::string name, crop_base_params params) :
            crop_base_params(params), device_name(name) {}
};

template <typename data_t>
void ref_crop(InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, crop_test_params prm) {
    data_t *dst_ptr = dst.data();

    int ndims = prm.in_dims.size();

    size_t OFFSET_N = prm.offsets.at(0);
    size_t OFFSET_C = prm.offsets.at(1);
    size_t OFFSET_D = ndims == 5 ? prm.offsets.at(ndims - 3) : 0;
    size_t OFFSET_H = prm.offsets.at(ndims - 2);
    size_t OFFSET_W = prm.offsets.at(ndims - 1);

    size_t ON = prm.out_dims[0];
    size_t OC = prm.out_dims[1];
    size_t OD = ndims == 5 ? prm.out_dims[ndims - 3] : 1;
    size_t OH = prm.out_dims[ndims - 2];
    size_t OW = prm.out_dims[ndims - 1];

    size_t IN = prm.in_dims[0];
    size_t IC = prm.in_dims[1];
    size_t ID = ndims == 5 ? prm.in_dims[ndims - 3] : 1;
    size_t IH = prm.in_dims[ndims - 2];
    size_t IW = prm.in_dims[ndims - 1];

    auto dst_off = [=](size_t n, size_t c, size_t d, size_t h, size_t w) -> size_t {
        return (n * OC * OD * OH * OW + c * OD * OH * OW + d * OH * OW + h * OW + w);
    };
    auto src_off = [=](size_t n, size_t c, size_t d, size_t h, size_t w) -> size_t {
        return (n * IC * ID * IH * IW + c * ID * IH * IW + d * IH * IW + h * IW + w);
    };

    ASSERT_GE(IN - OFFSET_N, ON);
    ASSERT_GE(IC - OFFSET_C, OC);
    ASSERT_GE(ID - OFFSET_D, OD);
    ASSERT_GE(IH - OFFSET_H, OH);
    ASSERT_GE(IW - OFFSET_W, OW);

    data_t* src_ptr = src.data();
    for (size_t n = 0; n < ON; ++n) {
        for (size_t c = 0; c < OC; ++c) {
            for (size_t d = 0; d < OD; ++d) {
                for (size_t h = 0; h < OH; ++h) {
                    for (size_t w = 0; w < OW; ++w) {
                        dst_ptr[dst_off(n, c, d, h, w)] = src_ptr[src_off(n + OFFSET_N, c + OFFSET_C, d + OFFSET_D,
                                                                          h + OFFSET_H, w + OFFSET_W)];
                    }
                }
            }
        }
    }
}

class smoke_CropOnlyTest: public TestsCommon,
                           public WithParamInterface<crop_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="crop" id="1" type="Crop" precision="FP32">
            <crop-data>
                <crop axis="0" offset="_OF0_" dim="_OD0_" />
                <crop axis="1" offset="_OF1_" dim="_OD1_" />
                <crop axis="2" offset="_OF2_" dim="_OD2_" />
                <crop axis="3" offset="_OF3_" dim="_OD3_" />
                <crop axis="4" offset="_OF4_" dim="_OD4_" />
            </crop-data>
            <input>
                <port id="0">
                    <dim>_ID0_</dim>
                    <dim>_ID1_</dim>
                    <dim>_ID2_</dim>
                    <dim>_ID3_</dim>
                    <dim>_ID4_</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_OD0_</dim>
                    <dim>_OD1_</dim>
                    <dim>_OD2_</dim>
                    <dim>_OD3_</dim>
                    <dim>_OD4_</dim>
                </port>
            </output>
        </layer>
)V0G0N";
    
    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
)V0G0N";

    std::string getModel(crop_test_params p) {
        std::string model = layers_t;

        auto dims_size = p.in_dims.size();

        if (dims_size == 4) {
            REMOVE_LINE(model, "<crop axis=\"4\" offset=\"_OF4_\" dim=\"_OD4_\" />");
            REMOVE_LINE(model, "<dim>_ID4_</dim>");
            REMOVE_LINE(model, "<dim>_OD4_</dim>");
        }

        REPLACE_WITH_NUM(model, "_ID0_", p.in_dims[0]);
        REPLACE_WITH_NUM(model, "_ID1_", p.in_dims[1]);
        REPLACE_WITH_NUM(model, "_ID2_", p.in_dims[2]);
        REPLACE_WITH_NUM(model, "_ID3_", p.in_dims[3]);
        if (dims_size == 5)
            REPLACE_WITH_NUM(model, "_ID4_", p.in_dims[4]);

        REPLACE_WITH_NUM(model, "_OD0_", p.out_dims[0]);
        REPLACE_WITH_NUM(model, "_OD1_", p.out_dims[1]);
        REPLACE_WITH_NUM(model, "_OD2_", p.out_dims[2]);
        REPLACE_WITH_NUM(model, "_OD3_", p.out_dims[3]);
        if (dims_size == 5)
            REPLACE_WITH_NUM(model, "_OD4_", p.out_dims[4]);

        REPLACE_WITH_NUM(model, "_OF0_", p.offsets[0]);
        REPLACE_WITH_NUM(model, "_OF1_", p.offsets[1]);
        REPLACE_WITH_NUM(model, "_OF2_", p.offsets[2]);
        REPLACE_WITH_NUM(model, "_OF3_", p.offsets[3]);
        if (dims_size == 5)
            REPLACE_WITH_NUM(model, "_OF4_", p.offsets[4]);

        model = IRTemplateGenerator::getIRTemplate("Crop_Only", p.in_dims, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {
        try {
            crop_test_params p = ::testing::WithParamInterface<crop_test_params>::GetParam();
            std::string model = getModel(p);
            
            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());

	        InferenceEngine::Layout layout = InferenceEngine::ANY;
	        switch (p.in_dims.size()) {
	            case 4: layout = InferenceEngine::NCHW; break;
	            case 5: layout = InferenceEngine::NCDHW; break;
	        }

            InputsDataMap inputs = network.getInputsInfo();
            DataPtr inPtr1 = inputs["in1"]->getInputData();

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(inPtr1->getTensorDesc());
            src->allocate();
            fill_data(src->buffer(), src->size());

            TBlob<float>* srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("in1", src));

            OutputsDataMap out = network.getOutputsInfo();
            BlobMap dstBlobs;
            std::pair<std::string, DataPtr> item = *out.begin();
            TBlob<float>::Ptr dst;
            dst = make_shared_blob<float>(item.second->getTensorDesc());
            dst->allocate();
            dstBlobs[item.first] = dst;

            TBlob<float>::Ptr dst_ref;
            dst_ref = make_shared_blob<float>(item.second->getTensorDesc());
            dst_ref->allocate();

            ref_crop(*srcPtr, *dst_ref, p);

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.device_name);
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(srcs);
            inferRequest.SetOutput(dstBlobs);
            inferRequest.Infer();

            compare(*dstBlobs.begin()->second, *dst_ref);

        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_1 crop_base_params({{1, 5, 32, 32}, {1, 2, 23, 23}, {0, 2, 5, 4}})
#define case_2 crop_base_params({{1, 5, 32, 32}, {1, 5, 5, 5}, {0, 0, 20, 20}})
#define case_3 crop_base_params({{1, 5, 32, 32}, {1, 5, 32, 10}, {0, 0, 0, 20}})
#define case_4 crop_base_params({{1, 5, 32, 20}, {1, 5, 30, 10}, {0, 0, 2, 10}})
#define case_5 crop_base_params({{1, 5, 32, 20, 14}, {1, 5, 30, 10, 8}, {0, 0, 2, 10, 6}})
#define case_6 crop_base_params({{5, 9, 32, 20, 14}, {2, 5, 30, 10, 8}, {3, 4, 2, 10, 6}})

TEST_P(smoke_CropOnlyTest, TestsCrop) {}

std::string  getTestCaseName(testing::TestParamInfo<crop_test_params> obj) {
    int ndims = obj.param.in_dims.size();

    return  obj.param.device_name +
        "_in" + std::to_string(obj.param.in_dims[0]) +
        "_ic" + std::to_string(obj.param.in_dims[1]) +
        "_id" + std::to_string(ndims == 5 ? obj.param.in_dims[ndims - 3] : 1) +
        "_ih" + std::to_string(obj.param.in_dims[ndims - 2]) +
        "_iw" + std::to_string(obj.param.in_dims[ndims - 1]) +
        "_on" + std::to_string(obj.param.out_dims[0]) +
        "_oc" + std::to_string(obj.param.out_dims[1]) +
        "_od" + std::to_string(ndims == 5 ? obj.param.out_dims[ndims - 3] : 1) +
        "_oh" + std::to_string(obj.param.out_dims[ndims - 2]) +
        "_ow" + std::to_string(obj.param.out_dims[ndims - 1]);
}

crop_test_params crop_only_test_cases[] = {
		crop_test_params("CPU", case_1),
		crop_test_params("CPU", case_2),
		crop_test_params("CPU", case_3),
		crop_test_params("CPU", case_4),
		crop_test_params("CPU", case_5),
		crop_test_params("CPU", case_6),
};

INSTANTIATE_TEST_CASE_P(
        TestsPooling, smoke_CropOnlyTest, ::testing::ValuesIn(crop_only_test_cases), getTestCaseName);
