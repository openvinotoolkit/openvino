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

struct simplernms_test_params {
    std::string device_name;

    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    float cls_threshold;
    size_t max_num_proposals;
    float iou_threshold;
    size_t min_bbox_size;
    size_t feat_stride;
    size_t pre_nms_topn;
    size_t post_nms_topn;
    float scale1;
    float scale2;
    float scale3;
};

template <typename data_t>
void ref_simplernms(const TBlob<data_t> &src, TBlob<data_t> &dst, simplernms_test_params prm)
{
}

class MKLDNNSimplerNMSOnlyTest: public TestsCommon,
                             public WithParamInterface<simplernms_test_params> {

    std::string layers_t = R"V0G0N(
        <layer name="power" id="1" type="Power" precision="FP32">
            <power_data power="1" scale="1" shift="0"/>
            <input>
                <port id="0"/>
            </input>
            <output>
                <port id="1"/>
            </output>
        </layer>
        <layer name="proposal" type="SimplerNMS" precision="FP32" id="2">
            <data cls_threshold="_CLS_THR_" max_num_proposals="_MAX_NUM_"
                iou_threshold="_IOU_THR_" min_bbox_size="_MIN_BB_SIZE_" feat_stride="_FEAT_STRIDE_"
                pre_nms_topn="_PRE_NMS_TOPN_" post_nms_topn="_POST_NMS_TOPN_"
                scale="_SCALE1_,_SCALE2_,_SCALE3_"/>
            <input>
                <port id="2">
                    <dim>18</dim>
                    <dim>39</dim>
                    <dim>64</dim>
                </port>
                <port id="3">
                    <dim>18</dim>
                    <dim>39</dim>
                    <dim>64</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>300</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
)V0G0N";
    
    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
)V0G0N";

    std::string getModel(simplernms_test_params p) {
        std::string model = layers_t;

        REPLACE_WITH_NUM(model, "_IN_", p.in.w);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);

        REPLACE_WITH_NUM(model, "_CLS_THR_", p.cls_threshold);
        REPLACE_WITH_NUM(model, "_MAX_NUM_", p.max_num_proposals);
        REPLACE_WITH_NUM(model, "_IOU_THR_", p.iou_threshold);
        REPLACE_WITH_NUM(model, "_MIN_BB_SIZE_", p.min_bbox_size);
        REPLACE_WITH_NUM(model, "_FEAT_STRIDE_", p.feat_stride);
        REPLACE_WITH_NUM(model, "_PRE_NMS_TOPN_", p.pre_nms_topn);
        REPLACE_WITH_NUM(model, "_POST_NMS_TOPN_", p.post_nms_topn);
        REPLACE_WITH_NUM(model, "_SCALE1_", p.scale1);
        REPLACE_WITH_NUM(model, "_SCALE2_", p.scale2);
        REPLACE_WITH_NUM(model, "_SCALE3_", p.scale3);

        model = IRTemplateGenerator::getIRTemplate("SimplerNMS_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            simplernms_test_params p = ::testing::WithParamInterface<simplernms_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());

            SizeVector dims_src = {p.in.w,
                p.in.h,
                p.in.c,
                1};

            Blob::Ptr src = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), NCHW));
            src->allocate();
            fill_data(src->buffer().as<float *>(), src->size());

            SizeVector dims_dst = {300, 5, 1};

            Blob::Ptr dst = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst->allocate();

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            OutputsDataMap outInfo;
            outInfo = network.getOutputsInfo();
            ASSERT_EQ(outInfo.size(), 1);
            ASSERT_NE(outInfo.begin()->second, nullptr);
            inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            inferRequest.SetBlob(outInfo.begin()->first, dst);
            inferRequest.Infer();

        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNSimplerNMSOnlyTest, nightly_TestSimplerNMS) {}