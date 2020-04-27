// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include <ie_core.hpp>
#include "ir_gen_helper.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace single_layer_tests;

struct roipooling_test_params {
    std::string device_name;

    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    size_t pooled_h;
    size_t pooled_w;
    float spatial_scale;
};

template <typename data_t>
void ref_roipool(const TBlob<data_t> &src, TBlob<data_t> &dst, roipooling_test_params prm)
{
}

class MKLDNNROIPoolingOnlyTest: public TestsCommon,
                             public WithParamInterface<roipooling_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="roi_pool" type="ROIPooling" precision="FP32" id="1">
            <data pooled_h="_POOLED_H_" pooled_w="_POOLED_H_" spatial_scale="_SPATIAL_SCALE_"/>
            <input>
                <port id="10">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IW_</dim>
                    <dim>_IH_</dim>
                </port>
                <port id="11">
                    <dim>300</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="12">
                    <dim>300</dim>
                    <dim>256</dim>
                    <dim>6</dim>
                    <dim>6</dim>
                </port>
            </output>
        </layer>
    </layers>
)V0G0N";
    
    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="10"/>
)V0G0N";

    std::string getModel(roipooling_test_params p) {
        std::string model = layers_t;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);

        REPLACE_WITH_NUM(model, "_POOLED_H_", p.pooled_h);
        REPLACE_WITH_NUM(model, "_POOLED_W_", p.pooled_w);
        REPLACE_WITH_NUM(model, "_SPATIAL_SCALE_", p.spatial_scale);

        model = IRTemplateGenerator::getIRTemplate("ROIPooling_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            roipooling_test_params p = ::testing::WithParamInterface<roipooling_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core ie;
            ASSERT_NO_THROW(ie.ReadNetwork(model, Blob::CPtr()));

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNROIPoolingOnlyTest, nightly_TestsROIPooling) {}