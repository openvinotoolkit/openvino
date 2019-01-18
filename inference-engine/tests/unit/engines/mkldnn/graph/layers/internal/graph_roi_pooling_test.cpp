// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct roi_pooling_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in1;

    struct {
        size_t n;
        size_t c;
    } in2;

    size_t pooled_h;
    size_t pooled_w;
    float spatial_scale;

    size_t num_prim_desc;

    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_roipooling(const InferenceEngine::TBlob<data_t> &src, const InferenceEngine::TBlob<data_t> &roi,
                    InferenceEngine::TBlob<data_t> &dst_blob, roi_pooling_test_params& params) {
    data_t* dst = dst_blob.data();
    const data_t* src_data = src.readOnly();
    const data_t* src_roi = roi.readOnly();

    int C = src.dims()[1];
    int H = src.dims()[2];
    int W = src.dims()[3];

    int ROIS = roi.dims()[0];

    double spatial_scale = params.spatial_scale;
    int pooled_h = params.pooled_h;
    int pooled_w = params.pooled_w;

    data_t *arg_max_ = new data_t[dst_blob.size()];

    for (size_t i = 0; i < dst_blob.size(); i++) {
        arg_max_[i] = -1;
        dst[i] = -FLT_MAX;
    }

    int roi_off;

    for (int n = 0; n < ROIS; ++n) {
        if(roi.dims().size() == 4) {
            roi_off = n*roi.dims()[1]*roi.dims()[2]*roi.dims()[3];
        }
        else {
            roi_off = n*roi.dims()[1];
        }

        const data_t* src_roi_ptr = &src_roi[roi_off];

        int roi_batch_ind = src_roi_ptr[0];
        int roi_start_w = round(src_roi_ptr[1] * spatial_scale);
        int roi_start_h = round(src_roi_ptr[2] * spatial_scale);
        int roi_end_w = round(src_roi_ptr[3] * spatial_scale);
        int roi_end_h = round(src_roi_ptr[4] * spatial_scale);

        int roi_height = (std::max)(roi_end_h - roi_start_h + 1, 1);
        int roi_width = (std::max)(roi_end_w - roi_start_w + 1, 1);

        for (int c = 0; c < C; ++c) {

            for (int ph = 0; ph < pooled_h; ++ph) {
                for (int pw = 0; pw < pooled_w; ++pw) {
                    int hstart = (ph * roi_height) / pooled_h;
                    if ( (hstart * pooled_h) > (ph * roi_height) ) {
                        --hstart;
                    }

                    int wstart = (pw * roi_width) / pooled_w;
                    if ( (wstart * pooled_w) > (pw * roi_width) ) {
                        --wstart;
                    }

                    int hend = ((ph + 1) * roi_height) / pooled_h;
                    if ( (hend * pooled_h) < ((ph + 1) * roi_height) ) {
                        ++hend;
                    }

                    int wend = ((pw + 1) * roi_width) / pooled_w;
                    if ( (wend * pooled_w) < ((pw + 1) * roi_width) ) {
                        ++wend;
                    }

                    hstart = (std::min)((std::max)(hstart + roi_start_h, 0), H);
                    hend = (std::min)((std::max)(hend + roi_start_h, 0), H);
                    wstart = (std::min)((std::max)(wstart + roi_start_w, 0), W);
                    wend = (std::min)((std::max)(wend + roi_start_w, 0), W);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);

                    const int pool_index = n*dst_blob.dims()[2]*dst_blob.dims()[1]*dst_blob.dims()[0] +
                            c*dst_blob.dims()[1]*dst_blob.dims()[0] + ph*dst_blob.dims()[0] + pw;

                    if (is_empty) {
                        dst[pool_index] = 0;
                        arg_max_[pool_index] = -1;
                    }

                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            int src_index_data = roi_batch_ind*src.dims()[1]*src.dims()[2]*src.dims()[3] +
                                                 c*src.dims()[2]*src.dims()[3] + h*src.dims()[3] + w;
                            data_t batch_data = src_data[src_index_data];

                            if (batch_data > dst[pool_index]) {
                                dst[pool_index] = batch_data;
                                arg_max_[pool_index] = batch_data;
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] arg_max_;
}

class MKLDNNGraphRoiPoolingTests: public TestsCommon,
                                     public WithParamInterface<roi_pooling_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="ROIPooling_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN1_</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>_IN2_</dim>
                    <dim>_IC2_</dim>
                </port>
            </output>
        </layer>
        <layer name="roi_pool" id="2" type="ROIPooling" precision="FP32">
            <data pooled_h="_PH_" pooled_w="_PW_" spatial_scale="_SS_"/>
            <input>
                <port id="2">
                    <dim>_IN1_</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
                <port id="3">
                    <dim>_IN2_</dim>
                    <dim>_IC2_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_ON_</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
    </edges>
</Net>
)V0G0N";

    std::string getModel(roi_pooling_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW1_", p.in1.w);
        REPLACE_WITH_NUM(model, "_IH1_", p.in1.h);
        REPLACE_WITH_NUM(model, "_IC1_", p.in1.c);
        REPLACE_WITH_NUM(model, "_IN1_", p.in1.n);

        REPLACE_WITH_NUM(model, "_IC2_", p.in2.c);
        REPLACE_WITH_NUM(model, "_IN2_", p.in2.n);

        REPLACE_WITH_NUM(model, "_OW_", p.pooled_w);
        REPLACE_WITH_NUM(model, "_OH_", p.pooled_h);
        REPLACE_WITH_NUM(model, "_OC_", (std::max)(p.in1.c, p.in2.c));
        REPLACE_WITH_NUM(model, "_ON_", (std::max)(p.in1.n, p.in2.n));

        REPLACE_WITH_NUM(model, "_PH_", p.pooled_h);
        REPLACE_WITH_NUM(model, "_PW_", p.pooled_w);
        REPLACE_WITH_NUM(model, "_SS_", p.spatial_scale);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            roi_pooling_test_params p = ::testing::WithParamInterface<roi_pooling_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::ROIPooling) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }
            InferenceEngine::SizeVector dims_src = {p.in1.n, p.in1.c, p.in1.h, p.in1.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::SizeVector dims_roi = {p.in2.n, p.in2.c};

            InferenceEngine::Blob::Ptr roi = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NC, dims_roi);
            roi->allocate();
            fill_data(roi->buffer(), roi->size());

            InferenceEngine::TBlob<float>* roiPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(roi.get());

            if (roiPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", roi));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_roipooling(*srcPtr, *roiPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphRoiPoolingTests, TestsRoiPooling) {}


INSTANTIATE_TEST_CASE_P(
        TestsRoiPooling, MKLDNNGraphRoiPoolingTests,
        ::testing::Values(
                roi_pooling_test_params{
                        {1, 256, 39, 64}, {150, 5}, 6, 6, 0.0625f, 5, MKLDNNPlugin::impl_desc_type::jit}));
