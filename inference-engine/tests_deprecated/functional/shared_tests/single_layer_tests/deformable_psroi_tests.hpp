// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "common_test_utils/data_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct deformable_psroi_test_params {
    std::string device_name;

    std::vector<size_t> src_dims;
    std::vector<size_t> bbox_dims;
    std::vector<size_t> out_dims;
    float spatial_scale;
    size_t output_dim;
    size_t group_size;
    size_t pooled_height;
    size_t pooled_width;
    int part_size;
    int sample_per_part;
    bool no_trans;
    float trans_std;
    std::vector<size_t> trans_dims;
};

inline float bilinear_interp(const float* data, const float x, const float y, const int width, const int height) {
    int x1 = static_cast<int>(std::floor(x));
    int x2 = static_cast<int>(std::ceil(x));
    int y1 = static_cast<int>(std::floor(y));
    int y2 = static_cast<int>(std::ceil(y));
    float dist_x = x - x1;
    float dist_y = y - y1;
    float value11 = data[y1 * width + x1];
    float value12 = data[y2 * width + x1];
    float value21 = data[y1 * width + x2];
    float value22 = data[y2 * width + x2];
    float value = (1 - dist_x) * (1 - dist_y) * value11 + (1 - dist_x) * dist_y * value12
                  + dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
    return value;
}

static void ref_deformable_psroi(const std::vector<Blob::Ptr> &srcs, std::vector<Blob::Ptr> &dsts, deformable_psroi_test_params prm) {
    float* dst_data = dsts[0]->buffer();
    const float *bottom_data_beginning = srcs[1]->buffer();
    const float *bottom_rois_beginning = srcs[0]->buffer();

    SizeVector inDims = srcs[1]->getTensorDesc().getDims();
    int channels = static_cast<int>(inDims[1]);
    int height = static_cast<int>(inDims[2]);
    int width = static_cast<int>(inDims[3]);

    SizeVector outDims = dsts[0]->getTensorDesc().getDims();
    int nn = static_cast<int>(outDims[0]);
    int nc = static_cast<int>(outDims[1]);
    int nh = static_cast<int>(outDims[2]);
    int nw = static_cast<int>(outDims[3]);

    int real_rois = 0;
    for (; real_rois < nn; real_rois++) {
        const float *bottom_rois = bottom_rois_beginning + real_rois * 5;
        int roi_batch_ind = static_cast<int>(bottom_rois[0]);
        if (roi_batch_ind == -1) {
            break;
        }
    }

    float *bottom_trans = nullptr;
    int num_classes = 1;
    int channels_each_class = prm.output_dim;
    if (srcs.size() == 3) {
        bottom_trans = srcs[2]->buffer();
        num_classes = static_cast<int>(srcs[2]->getTensorDesc().getDims()[1]) / 2;
        channels_each_class /= num_classes;
    }

    for (int n = 0; n < real_rois; n++) {
        const float *bottom_rois = bottom_rois_beginning + n * 5;
        int roi_batch_ind = static_cast<int>(bottom_rois[0]);
        float roi_start_w = static_cast<float>(round(bottom_rois[1])) * prm.spatial_scale - 0.5;
        float roi_start_h = static_cast<float>(round(bottom_rois[2])) * prm.spatial_scale - 0.5;
        float roi_end_w = static_cast<float>(round(bottom_rois[3]) + 1.0) * prm.spatial_scale - 0.5;
        float roi_end_h = static_cast<float>(round(bottom_rois[4]) + 1.0) * prm.spatial_scale - 0.5;
        float roi_width = std::max(static_cast<double>(roi_end_w - roi_start_w), 0.1);
        float roi_height = std::max(static_cast<double>(roi_end_h - roi_start_h), 0.1);

        for (int c = 0; c < nc; c++) {
            for (int h = 0; h < nh; h++) {
                for (int w = 0; w < nw; w++) {
                    size_t index = n*nc*nh*nw + c*nh*nw + h*nw + w;
                    dst_data[index] = 0.0f;

                    float bin_size_h = roi_height / static_cast<float>(prm.pooled_height);
                    float bin_size_w = roi_width  / static_cast<float>(prm.pooled_width);

                    float sub_bin_size_h = bin_size_h / static_cast<float>(prm.sample_per_part);
                    float sub_bin_size_w = bin_size_w / static_cast<float>(prm.sample_per_part);

                    int part_h = static_cast<int>(std::floor(static_cast<float>(h) / prm.pooled_height * prm.part_size));
                    int part_w = static_cast<int>(std::floor(static_cast<float>(w) / prm.pooled_width * prm.part_size));

                    int class_id = c / channels_each_class;
                    float trans_x = prm.no_trans ? 0 :
                                    bottom_trans[(((n * num_classes + class_id) * 2) * prm.part_size + part_h)
                                                 * prm.part_size + part_w] * prm.trans_std;
                    float trans_y = prm.no_trans ? 0 :
                                    bottom_trans[(((n * num_classes + class_id) * 2 + 1) * prm.part_size + part_h)
                                                 * prm.part_size + part_w] * prm.trans_std;

                    float wstart = w * bin_size_w + roi_start_w + trans_x * roi_width;
                    float hstart = h * bin_size_h + roi_start_h + trans_y * roi_height;

                    float sum = 0;
                    int count = 0;
                    int gw = (static_cast<float>(w) * prm.group_size / prm.pooled_width );
                    int gh = (static_cast<float>(h) * prm.group_size / prm.pooled_height );
                    gw = std::min(std::max(gw, 0), static_cast<int>(prm.group_size - 1));
                    gh = std::min(std::max(gh, 0), static_cast<int>(prm.group_size - 1));

                    const float* offset_bottom_data = bottom_data_beginning + (roi_batch_ind * channels) * height * width;
                    for (size_t ih = 0; ih < prm.sample_per_part; ih++) {
                        for (size_t iw = 0; iw < prm.sample_per_part; iw++) {
                            float w1 = wstart + iw * sub_bin_size_w;
                            float h1 = hstart + ih * sub_bin_size_h;
                            // bilinear interpolation
                            if (w1 < -0.5 || w1 > width - 0.5 || h1 < -0.5 || h1 > height - 0.5)
                                continue;
                            w1 = std::min(std::max(w1, 0.0f), width - 1.0f);
                            h1 = std::min(std::max(h1, 0.0f), height - 1.0f);
                            int c1 = static_cast<int>((c * prm.group_size + gh) * prm.group_size + gw);
                            float val = bilinear_interp(offset_bottom_data + c1 * height * width, w1, h1, width, height);
                            sum += val;
                            count++;
                        }
                    }
                    dst_data[index] = count == 0 ? 0 : sum / count;
                }
            }
        }
    }
    for (int n = real_rois; n < nn; n++) {
        for (int c = 0; c < nc; c++) {
            for (int h = 0; h < nh; h++) {
                for (int w = 0; w < nw; w++) {
                    int index = n * nc * nh * nw + c * nh * nw + h * nw + w;
                    dst_data[index] = 0.0f;
                }
            }
        }
    }
}

class DeformablePSROIOnlyTest : public TestsCommon,
                        public WithParamInterface<deformable_psroi_test_params> {

    std::string model_t = R"V0G0N(
<net name="DeformablePSROIOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="data" id="0" type="Input" precision="FP32">
            <output>
                <port id="0">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="bbox" id="1" type="Input" precision="FP32">
            <output>
                <port id="0">__BBOX_DIMS__
                </port>
            </output>
        </layer>__TRANS__
        <layer name="psroi" id="3" type="PSROIPooling" precision="FP32">
            <data mode="bilinear_deformable" no_trans="__NO_TRANS__" spatial_scale="__SPATIAL_SCALE__" output_dim="__OUTPUT_DIM__" part_size="__PART_SIZE__" group_size="__GROUP_SIZE__"
                pooled_height="__POOLED_HEIGHT__" pooled_width="__POOLED_WIDTH__" spatial_bins_x="__SAMPLE_PER_PART__" spatial_bins_y="__SAMPLE_PER_PART__"__TRANS_PARAMS__/>
            <input>
                <port id="0">__SRC_DIMS__
                </port>
                <port id="1">__BBOX_DIMS__
                </port>__TRANS_DIMS__
            </input>
            <output>
                <port id="0">__OUT_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>__EDGE_TRANS__
    </edges>
</net>
)V0G0N";

    std::string getModel(deformable_psroi_test_params p) {
        std::string model = model_t;

        std::string no_trans = "True";
        std::string trans = "";
        std::string trans_params = "";
        std::string trans_dims = "";
        std::string edge_trans = "";
        if (!p.no_trans) {
            no_trans = "False";

            trans = R"VOGON(
        <layer name="trans" id="2" type="Input" precision="FP32">
            <output>__TRANS_DIMS__
            </output>
        </layer>)VOGON";

            trans_params += " trans_std=\"" + std::to_string(p.trans_std) + "\"";

            trans_dims += "\n                <port id=\"2\">";
            for (auto &dim : p.trans_dims) {
                trans_dims += "\n                    <dim>";
                trans_dims += std::to_string(dim) + "</dim>";
            }
            trans_dims += "\n                </port>";

            edge_trans = "\n        <edge from-layer=\"2\" from-port=\"2\" to-layer=\"3\" to-port=\"2\"/>";
        }
        REPLACE_WITH_STR(model, "__TRANS__", trans);
        REPLACE_WITH_STR(model, "__TRANS_PARAMS__", trans_params);
        REPLACE_WITH_STR(model, "__TRANS_DIMS__", trans_dims);
        REPLACE_WITH_STR(model, "__EDGE_TRANS__", edge_trans);

        std::string src_dims = "";
        for (auto &dim : p.src_dims) {
            src_dims += "\n                    <dim>";
            src_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS__", src_dims);
        std::string bbox_dims = "";
        for (auto &dim : p.bbox_dims) {
            bbox_dims += "\n                    <dim>";
            bbox_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__BBOX_DIMS__", bbox_dims);
        std::string out_dims = "";
        for (auto &dim : p.out_dims) {
            out_dims += "\n                    <dim>";
            out_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__OUT_DIMS__", out_dims);

        REPLACE_WITH_STR(model, "__NO_TRANS__", no_trans);
        REPLACE_WITH_NUM(model, "__SPATIAL_SCALE__", p.spatial_scale);
        REPLACE_WITH_NUM(model, "__OUTPUT_DIM__", p.output_dim);
        REPLACE_WITH_NUM(model, "__PART_SIZE__", p.part_size);
        REPLACE_WITH_NUM(model, "__GROUP_SIZE__", p.group_size);
        REPLACE_WITH_NUM(model, "__POOLED_HEIGHT__", p.pooled_height);
        REPLACE_WITH_NUM(model, "__POOLED_WIDTH__", p.pooled_width);
        REPLACE_WITH_NUM(model, "__SAMPLE_PER_PART__", p.sample_per_part);

        return model;
    }

protected:
    virtual void SetUp() {
        try {
            deformable_psroi_test_params p = ::testing::WithParamInterface<deformable_psroi_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            std::vector<Blob::Ptr> srcs_vec;

            InputsDataMap in_info_map = net.getInputsInfo();
            for (auto info : in_info_map) {
                Blob::Ptr blob = make_shared_blob<float>(
                        {Precision::FP32, info.second->getTensorDesc().getDims(), Layout::ANY});
                blob->allocate();
                if (info.second->name() == "data") {
                    CommonTestUtils::fill_data_sine(blob->buffer(), blob->size(), 1.0f, 5.0f, 0.1f);
                } else if (info.second->name() == "bbox") {
                    CommonTestUtils::fill_data_bbox(blob->buffer(), blob->size(), p.src_dims[2], p.src_dims[3], 1.0f);
                } else if (info.second->name() == "trans") {
                    CommonTestUtils::fill_data_sine(blob->buffer(), blob->size(), 0.0f, 10.0f, 1.0f);
                }

                inferRequest.SetBlob(info.first, blob);
                srcs_vec.push_back(blob);
            }

            BlobMap dsts_map;
            std::vector<Blob::Ptr> dsts_vec;

            OutputsDataMap out_info_map = net.getOutputsInfo();
            for (auto info : out_info_map) {
                Blob::Ptr blob = make_shared_blob<float>(
                        {Precision::FP32, info.second->getTensorDesc().getDims(), Layout::ANY});
                blob->allocate();
                inferRequest.SetBlob(info.first, blob);
                dsts_map[info.first] = blob;

                Blob::Ptr blob_ref = make_shared_blob<float>(
                        {Precision::FP32, info.second->getTensorDesc().getDims(), Layout::ANY});
                blob_ref->allocate();
                dsts_vec.push_back(blob_ref);
            }

            ref_deformable_psroi(srcs_vec, dsts_vec, p);

            inferRequest.Infer();

            TBlob<float>* dstPtr = dynamic_cast<TBlob<float>*>(dsts_map.begin()->second.get());
            TBlob<float>* dstrefPtr = dynamic_cast<TBlob<float>*>(dsts_vec[0].get());

            compare(*dsts_map.begin()->second, *dsts_vec[0]);

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(DeformablePSROIOnlyTest, TestsDeformable) {}

/*** TBD ***/



