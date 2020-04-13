// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"
#include <cpp/ie_cnn_net_reader.h>

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct simplernms_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in_cls;

    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in_delta;

    struct {
        size_t n;
        size_t c;
    } in_info;

    struct {
        size_t n;
        size_t c;
    } out;

    size_t minBoxSize;
    size_t featStride;
    size_t preNmsTopn;
    size_t postNmsTopn;
    float iouThreshold;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

struct anchor { float start_x; float start_y; float end_x; float end_y; };

template <typename data_t>
struct simpler_nms_roi_t
{
    data_t x0, y0, x1, y1;

    constexpr static inline const data_t clamp_v(const data_t v, const data_t v_min, const data_t v_max)
    {
        return (std::max)(v_min, (std::min)(v, v_max));
    }

    data_t area() const { return std::max<data_t>(0, y1 - y0 + 1) * std::max<data_t>(0, x1 - x0 + 1); }

    simpler_nms_roi_t intersect (simpler_nms_roi_t other) const
    {
        return
                {
                        (std::max)(x0, other.x0),
                        (std::max)(y0, other.y0),
                        (std::min)(x1, other.x1),
                        (std::min)(y1, other.y1)
                };
    }
    simpler_nms_roi_t clamp (simpler_nms_roi_t other) const
    {
        return
                {
                        clamp_v(x0, other.x0, other.x1),
                        clamp_v(y0, other.y0, other.y1),
                        clamp_v(x1, other.x0, other.x1),
                        clamp_v(y1, other.y0, other.y1)
                };
    }
};

template <typename data_t>
struct simpler_nms_proposal_t { simpler_nms_roi_t<data_t> roi; data_t confidence; size_t ord; };
template <typename data_t>
struct simpler_nms_delta_t { data_t shift_x, shift_y, log_w, log_h; };

template <typename data_t>
inline simpler_nms_roi_t<data_t> simpler_nms_gen_bbox(
        const anchor& box,
        const simpler_nms_delta_t<data_t>& delta,
        int anchor_shift_x,
        int anchor_shift_y)
{
    auto anchor_w = box.end_x - box.start_x + 1;
    auto anchor_h = box.end_y - box.start_y + 1;
    auto center_x = box.start_x + anchor_w * .5f;
    auto center_y = box.start_y + anchor_h *.5f;

    data_t pred_center_x = delta.shift_x * anchor_w + center_x + anchor_shift_x;
    data_t pred_center_y = delta.shift_y * anchor_h + center_y + anchor_shift_y;
    data_t half_pred_w = exp(delta.log_w) * anchor_w * .5f;
    data_t half_pred_h = exp(delta.log_h) * anchor_h * .5f;

    return { pred_center_x - half_pred_w,
             pred_center_y - half_pred_h,
             pred_center_x + half_pred_w,
             pred_center_y + half_pred_h };
}
template <typename data_t>
inline void sort_and_keep_at_most_top_n(std::vector<simpler_nms_proposal_t<data_t>>& proposals, size_t top_n)
{
    const auto cmp_fn = [](const simpler_nms_proposal_t<data_t>& a,
                           const simpler_nms_proposal_t<data_t>& b)
    {
        return a.confidence > b.confidence || (a.confidence == b.confidence && a.ord > b.ord);
    };

    if (proposals.size() > top_n) {
        std::partial_sort(proposals.begin(), proposals.begin() + top_n, proposals.end(), cmp_fn);
        proposals.resize(top_n);
    }
    else {
        std::sort(proposals.begin(), proposals.end(), cmp_fn);
    }
}

template <typename data_t>
std::vector<simpler_nms_roi_t<data_t>> simpler_nms_perform_nms(const std::vector<simpler_nms_proposal_t<data_t>>& proposals,
                                                       float iou_threshold, size_t top_n) {
    //TODO(ruv): can I mark the 1st arg, proposals as const? ifndef DONT_PRECALC_AREA, i can
    //TODO(ruv): is it better to do the precalc or not? since we need to fetch the floats from memory anyway for -
    //           intersect calc, it's only a question of whether it's faster to do (f-f)*(f-f) or fetch another val
#define DONT_PRECALC_AREA

#ifndef DONT_PRECALC_AREA
    std::vector<Dtype> areas;
    areas.reserve(proposals.size());
    std::transform(proposals.begin(), proposals.end(), areas.begin(), [](const simpler_nms_proposals_t>& v)
                   {
                       return v.roi.area();
                   });
#endif

    std::vector<simpler_nms_roi_t<data_t>> res;
    res.reserve(top_n);
#ifdef DONT_PRECALC_AREA
    for (const auto & prop : proposals) {
        const auto bbox = prop.roi;
        const data_t area = bbox.area();
#else
        size_t proposal_count = proposals.size();
        for (size_t proposalIndex = 0; proposalIndex < proposal_count; ++proposalIndex) {
            const auto & bbox = proposals[proposalIndex].roi;
#endif

        // For any realistic WL, this condition is true for all top_n values anyway
        if (prop.confidence > 0) {
            bool overlaps = std::any_of(res.begin(), res.end(), [&](const simpler_nms_roi_t<data_t>& res_bbox)
            {
                data_t interArea = bbox.intersect(res_bbox).area();
#ifdef DONT_PRECALC_AREA
                data_t unionArea = res_bbox.area() + area - interArea;
#else
                data_t unionArea = res_bbox.area() + areas[proposalIndex] - interArea;
#endif
                return interArea > iou_threshold * unionArea;
            });

            if (! overlaps) {
                res.push_back(bbox);
                if (res.size() == top_n) break;
            }
        }
    }

    return res;
}

template <typename data_t>
void ref_simplernms(const InferenceEngine::TBlob<data_t> &src_cls, const InferenceEngine::TBlob<data_t> &src_delta, const InferenceEngine::TBlob<data_t> &src_info, InferenceEngine::TBlob<data_t> &dst_blob, simplernms_test_params prm) {
    int anchors_num = 3 * 3;
    data_t *anchors_ = new data_t[anchors_num * sizeof(anchor) / sizeof(float)];
    const anchor* anchors = (anchor*)anchors_;

    IE_ASSERT(src_cls.getTensorDesc().getDims().size() == 4);
    int H = src_cls.getTensorDesc().getDims()[2];
    int W = src_cls.getTensorDesc().getDims()[3];

    int SZ = H * W;

    data_t* dst = dst_blob.data();

    const data_t* cls_scores = src_cls.readOnly();
    const data_t* delta_pred = src_delta.readOnly();
    const data_t* im_info = src_info.readOnly();

    int IW = im_info[0];
    int IH = im_info[1];
    int IS = im_info[2];

    int scaled_min_bbox_size = prm.minBoxSize * IS;

    std::vector<simpler_nms_proposal_t<data_t>> sorted_proposals_confidence;

    for (auto y = 0; y < H; ++y)
    {
        int anchor_shift_y = y * prm.featStride;

        for (auto x = 0; x < W; ++x) {
            int anchor_shift_x = x * prm.featStride;
            int location_index = y * W + x;

            // we assume proposals are grouped by window location
            for (int anchor_index = 0; anchor_index < anchors_num ; anchor_index++) {
                data_t dx0 = delta_pred[location_index + SZ * (anchor_index * 4 + 0)];
                data_t dy0 = delta_pred[location_index + SZ * (anchor_index * 4 + 1)];
                data_t dx1 = delta_pred[location_index + SZ * (anchor_index * 4 + 2)];
                data_t dy1 = delta_pred[location_index + SZ * (anchor_index * 4 + 3)];

                simpler_nms_delta_t<data_t> bbox_delta { dx0, dy0, dx1, dy1 };

                data_t proposal_confidence = cls_scores[location_index + SZ * (anchor_index + anchors_num * 1)];

                simpler_nms_roi_t<data_t> tmp_roi = simpler_nms_gen_bbox(anchors[anchor_index], bbox_delta, anchor_shift_x, anchor_shift_y);
                simpler_nms_roi_t<data_t> roi = tmp_roi.clamp({ 0, 0, data_t(IW - 1), data_t(IH - 1) });

                int bbox_w = roi.x1 - roi.x0 + 1;
                int bbox_h = roi.y1 - roi.y0 + 1;

                if (bbox_w >= scaled_min_bbox_size && bbox_h >= scaled_min_bbox_size) {
                    simpler_nms_proposal_t<data_t> proposal { roi, proposal_confidence, sorted_proposals_confidence.size() };
                    sorted_proposals_confidence.push_back(proposal);
                }
            }
        }
    }

    sort_and_keep_at_most_top_n(sorted_proposals_confidence, prm.preNmsTopn);
    auto res = simpler_nms_perform_nms(sorted_proposals_confidence, prm.iouThreshold, prm.postNmsTopn);

    size_t res_num_rois = res.size();

    for (size_t i = 0; i < res_num_rois; ++i) {
        dst[5 * i + 0] = 0;    // roi_batch_ind, always zero on test time
        dst[5 * i + 1] = res[i].x0;
        dst[5 * i + 2] = res[i].y0;
        dst[5 * i + 3] = res[i].x1;
        dst[5 * i + 4] = res[i].y1;
    }

    delete[] anchors_;
}

class MKLDNNGraphSimplerNMSTests: public TestsCommon,
                                     public WithParamInterface<simplernms_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Lrn_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_INC_</dim>
                    <dim>_ICC_</dim>
                    <dim>_IHC_</dim>
                    <dim>_IWC_</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>_IND_</dim>
                    <dim>_ICD_</dim>
                    <dim>_IHD_</dim>
                    <dim>_IWD_</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    <dim>_INI_</dim>
                    <dim>_ICI_</dim>
                </port>
            </output>
        </layer>
        <layer name="proposal" id="3" type="SimplerNMS" precision="FP32">
            <data cls_threshold="0.500000" max_num_proposals="300" iou_threshold="_IOU_THRESHOLD_"
            min_bbox_size="_MIN_BOX_SIZE_" feat_stride="_FSRD_" pre_nms_topn="_PRENT_" post_nms_topn="_POSTNT_"
            scale="8.000000,16.000000,32.000000"/>

            <input>
                <port id="3">
                    <dim>_INC_</dim>
                    <dim>_ICC_</dim>
                    <dim>_IHC_</dim>
                    <dim>_IWC_</dim>
                </port>
                <port id="4">
                    <dim>_IND_</dim>
                    <dim>_ICD_</dim>
                    <dim>_IHD_</dim>
                    <dim>_IWD_</dim>
                </port>
                <port id="5">
                    <dim>_INI_</dim>
                    <dim>_ICI_</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>_ON_</dim>
                    <dim>_OC_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="3"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="5"/>
    </edges>
</Net>
)V0G0N";

    std::string getModel(simplernms_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IWC_", p.in_cls.w);
        REPLACE_WITH_NUM(model, "_IHC_", p.in_cls.h);
        REPLACE_WITH_NUM(model, "_ICC_", p.in_cls.c);
        REPLACE_WITH_NUM(model, "_INC_", p.in_cls.n);

        REPLACE_WITH_NUM(model, "_IWD_", p.in_delta.w);
        REPLACE_WITH_NUM(model, "_IHD_", p.in_delta.h);
        REPLACE_WITH_NUM(model, "_ICD_", p.in_delta.c);
        REPLACE_WITH_NUM(model, "_IND_", p.in_delta.n);

        REPLACE_WITH_NUM(model, "_ICI_", p.in_info.c);
        REPLACE_WITH_NUM(model, "_INI_", p.in_info.n);

        REPLACE_WITH_NUM(model, "_OC_", p.out.c);
        REPLACE_WITH_NUM(model, "_ON_", p.out.n);

        REPLACE_WITH_NUM(model, "_MIN_BOX_SIZE_", p.minBoxSize);
        REPLACE_WITH_NUM(model, "_FSRD_", p.featStride);
        REPLACE_WITH_NUM(model, "_PRENT_", p.preNmsTopn);
        REPLACE_WITH_NUM(model, "_POSTNT_", p.postNmsTopn);
        REPLACE_WITH_NUM(model, "_IOU_THRESHOLD_", p.iouThreshold);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            simplernms_test_params p = ::testing::WithParamInterface<simplernms_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::SimplerNMS) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }
            InferenceEngine::SizeVector dims_src_cls = {p.in_cls.n, p.in_cls.c, p.in_cls.h, p.in_cls.w};

            InferenceEngine::Blob::Ptr src_cls = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src_cls, InferenceEngine::NCHW});
            src_cls->allocate();
            fill_data(src_cls->buffer(), src_cls->size());

            InferenceEngine::TBlob<float>* srcClsPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src_cls.get());

            if (srcClsPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::SizeVector dims_delta = {p.in_delta.n, p.in_delta.c, p.in_delta.h, p.in_delta.w};

            InferenceEngine::Blob::Ptr src_delta = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_delta, InferenceEngine::NCHW});
            src_delta->allocate();
            fill_data(src_delta->buffer(), src_delta->size());

            InferenceEngine::TBlob<float>* srcDeltaPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src_delta.get());

            if (srcDeltaPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::SizeVector dims_info = {p.in_info.n, p.in_info.c};

            InferenceEngine::Blob::Ptr src_info = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_info, InferenceEngine::NC});
            src_info->allocate();
            fill_data(src_info->buffer(), src_info->size());
            float * data_info = src_info->buffer();
            data_info[0] = 20;
            data_info[1] = 20;
            data_info[2] = 3;

            InferenceEngine::TBlob<float>* srcInfoPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src_info.get());

            if (srcInfoPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src_cls));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src_delta));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in3", src_info));

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

            ref_simplernms(*srcClsPtr, *srcDeltaPtr, *srcInfoPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphSimplerNMSTests, TestsSimplerNMS) {}


INSTANTIATE_TEST_CASE_P(
        DISABLED_TestsSimplerNMS, MKLDNNGraphSimplerNMSTests,
        ::testing::Values(
                simplernms_test_params{{1, 18, 39, 64}, {1, 36, 39, 64}, {1, 3}, {150, 5}, 16, 16, 6000, 150, 0.7f, 1,
                                       MKLDNNPlugin::impl_desc_type::ref, {
                                         [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                             ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                                             ASSERT_EQ(3, impl.getConfig().inConfs.size());
                                             ASSERT_EQ(1, impl.getConfig().outConfs.size());
                                             ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(0).desc.getLayout());
                                             ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(1).desc.getLayout());
                                             ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().inConfs.at(2).desc.getLayout());
                                             ASSERT_EQ(InferenceEngine::Layout::ANY, impl.getConfig().outConfs.at(0).desc.getLayout());
                                         }
                                 }}));
