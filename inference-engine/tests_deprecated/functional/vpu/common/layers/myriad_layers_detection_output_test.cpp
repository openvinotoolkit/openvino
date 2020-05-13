// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <thread>
#include "../bbox_util.h"
#include "myriad_layers_tests.hpp"
#include "ie_memcpy.h"

//#define USE_OPENCV_IMSHOW

#ifdef USE_OPENCV_IMSHOW
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif

using namespace InferenceEngine;

namespace {

const int WIDTH = 300;
const int HEIGHT = 300;

const size_t NUM_LOC = 12996;
const size_t NUM_CONF = 6498;
const size_t NUM_OUT = 200;

const int MAX_NONMATCH_OBJS = 0;

const int NUM_CLASSES = 2;
const int BACKGROUND_LABEL_ID = 0;
const int TOP_K = 400;
const int VARIANCE_ENCODED_IN_TARGET = 0;
const int KEEP_TOP_K = 200;
const CodeType CODE_TYPE = CENTER_SIZE;
const int SHARE_LOCATION = 1;
const int INTERPOLATE_ORIENTATION = 0;
const float NMS_THRESHOLD = 0.45f;
const float CONFIDENCE_THRESHOLD = 0.01f;
const int NUM_ORIENT_CLASSES = 0;

void refDetectionOutput(const TBlob<float>::Ptr &locations,
                        const TBlob<float>::Ptr &confidence,
                        const TBlob<float>::Ptr &prior,
                        TBlob<float>::Ptr &output,
                        const bool decrease_label_id) {
    int _num_loc_classes = SHARE_LOCATION ? 1 : NUM_CLASSES;
    int _num_priors = prior->getTensorDesc().getDims()[2] / 4;

    ASSERT_EQ(_num_priors * _num_loc_classes * 4, locations->getTensorDesc().getDims()[1])
        << "Number of priors must match number of location predictions";
    ASSERT_EQ(_num_priors * NUM_CLASSES, confidence->getTensorDesc().getDims()[1])
        << "Number of priors must match number of confidence predictions";

    float* dst_data = output->data();

    const float* loc_data = locations->readOnly();
    const float* orient_data = nullptr; // TODO : support orientation, when myriad will do it
    const float* conf_data = confidence->readOnly();
    const float* prior_data = prior->readOnly();
    const int num = locations->getTensorDesc().getDims()[0];

    // Retrieve all location predictions.
    std::vector<LabelBBox> all_loc_preds;
    GetLocPredictions(loc_data, num, _num_priors, _num_loc_classes, SHARE_LOCATION, &all_loc_preds);

    // Retrieve all confidences.
    std::vector<std::map<int, std::vector<float>>> all_conf_scores;
    GetConfidenceScores(conf_data, num, _num_priors, NUM_CLASSES, &all_conf_scores);

    // Retrieve all orientations
    std::vector<std::vector<std::vector<float>>> all_orient_scores;
    if (orient_data) {
        GetOrientationScores(orient_data, num, _num_priors, NUM_ORIENT_CLASSES, &all_orient_scores);
    }

    // Retrieve all prior bboxes. It is same within a batch since we assume all
    // images in a batch are of same dimension.
    std::vector<NormalizedBBox> prior_bboxes(_num_priors);
    std::vector<float> prior_variances(_num_priors * 4);
    GetPriorBBoxes(prior_data, _num_priors, prior_bboxes, prior_variances);

    // Decode all loc predictions to bboxes.
    std::vector<LabelBBox> all_decode_bboxes;
    DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                    SHARE_LOCATION, _num_loc_classes, BACKGROUND_LABEL_ID,
                    CODE_TYPE, VARIANCE_ENCODED_IN_TARGET, &all_decode_bboxes);

    int num_kept = 0;

    std::vector<std::map<int, std::vector<int>>> all_indices;

    for (int image_index_in_batch = 0; image_index_in_batch < num; ++image_index_in_batch) {
        if (orient_data) {
            ASSERT_EQ(_num_priors, all_orient_scores[image_index_in_batch].size())
                << "Orientation scores not equal to num priors.";
        }

        const LabelBBox& decode_bboxes = all_decode_bboxes[image_index_in_batch];
        const std::map<int, std::vector<float>>& conf_scores = all_conf_scores[image_index_in_batch];
        std::map<int, std::vector<int> > indices;
        int num_det = 0;

        for (int label_index = 0; label_index < NUM_CLASSES; ++label_index) {
            if (label_index == BACKGROUND_LABEL_ID) {
                // Ignore background class.
                continue;
            }

            ASSERT_NE(conf_scores.end(), conf_scores.find(label_index))
                << "Could not find confidence predictions for label " << label_index;

            const std::vector<float>& scores = conf_scores.find(label_index)->second;

            int label = SHARE_LOCATION ? -1 : label_index;

            if (decode_bboxes.find(label) == decode_bboxes.end()) {
                // Something bad happened if there are no predictions for current label.
                continue;
            }

            const std::vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;

            ApplyNMSFast(bboxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, TOP_K, &(indices[label_index]));
            num_det += indices[label_index].size();
        }

        if (KEEP_TOP_K > -1 && num_det > KEEP_TOP_K) {
            std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;

            for (std::map<int, std::vector<int> >::iterator it = indices.begin(); it != indices.end(); ++it) {
                int label = it->first;

                const std::vector<int>& label_indices = it->second;

                if (conf_scores.find(label) == conf_scores.end()) {
                    // Something bad happened for current label.
                    continue;
                }

                const std::vector<float>& scores = conf_scores.find(label)->second;

                for (int j = 0; j < label_indices.size(); ++j) {
                    int idx = label_indices[j];

                    ASSERT_LT(idx, scores.size())
                        << "Label index is out of array size";

                    score_index_pairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                }
            }

            // Keep top k results per image.
            std::sort(score_index_pairs.begin(), score_index_pairs.end(), SortScorePairDescend<std::pair<int, int>>);
            score_index_pairs.resize(KEEP_TOP_K);

            // Store the new indices.
            std::map<int, std::vector<int> > new_indices;

            for (int j = 0; j < score_index_pairs.size(); ++j) {
                int label = score_index_pairs[j].second.first;
                int idx = score_index_pairs[j].second.second;
                new_indices[label].push_back(idx);
            }

            all_indices.push_back(new_indices);
            num_kept += KEEP_TOP_K;

        } else {
            all_indices.push_back(indices);
            num_kept += num_det;
        }
    }

    const int DETECTION_OUTPUT_SIZE = output->getTensorDesc().getDims()[0];
    for (int i = 0; i < num * KEEP_TOP_K * DETECTION_OUTPUT_SIZE; i++) {
        dst_data[i] = 0;
    }

    auto mark_end = [] (float* data) -> void {
        *data++ = -1;
        *data++ = 0;
        *data++ = 0;
        *data++ = 0;
        *data++ = 0;
        *data++ = 0;
        *data++ = 0;
    };

    if (num_kept == 0) {
        // Nothing to detect
        mark_end(dst_data);
        return;
    }

    int count = 0;

    for (int image_index_in_batch = 0; image_index_in_batch < num; ++image_index_in_batch) {
        const std::map<int, std::vector<float>>& conf_scores = all_conf_scores[image_index_in_batch];
        const std::vector<std::vector<float> >* p_orient_scores =  orient_data ? &(all_orient_scores[image_index_in_batch]) : NULL;

        if (orient_data) {
            ASSERT_EQ(_num_priors, p_orient_scores->size())
                << "Orientation scores not equal to num priors";
        }

        const LabelBBox& decode_bboxes = all_decode_bboxes[image_index_in_batch];

        for (auto it = all_indices[image_index_in_batch].begin(); it != all_indices[image_index_in_batch].end(); ++it) {
            int label = it->first;

            if (conf_scores.find(label) == conf_scores.end()) {
                // Something bad happened if there are no predictions for current label.
                continue;
            }

            const std::vector<float>& scores = conf_scores.find(label)->second;
            int loc_label = SHARE_LOCATION ? -1 : label;

            if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
                // Something bad happened if there are no predictions for current label.
                continue;
            }

            const std::vector<NormalizedBBox>& bboxes = decode_bboxes.find(loc_label)->second;

            ASSERT_EQ(_num_priors, bboxes.size())
                << "Bounding boxes num is not equal to num priors";

            std::vector<int>& indices = it->second;

            for (int j = 0; j < indices.size(); ++j) {
                int idx = indices[j];

                dst_data[count * DETECTION_OUTPUT_SIZE + 0] = image_index_in_batch;
                dst_data[count * DETECTION_OUTPUT_SIZE + 1] = decrease_label_id ? label - 1 : label;
                dst_data[count * DETECTION_OUTPUT_SIZE + 2] = scores[idx];

                const NormalizedBBox &clip_bbox = bboxes[idx];
                dst_data[count * DETECTION_OUTPUT_SIZE + 3] = clip_bbox.xmin();
                dst_data[count * DETECTION_OUTPUT_SIZE + 4] = clip_bbox.ymin();
                dst_data[count * DETECTION_OUTPUT_SIZE + 5] = clip_bbox.xmax();
                dst_data[count * DETECTION_OUTPUT_SIZE + 6] = clip_bbox.ymax();

                // NormalizedBBox::orientation
                if (DETECTION_OUTPUT_SIZE == 8) {
                    float orientation = -10;

                    if (p_orient_scores) {
                        orientation = get_orientation((*p_orient_scores)[idx], INTERPOLATE_ORIENTATION);
                    }

                    dst_data[count * DETECTION_OUTPUT_SIZE + 7] = orientation;
                }

                ++count;
            }
        }
    }

    // TODO: Logic is correct only for mb=1
    if (count < KEEP_TOP_K) {
        // marker at end of boxes list
        mark_end(dst_data + count * DETECTION_OUTPUT_SIZE);
    }
}

const std::string PRIOR_BOX_CLUSTERED_MODEL = R"V0G0N(
<net name="PRIOR_BOX_CLUSTERED_MODEL" version="2" batch="1">
    <layers>
        <layer name="data1" type="Input" precision="FP16" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>300</dim>
                    <dim>300</dim>
                </port>
            </output>
        </layer>
        <layer name="data1_copy" type="Power" precision="FP16" id="2">
            <power_data power="1" scale="1" shift="0"/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>300</dim>
                    <dim>300</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>300</dim>
                    <dim>300</dim>
                </port>
            </output>
        </layer>
        <layer name="data2" type="Input" precision="FP16" id="3">
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>384</dim>
                    <dim>19</dim>
                    <dim>19</dim>
                </port>
            </output>
        </layer>
        <layer name="data2_copy" type="Power" precision="FP16" id="4">
            <power_data power="1" scale="1" shift="0"/>
            <input>
                <port id="5">
                    <dim>1</dim>
                    <dim>384</dim>
                    <dim>19</dim>
                    <dim>19</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>1</dim>
                    <dim>384</dim>
                    <dim>19</dim>
                    <dim>19</dim>
                </port>
            </output>
        </layer>
        <layer name="priorboxclustered" type="PriorBoxClustered" precision="FP16" id="5">
            <data
                min_size="#"
                max_size="#"
                aspect_ratio="#"
                flip="1"
                clip="0"
                variance="0.100000,0.100000,0.200000,0.200000"
                img_size="0"
                img_h="0"
                img_w="0"
                step="16.000000"
                step_h="0.000000"
                step_w="0.000000"
                offset="0.500000"
                width="9.400000,25.100000,14.700000,34.700001,143.000000,77.400002,128.800003,51.099998,75.599998"
                height="15.000000,39.599998,25.500000,63.200001,227.500000,162.899994,124.500000,105.099998,72.599998"
            />
            <input>
                <port id="7">
                    <dim>1</dim>
                    <dim>384</dim>
                    <dim>19</dim>
                    <dim>19</dim>
                </port>
                <port id="8">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>300</dim>
                    <dim>300</dim>
                </port>
            </input>
            <output>
                <port id="9">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>12996</dim>
                </port>
            </output>
        </layer>
        <layer name="priorboxclustered_copy" type="Power" precision="FP16" id="6">
            <power_data power="1" scale="1" shift="0"/>
            <input>
                <port id="10">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>12996</dim>
                </port>
            </input>
            <output>
                <port id="11">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>12996</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="5" to-port="8"/>
        <edge from-layer="3" from-port="4" to-layer="4" to-port="5"/>
        <edge from-layer="3" from-port="4" to-layer="5" to-port="7"/>
        <edge from-layer="5" from-port="9" to-layer="6" to-port="10"/>
    </edges>
</net>
)V0G0N";

const std::string DETECTION_OUTPUT_MODEL = R"V0G0N(
<Net Name="DETECTION_OUTPUT_MYRIAD_MODEL" version="2" precision="FP16" batch="1">
<layers>
    <layer name="data1" type="Input" precision="FP16" id="1">
        <output>
            <port id="1">
                <dim>1</dim>
                <dim>3</dim>
                <dim>300</dim>
                <dim>300</dim>
            </port>
        </output>
    </layer>
    <layer name="data1_copy" type="Power" precision="FP16" id="2">
        <power_data power="1" scale="1" shift="0"/>
        <input>
            <port id="2">
                <dim>1</dim>
                <dim>3</dim>
                <dim>300</dim>
                <dim>300</dim>
            </port>
        </input>
        <output>
            <port id="3">
                <dim>1</dim>
                <dim>3</dim>
                <dim>300</dim>
                <dim>300</dim>
            </port>
        </output>
    </layer>
    <layer name="data2" type="Input" precision="FP16" id="3">
        <output>
            <port id="4">
                <dim>1</dim>
                <dim>384</dim>
                <dim>19</dim>
                <dim>19</dim>
            </port>
        </output>
    </layer>
    <layer name="data2_copy" type="Power" precision="FP16" id="4">
        <power_data power="1" scale="1" shift="0"/>
        <input>
            <port id="5">
                <dim>1</dim>
                <dim>384</dim>
                <dim>19</dim>
                <dim>19</dim>
            </port>
        </input>
        <output>
            <port id="6">
                <dim>1</dim>
                <dim>384</dim>
                <dim>19</dim>
                <dim>19</dim>
            </port>
        </output>
    </layer>
    <layer name="priorboxclustered" type="PriorBoxClustered" precision="FP16" id="5">
        <data
            min_size="#"
            max_size="#"
            aspect_ratio="#"
            flip="1"
            clip="0"
            variance="0.100000,0.100000,0.200000,0.200000"
            img_size="0"
            img_h="0"
            img_w="0"
            step="16.000000"
            step_h="0.000000"
            step_w="0.000000"
            offset="0.500000"
            width="9.400000,25.100000,14.700000,34.700001,143.000000,77.400002,128.800003,51.099998,75.599998"
            height="15.000000,39.599998,25.500000,63.200001,227.500000,162.899994,124.500000,105.099998,72.599998"/>
        <input>
            <port id="7">
                <dim>1</dim>
                <dim>384</dim>
                <dim>19</dim>
                <dim>19</dim>
            </port>
            <port id="8">
                <dim>1</dim>
                <dim>3</dim>
                <dim>300</dim>
                <dim>300</dim>
            </port>
        </input>
        <output>
            <port id="9">
                <dim>1</dim>
                <dim>2</dim>
                <dim>12996</dim>
            </port>
        </output>
    </layer>
    <layer name="locations" type="Input" precision="FP16" id="6">
        <output>
            <port id="10">
                <dim>1</dim>
                <dim>12996</dim>
            </port>
        </output>
    </layer>
    <layer name="confidence" type="Input" precision="FP16" id="7">
        <output>
            <port id="11">
                <dim>1</dim>
                <dim>6498</dim>
            </port>
        </output>
    </layer>
    <layer name="detection_out" type="DetectionOutput" precision="FP16" id="8">
        <data num_classes="2"
              share_location="1"
              background_label_id="0"
              nms_threshold="0.45"
              top_k="400"
              code_type="caffe.PriorBoxParameter.CENTER_SIZE"
              variance_encoded_in_target="0"
              keep_top_k="200"
              confidence_threshold="0.01"
              visualize="0"
              normalized="1"
        />
        <input>
            <port id="12">
                <dim>1</dim>
                <dim>12996</dim>
            </port>
            <port id="13">
                <dim>1</dim>
                <dim>6498</dim>
            </port>
            <port id="14">
                <dim>1</dim>
                <dim>2</dim>
                <dim>12996</dim>
            </port>
        </input>
        <output>
            <port id="15">
                <dim>1</dim>
                <dim>1</dim>
                <dim>200</dim>
                <dim>7</dim>
            </port>
        </output>
    </layer>
</layers>
<edges>
    <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
    <edge from-layer="1" from-port="1" to-layer="5" to-port="8"/>
    <edge from-layer="3" from-port="4" to-layer="4" to-port="5"/>
    <edge from-layer="3" from-port="4" to-layer="5" to-port="7"/>
    <edge from-layer="6" from-port="10" to-layer="8" to-port="12"/>
    <edge from-layer="7" from-port="11" to-layer="8" to-port="13"/>
    <edge from-layer="5" from-port="9" to-layer="8" to-port="14"/>
</edges>
</Net>
)V0G0N";

const std::string DETECTION_OUTPUT_MODEL_WITH_CONST = R"V0G0N(
<Net Name="DETECTION_OUTPUT_MODEL_WITH_CONST" version="2" precision="FP16" batch="1">
<layers>
    <layer name="locations" type="Input" precision="FP16" id="1">
        <output>
            <port id="1">
                <dim>1</dim>
                <dim>12996</dim>
            </port>
        </output>
    </layer>
    <layer name="confidence" type="Input" precision="FP16" id="2">
        <output>
            <port id="2">
                <dim>1</dim>
                <dim>6498</dim>
            </port>
        </output>
    </layer>
    <layer name="priorboxclustered" type="Const" precision="FP16" id="3">
        <output>
            <port id="3">
                <dim>1</dim>
                <dim>2</dim>
                <dim>12996</dim>
            </port>
        </output>
        <blobs>
            <custom offset="0" size="51984"/>
        </blobs>
    </layer>
    <layer name="detection_out" type="DetectionOutput" precision="FP16" id="4">
        <data num_classes="2"
              share_location="1"
              background_label_id="0"
              nms_threshold="0.45"
              top_k="400"
              code_type="caffe.PriorBoxParameter.CENTER_SIZE"
              variance_encoded_in_target="0"
              keep_top_k="200"
              confidence_threshold="0.01"
              visualize="0"
              normalized="1"
        />
        <input>
            <port id="4">
                <dim>1</dim>
                <dim>12996</dim>
            </port>
            <port id="5">
                <dim>1</dim>
                <dim>6498</dim>
            </port>
            <port id="6">
                <dim>1</dim>
                <dim>2</dim>
                <dim>12996</dim>
            </port>
        </input>
        <output>
            <port id="7">
                <dim>1</dim>
                <dim>1</dim>
                <dim>200</dim>
                <dim>7</dim>
            </port>
        </output>
    </layer>
</layers>
<edges>
    <edge from-layer="1" from-port="1" to-layer="4" to-port="4"/>
    <edge from-layer="2" from-port="2" to-layer="4" to-port="5"/>
    <edge from-layer="3" from-port="3" to-layer="4" to-port="6"/>
</edges>
</Net>
)V0G0N";

const std::string DETECTION_OUTPUT_MODEL_MXNET = R"V0G0N(
<Net Name="DETECTION_OUTPUT_MODEL_MXNET" version="2" precision="FP16" batch="1">
<layers>
    <layer name="locations" type="Input" precision="FP16" id="1">
        <output>
            <port id="1">
                <dim>1</dim>
                <dim>12996</dim>
            </port>
        </output>
    </layer>
    <layer name="confidence" type="Input" precision="FP16" id="2">
        <output>
            <port id="2">
                <dim>1</dim>
                <dim>6498</dim>
            </port>
        </output>
    </layer>
    <layer name="priorboxclustered" type="Const" precision="FP16" id="3">
        <output>
            <port id="3">
                <dim>1</dim>
                <dim>2</dim>
                <dim>12996</dim>
            </port>
        </output>
        <blobs>
            <custom offset="0" size="51984"/>
        </blobs>
    </layer>
    <layer name="detection_out" type="DetectionOutput" precision="FP16" id="4">
        <data num_classes="2"
              decrease_label_id="1"
              share_location="1"
              background_label_id="0"
              nms_threshold="0.45"
              top_k="400"
              code_type="caffe.PriorBoxParameter.CENTER_SIZE"
              variance_encoded_in_target="0"
              keep_top_k="200"
              confidence_threshold="0.01"
              visualize="0"
              normalized="1"
        />
        <input>
            <port id="4">
                <dim>1</dim>
                <dim>12996</dim>
            </port>
            <port id="5">
                <dim>1</dim>
                <dim>6498</dim>
            </port>
            <port id="6">
                <dim>1</dim>
                <dim>2</dim>
                <dim>12996</dim>
            </port>
        </input>
        <output>
            <port id="7">
                <dim>1</dim>
                <dim>1</dim>
                <dim>200</dim>
                <dim>7</dim>
            </port>
        </output>
    </layer>
</layers>
<edges>
    <edge from-layer="1" from-port="1" to-layer="4" to-port="4"/>
    <edge from-layer="2" from-port="2" to-layer="4" to-port="5"/>
    <edge from-layer="3" from-port="3" to-layer="4" to-port="6"/>
</edges>
</Net>
)V0G0N";

void cvt(float src, float& dst) {
    dst = src;
}
void cvt(float src, ie_fp16& dst) {
    dst = PrecisionUtils::f32tof16(src);
}
void cvt(ie_fp16 src, float& dst) {
    dst = PrecisionUtils::f16tof32(src);
}

struct BBox {
    int x, y;
    int width, height;
};

std::ostream& operator<<(std::ostream& os, const BBox& bbox)
{
    return os << "[" << bbox.x << ", " << bbox.y << ", "
              << bbox.width << ", " << bbox.height << "]";
}

struct DetectionObject {
    int batch_id;
    int class_id;
    float confidence;
    BBox bbox;
};

std::ostream& operator<<(std::ostream& os, const DetectionObject& obj)
{
    return os << "[" << obj.batch_id << ", " << obj.class_id << ", "
              << obj.confidence << ", " << obj.bbox << "]";
}

template <typename T>
void cvtDetectionObjects(T* data, int num, int width, int height, std::vector<DetectionObject>& out) {
    out.clear();
    out.reserve(num);

    for (int i = 0; i < num; ++i) {
        float batch_id, class_id, conf;
        cvt(data[i * 7 + 0], batch_id);
        cvt(data[i * 7 + 1], class_id);
        cvt(data[i * 7 + 2], conf);

        if (batch_id == -1) {
            break;
        }

        float xmin, ymin, xmax, ymax;
        cvt(data[i * 7 + 3], xmin);
        cvt(data[i * 7 + 4], ymin);
        cvt(data[i * 7 + 5], xmax);
        cvt(data[i * 7 + 6], ymax);

        BBox bbox;
        bbox.x = width * xmin;
        bbox.y = height * ymin;
        bbox.width  = width * (xmax - xmin);
        bbox.height = height * (ymax - ymin);

        out.push_back({int(batch_id), int(class_id), conf, bbox});
    }
}

bool cmpDetectionObject(const DetectionObject& first, const DetectionObject& second) {
    const float max_confidence_delta = 0.001f;
    const int max_pixel_delta = 1;

    return ((first.batch_id == second.batch_id) &&
            (first.class_id == second.class_id) &&
            (std::abs(first.bbox.x - second.bbox.x) <= max_pixel_delta) &&
            (std::abs(first.bbox.y - second.bbox.y) <= max_pixel_delta) &&
            (std::abs(first.bbox.width - second.bbox.width) <= 2*max_pixel_delta) &&
            (std::abs(first.bbox.height - second.bbox.height) <= 2*max_pixel_delta) &&
            (std::fabs(first.confidence - second.confidence) <=  max_confidence_delta));
}

void checkDetectionObjectArrays(std::vector<DetectionObject> gold, std::vector<DetectionObject> actual, int max_nonmatch_objs = 0) {
    for (auto it_gold = gold.begin(); it_gold != gold.end();) {
        std::vector<std::vector<DetectionObject>::iterator> candidates;
        for (auto it_actual = actual.begin(); it_actual != actual.end(); it_actual++) {
            if (cmpDetectionObject(*it_gold, *it_actual))
                candidates.push_back(it_actual);
        }

        if (0 == candidates.size()) {
            ++it_gold;
        } else {
            int best_index = 0;
            float best_abs_delta = std::fabs(candidates[0]->confidence - it_gold->confidence);

            for (size_t i = 1; i < candidates.size(); i++) {
                float abs_delta = std::fabs(candidates[i]->confidence - it_gold->confidence);
                if (abs_delta < best_abs_delta) {
                    best_index = i;
                    best_abs_delta = abs_delta;
                }
            }

            actual.erase(candidates[best_index]);
            it_gold = gold.erase(it_gold);
        }
    }

    for (auto miss_gold : gold) {
        std::cout << "Mistmatch in gold array: " << miss_gold << std::endl;
    }

    for (auto miss_actual : actual) {
        std::cout << "Mistmatch in actual array: " << miss_actual << std::endl;
    }

    EXPECT_LE(gold.size(), max_nonmatch_objs);
    EXPECT_LE(actual.size(), max_nonmatch_objs);
}

}

class myriadDetectionOutputTests_smoke : public myriadLayersTests_nightly {
public:
    std::vector<float> gen_locations;
    std::vector<float> gen_confidence;
    Blob::Ptr priorOutput;

    void PrepareInput() {
        gen_locations.resize(NUM_LOC);
        for (size_t i = 0; i < NUM_LOC; i += 4) {
            int x = std::rand() % WIDTH;
            int y = std::rand() % HEIGHT;
            int w = std::rand() % (WIDTH - x);
            int h = std::rand() % (HEIGHT - y);

            gen_locations[i + 0] = static_cast<float>(x) / WIDTH; // xmin
            gen_locations[i + 1] = static_cast<float>(y) / HEIGHT; // ymin
            gen_locations[i + 2] = static_cast<float>(x + w) / WIDTH; // xmax
            gen_locations[i + 3] = static_cast<float>(y + h) / HEIGHT; // ymax
        }

        gen_confidence.resize(NUM_CONF);
        for (size_t i = 0; i < NUM_CONF; ++i) {
            gen_confidence[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        StatusCode st;

        InferenceEngine::Core ie;
        auto network = ie.ReadNetwork(PRIOR_BOX_CLUSTERED_MODEL, InferenceEngine::Blob::CPtr());

        auto inputsInfo = network.getInputsInfo();
        inputsInfo["data1"]->setPrecision(Precision::FP16);
        inputsInfo["data2"]->setPrecision(Precision::FP16);

        auto outputsInfo = network.getOutputsInfo();
        outputsInfo["data1_copy"]->setPrecision(Precision::FP16);
        outputsInfo["data2_copy"]->setPrecision(Precision::FP16);
        outputsInfo["priorboxclustered_copy"]->setPrecision(Precision::FP16);

        IExecutableNetwork::Ptr exeNetwork;
        ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetwork, network, {}, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(exeNetwork, nullptr) << _resp.msg;

        IInferRequest::Ptr inferRequest;
        ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(inferRequest->GetBlob("priorboxclustered_copy", priorOutput, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }

    TBlob<float>::Ptr refOutput;

    void CalcRefOutput(const bool decrease_label_id) {
        auto locations = make_shared_blob<float>({Precision::FP32, {1, NUM_LOC}, Layout::ANY});
        locations->allocate();
        {
            float *dst = locations->data();
            ie_memcpy(dst, locations->byteSize(), gen_locations.data(), NUM_LOC * sizeof(float));
        }

        auto confidence = make_shared_blob<float>({Precision::FP32, {1, NUM_CONF}, Layout::ANY});
        confidence->allocate();
        {
            float *dst = confidence->data();
            ie_memcpy(dst, confidence->byteSize(), gen_confidence.data(), NUM_CONF * sizeof(float));
        }

        auto prior = make_shared_blob<float>({Precision::FP32, {1, 2, NUM_LOC}, Layout::ANY});
        prior->allocate();
        {
            float *dst = prior->buffer().as<float *>();
            ie_fp16 *src = priorOutput->buffer().as<ie_fp16 *>();
            for (int i = 0; i < 2 * NUM_LOC; ++i) {
                dst[i] = PrecisionUtils::f16tof32(src[i]);
            }
        }

        refOutput = make_shared_blob<float>({Precision::FP32, {7, NUM_OUT, 1, 1}, Layout::ANY});
        refOutput->allocate();

        ASSERT_NO_FATAL_FAILURE(refDetectionOutput(locations, confidence, prior, refOutput, decrease_label_id));
    }

    Blob::Ptr myriadOutput;

    void CheckResults() {
        ASSERT_NE(myriadOutput, nullptr);

        ASSERT_EQ(7 * NUM_OUT, refOutput->size());
        ASSERT_EQ(7 * NUM_OUT, myriadOutput->size());

    #ifdef USE_OPENCV_IMSHOW
        {
            const float *ref_data = refOutput->readOnly();
            const ie_fp16 *actual_data = myriadOutput->cbuffer().as<const ie_fp16 *>();

            cv::Mat ref_img(HEIGHT, WIDTH, CV_8UC1, cv::Scalar::all(0));
            cv::Mat actual_img(HEIGHT, WIDTH, CV_8UC1, cv::Scalar::all(0));

            for (int i = 0; i < NUM_OUT; ++i)
            {
                float xmin, ymin, xmax, ymax;
                cv::Rect rect;

                xmin = ref_data[i * 7 + 3];
                ymin = ref_data[i * 7 + 4];
                xmax = ref_data[i * 7 + 5];
                ymax = ref_data[i * 7 + 6];

                rect.x = WIDTH * xmin;
                rect.y = HEIGHT * ymin;
                rect.width  = WIDTH * (xmax - xmin);
                rect.height = HEIGHT * (ymax - ymin);

                cv::rectangle(ref_img, rect, cv::Scalar::all(255));

                xmin = PrecisionUtils::f16tof32(actual_data[i * 7 + 3]);
                ymin = PrecisionUtils::f16tof32(actual_data[i * 7 + 4]);
                xmax = PrecisionUtils::f16tof32(actual_data[i * 7 + 5]);
                ymax = PrecisionUtils::f16tof32(actual_data[i * 7 + 6]);

                rect.x = WIDTH * xmin;
                rect.y = HEIGHT * ymin;
                rect.width  = WIDTH * (xmax - xmin);
                rect.height = HEIGHT * (ymax - ymin);

                cv::rectangle(actual_img, rect, cv::Scalar::all(255));
            }

            cv::Mat diff;
            cv::absdiff(ref_img, actual_img, diff);

            cv::imshow("ref_img", ref_img);
            cv::imshow("actual_img", actual_img);
            cv::imshow("diff", diff);
            cv::waitKey();
        }
    #endif

        {
            const float *ref_data = refOutput->readOnly();
            const ie_fp16 *actual_data = myriadOutput->cbuffer().as<const ie_fp16 *>();

            std::vector<DetectionObject> ref_objs, actual_objs;
            cvtDetectionObjects(ref_data, NUM_OUT, WIDTH, HEIGHT, ref_objs);
            cvtDetectionObjects(actual_data, NUM_OUT, WIDTH, HEIGHT, actual_objs);

            checkDetectionObjectArrays(ref_objs, actual_objs, MAX_NONMATCH_OBJS);
        }
    }
};

TEST_F(myriadDetectionOutputTests_smoke, NoConst) {
    ASSERT_NO_FATAL_FAILURE(PrepareInput());
    ASSERT_NO_FATAL_FAILURE(CalcRefOutput(false));

    StatusCode st;
    
    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(DETECTION_OUTPUT_MODEL, InferenceEngine::Blob::CPtr());

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["data1"]->setPrecision(Precision::FP16);
    inputsInfo["data2"]->setPrecision(Precision::FP16);
    inputsInfo["locations"]->setPrecision(Precision::FP16);
    inputsInfo["confidence"]->setPrecision(Precision::FP16);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo["data1_copy"]->setPrecision(Precision::FP16);
    outputsInfo["data2_copy"]->setPrecision(Precision::FP16);
    outputsInfo["detection_out"]->setPrecision(Precision::FP16);

    IExecutableNetwork::Ptr exeNetwork;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(exeNetwork, nullptr) << _resp.msg;

    IInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr locations;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("locations", locations, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    {
        ie_fp16 *dst = locations->buffer().as<ie_fp16 *>();
        for (int i = 0; i < NUM_LOC; ++i) {
            dst[i] = PrecisionUtils::f32tof16(gen_locations[i]);
        }
    }

    Blob::Ptr confidence;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("confidence", confidence, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    {
        ie_fp16 *dst = confidence->buffer().as<ie_fp16 *>();
        for (int i = 0; i < NUM_CONF; ++i) {
            dst[i] = PrecisionUtils::f32tof16(gen_confidence[i]);
        }
    }

    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(inferRequest->GetBlob("detection_out", myriadOutput, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    CheckResults();
}

TEST_F(myriadDetectionOutputTests_smoke, MxNet) {
    ASSERT_NO_FATAL_FAILURE(PrepareInput());
    ASSERT_NO_FATAL_FAILURE(CalcRefOutput(true));

    StatusCode st;

    TBlob<uint8_t>::Ptr weights(new TBlob<uint8_t>({Precision::U8, {priorOutput->byteSize()}, Layout::C}, priorOutput->buffer().as<uint8_t *>()));
    
    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(DETECTION_OUTPUT_MODEL_MXNET, weights);

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["locations"]->setPrecision(Precision::FP16);
    inputsInfo["confidence"]->setPrecision(Precision::FP16);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo["detection_out"]->setPrecision(Precision::FP16);

    IExecutableNetwork::Ptr exeNetwork;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(exeNetwork, nullptr) << _resp.msg;

    IInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr locations;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("locations", locations, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    {
        ie_fp16 *dst = locations->buffer().as<ie_fp16 *>();
        for (int i = 0; i < NUM_LOC; ++i) {
            dst[i] = PrecisionUtils::f32tof16(gen_locations[i]);
        }
    }

    Blob::Ptr confidence;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("confidence", confidence, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    {
        ie_fp16 *dst = confidence->buffer().as<ie_fp16 *>();
        for (int i = 0; i < NUM_CONF; ++i) {
            dst[i] = PrecisionUtils::f32tof16(gen_confidence[i]);
        }
    }

    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(inferRequest->GetBlob("detection_out", myriadOutput, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    CheckResults();
}

TEST_F(myriadDetectionOutputTests_smoke, WithConst) {
    ASSERT_NO_FATAL_FAILURE(PrepareInput());
    ASSERT_NO_FATAL_FAILURE(CalcRefOutput(false));

    StatusCode st;

    TBlob<uint8_t>::Ptr weights(new TBlob<uint8_t>({Precision::U8, {priorOutput->byteSize()}, Layout::C}, priorOutput->buffer().as<uint8_t *>()));
   
    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(DETECTION_OUTPUT_MODEL_WITH_CONST, weights);

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["locations"]->setPrecision(Precision::FP16);
    inputsInfo["confidence"]->setPrecision(Precision::FP16);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo["detection_out"]->setPrecision(Precision::FP16);

    IExecutableNetwork::Ptr exeNetwork;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(exeNetwork, nullptr) << _resp.msg;

    IInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr locations;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("locations", locations, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    {
        ie_fp16 *dst = locations->buffer().as<ie_fp16 *>();
        for (int i = 0; i < NUM_LOC; ++i) {
            dst[i] = PrecisionUtils::f32tof16(gen_locations[i]);
        }
    }

    Blob::Ptr confidence;
    ASSERT_NO_THROW(st = inferRequest->GetBlob("confidence", confidence, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    {
        ie_fp16 *dst = confidence->buffer().as<ie_fp16 *>();
        for (int i = 0; i < NUM_CONF; ++i) {
            dst[i] = PrecisionUtils::f32tof16(gen_confidence[i]);
        }
    }

    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(inferRequest->GetBlob("detection_out", myriadOutput, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    CheckResults();
}
