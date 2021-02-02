// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <ngraph_functions/builders.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include "single_layer_tests/detection_output.hpp"
#include "common/bbox_util.h"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace {

const int DEFAULT_SEED_VALUE = 43;
const int WIDTH = 300;
const int HEIGHT = 300;
const size_t NUM_LOC = 12996;
const size_t NUM_CONF = 6498;
const CodeType CODE_TYPE = CENTER_SIZE;
const int NUM_ORIENT_CLASSES = 0;
const int KEEP_TOP_K = 200;
const size_t NUM_OUT = 200;
const int INTERPOLATE_ORIENTATION = 0;

std::vector<std::uint8_t> convertVecFloat2Uint(const std::vector<float>& ref) {
    std::vector<std::uint8_t> u8Data (ref.size() * sizeof(float), 0);
    std::copy_n(reinterpret_cast<const std::uint8_t*>(ref.data()),
                ref.size() * sizeof(float), u8Data.data());

    return u8Data;
}
std::vector<float> convertVecUint2Float(const std::vector<std::uint8_t>& ref) {
    EXPECT_EQ(ref.size() % sizeof(float), 0);

    std::vector<float> floatData (ref.size() / sizeof(float), 0);
    std::copy_n(ref.data(), ref.size(),
                reinterpret_cast<std::uint8_t*>(floatData.data()));

    return floatData;
}

class DetectionOutputTestHelper {
public:
    struct BBox {
        int x, y;
        int width, height;
    };

    struct DetectionObject {
        int batch_id;
        int class_id;
        float confidence;
        BBox bbox;
    };

public:
    void Compare(const std::vector<float>& expected, const InferenceEngine::Blob::Ptr& actual) {
        EXPECT_EQ(7 * NUM_OUT, expected.size());
        EXPECT_EQ(7 * NUM_OUT, actual->size());

        const auto *actual_data = actual->cbuffer().as<const float *>();

        std::vector<DetectionObject> ref_objs, actual_objs;
        cvtDetectionObjects(expected.data(), NUM_OUT, WIDTH, HEIGHT, ref_objs);
        cvtDetectionObjects(actual_data, NUM_OUT, WIDTH, HEIGHT, actual_objs);

        checkDetectionObjectArrays(ref_objs, actual_objs, 0);
    }

    std::vector<float> Ref(const InferenceEngine::Blob::Ptr& locations, const InferenceEngine::Blob::Ptr& confidence,
                           const InferenceEngine::Blob::Ptr& prior, const InferenceEngine::Blob::Ptr& output,
                           const ngraph::op::DetectionOutputAttrs& attrs) {
        std::vector<float> dst_data(7 * NUM_OUT, 0);

        int _num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
        int _num_priors = prior->getTensorDesc().getDims()[2] / 4;

        EXPECT_EQ(_num_priors * _num_loc_classes * 4, locations->getTensorDesc().getDims()[1])
                        << "Number of priors must match number of location predictions";
        EXPECT_EQ(_num_priors * attrs.num_classes, confidence->getTensorDesc().getDims()[1])
                        << "Number of priors must match number of confidence predictions";

        const auto* loc_data = locations->cbuffer().as<float *>();
        const float* orient_data = nullptr; // TODO : support orientation, when myriad will do it
        const auto* conf_data = confidence->cbuffer().as<float *>();
        const auto* prior_data = prior->cbuffer().as<float *>();

        const int num = locations->getTensorDesc().getDims()[0];

        // Retrieve all location predictions.
        std::vector<LabelBBox> all_loc_preds;
        GetLocPredictions(loc_data, num, _num_priors, _num_loc_classes, attrs.share_location, &all_loc_preds);

        // Retrieve all confidences.
        std::vector<std::map<int, std::vector<float>>> all_conf_scores;
        GetConfidenceScores(conf_data, num, _num_priors, attrs.num_classes, &all_conf_scores);

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
                        attrs.share_location, _num_loc_classes, attrs.background_label_id,
                        CODE_TYPE, attrs.variance_encoded_in_target, &all_decode_bboxes);
        // todo string to enum

        int num_kept = 0;

        std::vector<std::map<int, std::vector<int>>> all_indices;

        for (int image_index_in_batch = 0; image_index_in_batch < num; ++image_index_in_batch) {
            if (orient_data) {
                EXPECT_EQ(_num_priors, all_orient_scores[image_index_in_batch].size())
                                << "Orientation scores not equal to num priors.";
            }

            const LabelBBox& decode_bboxes = all_decode_bboxes[image_index_in_batch];
            const std::map<int, std::vector<float>>& conf_scores = all_conf_scores[image_index_in_batch];
            std::map<int, std::vector<int> > indices;
            int num_det = 0;

            for (int label_index = 0; label_index < attrs.num_classes; ++label_index) {
                if (label_index == attrs.background_label_id) {
                    // Ignore background class.
                    continue;
                }

                EXPECT_NE(conf_scores.end(), conf_scores.find(label_index))
                                << "Could not find confidence predictions for label " << label_index;

                const std::vector<float>& scores = conf_scores.find(label_index)->second;

                int label = attrs.share_location ? -1 : label_index;

                if (decode_bboxes.find(label) == decode_bboxes.end()) {
                    // Something bad happened if there are no predictions for current label.
                    continue;
                }

                const std::vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;

                ApplyNMSFast(bboxes, scores, attrs.confidence_threshold, attrs.nms_threshold, attrs.top_k, &(indices[label_index]));
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

                        EXPECT_LT(idx, scores.size())
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

        const int DETECTION_OUTPUT_SIZE = output->getTensorDesc().getDims()[3];
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
            mark_end(dst_data.data());
            return dst_data;
        }

        int count = 0;

        for (int image_index_in_batch = 0; image_index_in_batch < num; ++image_index_in_batch) {
            const std::map<int, std::vector<float>>& conf_scores = all_conf_scores[image_index_in_batch];
            const std::vector<std::vector<float> >* p_orient_scores =  orient_data ? &(all_orient_scores[image_index_in_batch]) : NULL;

            if (orient_data) {
                EXPECT_EQ(_num_priors, p_orient_scores->size())
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
                int loc_label = attrs.share_location ? -1 : label;

                if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
                    // Something bad happened if there are no predictions for current label.
                    continue;
                }

                const std::vector<NormalizedBBox>& bboxes = decode_bboxes.find(loc_label)->second;

                EXPECT_EQ(_num_priors, bboxes.size())
                                << "Bounding boxes num is not equal to num priors";

                std::vector<int>& indices = it->second;

                for (int j = 0; j < indices.size(); ++j) {
                    int idx = indices[j];

                    dst_data[count * DETECTION_OUTPUT_SIZE + 0] = image_index_in_batch;
                    dst_data[count * DETECTION_OUTPUT_SIZE + 1] = attrs.decrease_label_id ? label - 1 : label;
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
            mark_end(dst_data.data() + count * DETECTION_OUTPUT_SIZE);
        }

        return dst_data;
    }

private:
    void cvt(float src, float& dst) {
        dst = src;
    }
    void cvt(float src, ie_fp16& dst) {
        dst = PrecisionUtils::f32tof16(src);
    }
    void cvt(ie_fp16 src, float& dst) {
        dst = PrecisionUtils::f16tof32(src);
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

            out.push_back({static_cast<int>(batch_id), static_cast<int>(class_id), conf, bbox});
        }
    }

    // TODO #-48015
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

    friend std::ostream& operator<<(std::ostream& os, const BBox& bbox) {
        return os << "[" << bbox.x << ", " << bbox.y << ", "
                  << bbox.width << ", " << bbox.height << "]";
    }

    friend std::ostream& operator<<(std::ostream& os, const DetectionObject& obj) {
        return os << "[" << obj.batch_id << ", " << obj.class_id << ", "
                  << obj.confidence << ", " << obj.bbox << "]";
    }
};

class MyriadDetectionOutputLayerTest: public DetectionOutputLayerTest {
private:
    DetectionOutputTestHelper _helper;
    std::vector<float> _genLocations;
    std::vector<float> _genConfidence;
    std::vector<float> _genPrior;

public:
    void Infer() override {
        GenInput();

        inferRequest = executableNetwork.CreateInferRequest();
        inputs.clear();

        size_t it = 0;
        for (const auto &input : cnnNetwork.getInputsInfo()) {
            const auto &info = input.second;
            InferenceEngine::Blob::Ptr blob;
            blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();
            auto *dst = blob->buffer().as<float *>();
            if (it == 0) {
                std::copy_n(_genLocations.begin(), NUM_LOC, dst);
            } else if (it == 1) {
                std::copy_n(_genConfidence.begin(), NUM_CONF, dst);
            } else {
                std::copy_n(_genPrior.begin(), 2 * NUM_LOC, dst);
            }
            inferRequest.SetBlob(info->name(), blob);
            inputs.push_back(blob);
            it++;
        }

        inferRequest.Infer();
    }

    void Compare(const std::vector<std::uint8_t>& expected, const InferenceEngine::Blob::Ptr& actual) override {
        _helper.Compare(convertVecUint2Float(expected), actual);
    }

protected:
    std::vector<std::vector<std::uint8_t>> CalculateRefs() override {
        const auto refFloat = _helper.Ref(inputs[0], inputs[1], inputs[2], GetOutputs()[0], attrs);

        return {convertVecFloat2Uint(refFloat)};
    }

private:
    void GenInput() {
        // TODO #-48015
        srand(DEFAULT_SEED_VALUE);

        _genLocations.resize(NUM_LOC);
        for (size_t i = 0; i < NUM_LOC; i += 4) {
            int x = std::rand() % WIDTH;
            int y = std::rand() % HEIGHT;
            int w = std::rand() % (WIDTH - x);
            int h = std::rand() % (HEIGHT - y);

            _genLocations[i + 0] = static_cast<float>(x) / WIDTH; // xmin
            _genLocations[i + 1] = static_cast<float>(y) / HEIGHT; // ymin
            _genLocations[i + 2] = static_cast<float>(x + w) / WIDTH; // xmax
            _genLocations[i + 3] = static_cast<float>(y + h) / HEIGHT; // ymax
        }

        _genConfidence.resize(NUM_CONF);
        for (size_t i = 0; i < NUM_CONF; ++i) {
            _genConfidence[i] = static_cast<float>(std::rand()) / static_cast <float> (RAND_MAX);
        }

        // hardcoded values from deprecated tests
        ngraph::op::PriorBoxClusteredAttrs attributes;
        attributes.widths = {9.400000, 25.100000, 14.700000, 34.700001, 143.000000, 77.400002, 128.800003, 51.099998, 75.599998};
        attributes.heights = {15.000000, 39.599998, 25.500000, 63.200001, 227.500000, 162.899994, 124.500000, 105.099998, 72.599998};
        attributes.clip = false;
        attributes.step_widths = 0;
        attributes.step_heights = 0;
        attributes.offset = 0.5;
        attributes.variances = {0.100000, 0.100000, 0.200000, 0.200000};

        auto in = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto layer_shape = ngraph::opset3::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{2}, {19, 19});
        auto image_shape = ngraph::opset3::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{2}, {WIDTH, HEIGHT});
        auto pb = std::make_shared<ngraph::opset3::PriorBoxClustered>(layer_shape, image_shape, attributes);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(pb)};
        auto pbFunction = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{in}, "PB");

        ngraph::pass::InitNodeInfo().run_on_function(pbFunction);
        ngraph::pass::ConstantFolding().run_on_function(pbFunction);

        auto fused = std::dynamic_pointer_cast<ngraph::opset3::Constant>(pbFunction->get_result()->input_value(0).get_node_shared_ptr());
        _genPrior = fused->get_vector<float>();
    }
};

const int numClasses = 2;
const int backgroundLabelId = 0;
const std::vector<int> topK = {400};
const std::vector<std::vector<int>> keepTopK = { {KEEP_TOP_K} };
const std::vector<std::string> codeType = {"caffe.PriorBoxParameter.CENTER_SIZE"};
const float nmsThreshold = 0.45f;
const float confidenceThreshold = 0.01f;
const std::vector<bool> clipAfterNms = {false};
const std::vector<bool> clipBeforeNms = {false};
const std::vector<bool> decreaseLabelId = {true, false};
const float objectnessScore = 0.0f;
const std::vector<size_t> numberBatch = {1};

const auto commonAttributes = ::testing::Combine(
        ::testing::Values(numClasses),
        ::testing::Values(backgroundLabelId),
        ::testing::ValuesIn(topK),
        ::testing::ValuesIn(keepTopK),
        ::testing::ValuesIn(codeType),
        ::testing::Values(nmsThreshold),
        ::testing::Values(confidenceThreshold),
        ::testing::ValuesIn(clipAfterNms),
        ::testing::ValuesIn(clipBeforeNms),
        ::testing::ValuesIn(decreaseLabelId)
);

/* =============== 3 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams3In = {
    ParamsWhichSizeDepends{false, true, true, WIDTH, WIDTH, {1, NUM_LOC}, {1, NUM_CONF}, {1, 2, NUM_LOC}, {}, {}},
};

const auto params3Inputs = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams3In),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(objectnessScore),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);

TEST_P(MyriadDetectionOutputLayerTest, CompareWithRefs) {
    Run();
};

INSTANTIATE_TEST_CASE_P(smoke_DetectionOutput3In, MyriadDetectionOutputLayerTest, params3Inputs, DetectionOutputLayerTest::getTestCaseName);

}  // namespace
