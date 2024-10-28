// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output_inst.h"
#include "impls/registry/implementation_map.hpp"
#include "register.hpp"
#include "cpu_impl_helpers.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

#ifdef HAVE_SSE
#include <immintrin.h>
#include <xmmintrin.h>
#endif // HAVE_SSE

namespace cldnn {
namespace cpu {

template <typename T>
bool comp_score_descend(const std::pair<float, T>& pair1,
                        const std::pair<float, T>& pair2) {
    return (pair1.first > pair2.first) || (pair1.first == pair2.first && pair1.second < pair2.second);
}

template <>
bool comp_score_descend<std::pair<int, int>>(const std::pair<float, std::pair<int, int>>& pair1,
                                             const std::pair<float, std::pair<int, int>>& pair2) {
    return (pair1.first > pair2.first) || (pair1.first == pair2.first && pair1.second.second < pair2.second.second);
}

/************************ Detection Output CPU ************************/
struct detection_output_impl : typed_primitive_impl<detection_output> {
    using parent = typed_primitive_impl<detection_output>;
    using parent::parent;

public:
    enum NMSType {CAFFE, MXNET};
    NMSType nms_type = NMSType::CAFFE;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::detection_output_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<detection_output_impl>(*this);
    }

    detection_output_impl() : parent() {}

    explicit detection_output_impl(const detection_output_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<detection_output>());
        const auto& node = arg.as<detection_output>();
        nms_type = (node.get_primitive()->decrease_label_id ? NMSType::MXNET : NMSType::CAFFE);
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << make_data(&nms_type, sizeof(NMSType));
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> make_data(&nms_type, sizeof(NMSType));
    }

    static inline void intersect_bbox(const bounding_box& bbox1,
                                      const bounding_box& bbox2,
                                      bounding_box& intersect_bbox) {
        if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
            bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
            intersect_bbox.xmin = 0;
            intersect_bbox.ymin = 0;
            intersect_bbox.xmax = 0;
            intersect_bbox.ymax = 0;
        } else {
            intersect_bbox.xmin = std::max<float>(bbox1.xmin, bbox2.xmin);
            intersect_bbox.ymin = std::max<float>(bbox1.ymin, bbox2.ymin);
            intersect_bbox.xmax = std::min<float>(bbox1.xmax, bbox2.xmax);
            intersect_bbox.ymax = std::min<float>(bbox1.ymax, bbox2.ymax);
        }
    }

    static float jaccard_overlap(const bounding_box& bbox1, const bounding_box& bbox2) {
        bounding_box inter_bbox;
        intersect_bbox(bbox1, bbox2, inter_bbox);

        float intersectWidth, intersectHeight;
        intersectWidth = inter_bbox.xmax - inter_bbox.xmin;
        intersectHeight = inter_bbox.ymax - inter_bbox.ymin;
        if (intersectWidth > 0 && intersectHeight > 0) {
            float intersect_size = intersectWidth * intersectHeight;
            float bbox1_size = bbox1.area();
            float bbox2_size = bbox2.area();
            return intersect_size / (bbox1_size + bbox2_size - intersect_size);
        } else {
            return 0.0f;
        }
    }

    static void decode_bounding_box(const bounding_box& prior_bbox,
                                    const std::array<float, PRIOR_BOX_SIZE>& prior_variance,
                                    const prior_box_code_type code_type,
                                    const bool variance_encoded_in_target,
                                    const bounding_box& bbox,
                                    bounding_box* decoded_bbox,
                                    const bool prior_is_normalized,
                                    const size_t image_width,
                                    const size_t image_height,
                                    const bool clip_before_nms) {
        float prior_bbox_xmin = prior_bbox.xmin;
        float prior_bbox_ymin = prior_bbox.ymin;
        float prior_bbox_xmax = prior_bbox.xmax;
        float prior_bbox_ymax = prior_bbox.ymax;

        float bbox_xmin = bbox.xmin;
        float bbox_ymin = bbox.ymin;
        float bbox_xmax = bbox.xmax;
        float bbox_ymax = bbox.ymax;

        if (!prior_is_normalized) {
            prior_bbox_xmin /= image_width;
            prior_bbox_ymin /= image_height;
            prior_bbox_xmax /= image_width;
            prior_bbox_ymax /= image_height;
        }

        switch (code_type) {
            case prior_box_code_type::corner: {
                if (variance_encoded_in_target) {
                    // variance is encoded in target, we simply need to add the offset predictions.
                    decoded_bbox->xmin = prior_bbox_xmin + bbox_xmin;
                    decoded_bbox->ymin = prior_bbox_ymin + bbox_ymin;
                    decoded_bbox->xmax = prior_bbox_xmax + bbox_xmax;
                    decoded_bbox->ymax = prior_bbox_ymax + bbox_ymax;
                } else {
                    // variance is encoded in bbox, we need to scale the offset accordingly.
                    decoded_bbox->xmin = prior_bbox_xmin + prior_variance[0] * bbox_xmin;
                    decoded_bbox->ymin = prior_bbox_ymin + prior_variance[1] * bbox_ymin;
                    decoded_bbox->xmax = prior_bbox_xmax + prior_variance[2] * bbox_xmax;
                    decoded_bbox->ymax = prior_bbox_ymax + prior_variance[3] * bbox_ymax;
                }
                break;
            }
            case prior_box_code_type::center_size: {
                const float prior_width = prior_bbox_xmax - prior_bbox_xmin;
                const float prior_height = prior_bbox_ymax - prior_bbox_ymin;
                const float prior_center_x = (prior_bbox_xmin + prior_bbox_xmax) / 2.f;
                const float prior_center_y = (prior_bbox_ymin + prior_bbox_ymax) / 2.f;
                float decode_bbox_center_x, decode_bbox_center_y;
                float decode_bbox_width, decode_bbox_height;
                if (variance_encoded_in_target) {
                    // variance is encoded in target, we simply need to restore the offset predictions.
                    decode_bbox_center_x = bbox_xmin * prior_width + prior_center_x;
                    decode_bbox_center_y = bbox_ymin * prior_height + prior_center_y;
                    decode_bbox_width = (exp(bbox_xmax) * prior_width);
                    decode_bbox_height = (exp(bbox_ymax) * prior_height);
                } else {
                    // variance is encoded in bbox, we need to scale the offset accordingly.
                    decode_bbox_center_x = prior_variance[0] * bbox_xmin * prior_width + prior_center_x;
                    decode_bbox_center_y = prior_variance[1] * bbox_ymin * prior_height + prior_center_y;
                    decode_bbox_width = (exp(prior_variance[2] * bbox_xmax) * prior_width);
                    decode_bbox_height = (exp(prior_variance[3] * bbox_ymax) * prior_height);
                }
                decoded_bbox->xmin = decode_bbox_center_x - decode_bbox_width / 2.0f;
                decoded_bbox->ymin = decode_bbox_center_y - decode_bbox_height / 2.0f;
                decoded_bbox->xmax = decode_bbox_center_x + decode_bbox_width / 2.0f;
                decoded_bbox->ymax = decode_bbox_center_y + decode_bbox_height / 2.0f;
                break;
            }
            case prior_box_code_type::corner_size: {
                const float prior_width = prior_bbox_xmax - prior_bbox_xmin;
                assert(prior_width > 0);
                const float prior_height = prior_bbox_ymax - prior_bbox_ymin;
                assert(prior_height > 0);
                if (variance_encoded_in_target) {
                    // variance is encoded in target, we simply need to add the offset predictions.
                    decoded_bbox->xmin = prior_bbox_xmin + bbox_xmin * prior_width;
                    decoded_bbox->ymin = prior_bbox_ymin + bbox_ymin * prior_height;
                    decoded_bbox->xmax = prior_bbox_xmax + bbox_xmax * prior_width;
                    decoded_bbox->ymax = prior_bbox_ymax + bbox_ymax * prior_height;
                } else {
                    // variance is encoded in bbox, we need to scale the offset accordingly.
                    decoded_bbox->xmin = prior_bbox_xmin + prior_variance[0] * bbox_xmin * prior_width;
                    decoded_bbox->ymin = prior_bbox_ymin + prior_variance[1] * bbox_ymin * prior_height;
                    decoded_bbox->xmax = prior_bbox_xmax + prior_variance[2] * bbox_xmax * prior_width;
                    decoded_bbox->ymax = prior_bbox_ymax + prior_variance[3] * bbox_ymax * prior_height;
                }
                break;
            }
            default: {
                assert(0);
            }
        }

        if (clip_before_nms) {
            decoded_bbox->xmin = std::max(0.0f, std::min(1.0f, decoded_bbox->xmin));
            decoded_bbox->ymin = std::max(0.0f, std::min(1.0f, decoded_bbox->ymin));
            decoded_bbox->xmax = std::max(0.0f, std::min(1.0f, decoded_bbox->xmax));
            decoded_bbox->ymax = std::max(0.0f, std::min(1.0f, decoded_bbox->ymax));
        }
    }

    void mxnet_nms(const std::vector<std::vector<bounding_box>>& bboxes,
                   const float nms_threshold,
                   const int top_k,
                   const bool share_location,
                   std::map<int, std::vector<int>>& indices,
                   std::vector<std::pair<float, std::pair<int, int>>>& scoreIndexPairs) {
        std::sort(scoreIndexPairs.begin(),
                  scoreIndexPairs.end(),
                  comp_score_descend<std::pair<int, int>>);

        if (top_k != -1)
            if (scoreIndexPairs.size() > static_cast<size_t>(top_k))
                scoreIndexPairs.resize(top_k);
        while (scoreIndexPairs.size() != 0) {
            const int cls = scoreIndexPairs.front().second.first;
            const int prior = scoreIndexPairs.front().second.second;
            std::vector<int>& currInd = indices[cls];
            bool keep = true;
            for (size_t i = 0; i < currInd.size(); i++) {
                const int keptIdx = currInd[i];
                const auto& currBbox = share_location ? bboxes[0] : bboxes[cls];
                float overlap = jaccard_overlap(currBbox[prior], currBbox[keptIdx]);
                if (overlap > nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                currInd.push_back(prior);
            }
            scoreIndexPairs.erase(scoreIndexPairs.begin());
        }
    }

    static void caffe_nms(const std::vector<bounding_box>& bboxes,
                          std::vector<std::pair<float, int>>& scores,
                          const float nms_threshold,
                          const int top_k,
                          std::vector<int>& indices) {
        if (top_k > -1 && static_cast<size_t>(top_k) < static_cast<size_t>(scores.size())) {
            std::partial_sort(scores.begin(),
                              scores.begin() + top_k,
                              scores.end(),
                              comp_score_descend<int>);
            scores.resize(top_k);
        } else {
            std::stable_sort(scores.begin(), scores.end(), comp_score_descend<int>);
        }
        // NMS
        for (const auto& s : scores) {
            const int idx = s.second;
            bool keep = true;
            for (int k = 0; k < static_cast<int>(indices.size()); ++k) {
                const int kept_idx = indices[k];
                float overlap = jaccard_overlap(bboxes[idx], bboxes[kept_idx]);
                if (overlap > nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                indices.push_back(idx);
            }
        }
    }

    template <typename dtype>
    void generate_detections(stream& stream, const detection_output_inst& instance,
                             const int num_of_images,
                             const std::vector<std::vector<std::vector<bounding_box>>>& all_bboxes,
                             std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences,
                             std::vector<std::vector<std::pair<float, std::pair<int, int>>>>& scoreIndexPairs) {
        mem_lock<dtype, mem_lock_type::write> lock{instance.output_memory_ptr(), stream};
        auto out_ptr = lock.begin();

        const auto& args = instance.argument;

        auto confidence_layout = instance.confidence_memory()->get_layout();
        auto priors_layout = instance.prior_box_memory()->get_layout();

        const int num_of_priors = priors_layout.spatial(1) / args->prior_info_size;
        const int num_classes = (args->num_classes == -1) ? confidence_layout.feature() / num_of_priors : args->num_classes;
        // Per image -> For each label: Pair (score, prior index)
        std::vector<std::map<int, std::vector<std::pair<float, int>>>> final_detections;
        for (int image = 0; image < num_of_images; ++image) {
            const std::vector<std::vector<bounding_box>>& bboxes_per_image = all_bboxes[image];
            std::vector<std::vector<std::pair<float, int>>>& conf_per_image = confidences[image];
            std::map<int, std::vector<int>> indices;
            int num_det = 0;
            if (nms_type == NMSType::CAFFE) {
                for (int cls = 0; cls < num_classes; ++cls) {
                    if (static_cast<int>(cls) == args->background_label_id) {
                        conf_per_image[cls].clear();
                        continue;  // Skip background class.
                    }
                    std::vector<std::pair<float, int>>& scores = conf_per_image[cls];
                    const int label = args->share_location ? 0 : cls;
                    caffe_nms(bboxes_per_image[label], scores, args->nms_threshold, args->top_k, indices[cls]);
                    num_det += static_cast<int>(indices[cls].size());
                }
            } else {
                std::vector<std::pair<float, std::pair<int, int>>>& score_image = scoreIndexPairs[image];
                mxnet_nms(bboxes_per_image, args->nms_threshold, args->top_k, args->share_location, indices, score_image);
                for (auto it = indices.begin(); it != indices.end(); it++) {
                    num_det += static_cast<int>(it->second.size());
                }
            }

            if (args->keep_top_k > -1 && num_det > args->keep_top_k) {
                std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
                for (auto it = indices.begin(); it != indices.end(); ++it) {
                    int label = it->first;
                    const std::vector<int>& labelIndices = it->second;
                    std::vector<std::pair<float, int>>& scores = confidences[image][label];
                    for (int j = 0; j < static_cast<int>(labelIndices.size()); ++j) {
                        int idx = labelIndices[j];
                        for (const auto& s : scores) {
                            if (s.second == idx) {
                                score_index_pairs.push_back(std::make_pair(s.first, std::make_pair(label, idx)));
                            }
                        }
                    }
                }

                std::sort(score_index_pairs.begin(),
                          score_index_pairs.end(),
                          comp_score_descend<std::pair<int, int>>);
                score_index_pairs.resize(args->keep_top_k);

                std::map<int, std::vector<std::pair<float, int>>> new_indices;
                for (int j = 0; j < static_cast<int>(score_index_pairs.size()); ++j) {
                    int label = score_index_pairs[j].second.first;
                    int idx = score_index_pairs[j].second.second;
                    new_indices[label].push_back(std::make_pair(score_index_pairs[j].first, idx));
                }
                final_detections.push_back(new_indices);
            } else {
                std::map<int, std::vector<std::pair<float, int>>> new_indices;
                for (auto it = indices.begin(); it != indices.end(); ++it) {
                    int label = it->first;
                    const std::vector<int>& labelIndices = it->second;
                    std::vector<std::pair<float, int>>& scores = confidences[image][label];
                    for (int j = 0; j < static_cast<int>(labelIndices.size()); ++j) {
                        int idx = labelIndices[j];
                        for (const auto& s : scores) {
                            if (s.second == idx) {
                                new_indices[label].push_back(std::make_pair(s.first, idx));
                            }
                        }
                    }
                }
                final_detections.push_back(new_indices);
            }
        }

        int count = 0;
        for (int image = 0; image < num_of_images; ++image) {
            const std::vector<std::vector<bounding_box>>& bboxes_per_image = all_bboxes[image];
            for (auto it = final_detections[image].begin(); it != final_detections[image].end(); ++it) {
                int label = it->first;
                int loc_label = args->share_location ? 0 : label;
                const std::vector<bounding_box>& bboxes = bboxes_per_image[loc_label];
                std::vector<std::pair<float, int>>& label_detections = it->second;
                for (std::pair<float, int> score_prior : label_detections) {
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE] = (dtype)static_cast<float>(image);
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 1] =
                        args->decrease_label_id ? ((dtype)(static_cast<float>(label - 1.0f))) : (dtype)static_cast<float>(label);
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 2] = (dtype)score_prior.first;
                    const bounding_box& bbox = bboxes[score_prior.second];
                    float xmin = bbox.xmin;
                    float ymin = bbox.ymin;
                    float xmax = bbox.xmax;
                    float ymax = bbox.ymax;

                    if (args->clip_after_nms) {
                        xmin = std::max(0.0f, std::min(1.0f, xmin));
                        ymin = std::max(0.0f, std::min(1.0f, ymin));
                        xmax = std::max(0.0f, std::min(1.0f, xmax));
                        ymax = std::max(0.0f, std::min(1.0f, ymax));
                    }

                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 3] = (dtype)xmin;
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 4] = (dtype)ymin;
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 5] = (dtype)xmax;
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 6] = (dtype)ymax;
                    ++count;
                }
            }
        }
        const int final_cnt = count;
        for (int i = count; i < num_of_images * args->keep_top_k; i++) {
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE] = (i == final_cnt ? (dtype)-1.f : (dtype)0.f);
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 1] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 2] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 3] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 4] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 5] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 6] = (dtype)0.f;
            ++count;
        }
    }

    // Compute the linear index taking the padding into account.
    static inline int get_linear_feature_index(const int batch_id,
                                               const int feature_id,
                                               const int input_buffer_size_f,
                                               const int input_buffer_size_y,
                                               const int input_buffer_size_x,
                                               const int input_padding_lower_y,
                                               const int input_padding_lower_x) {
        // This helper function assumes input layout with x_size = 1 and y_size = 1;
        // Location and confidence inputs should be tensors with size {b,f,1,1}.
        // This is validated in detection output primitive instance creation.

        int input_idx = (batch_id * input_buffer_size_f + feature_id) * input_buffer_size_y * input_buffer_size_x;
        input_idx += input_padding_lower_y * input_buffer_size_x + input_padding_lower_x;

        return input_idx;
    }

    template <typename dtype>
    void extract_locations_per_image(stream& stream, const detection_output_inst& instance,
                                     std::vector<std::vector<std::vector<bounding_box>>>& locations,
                                     const int num_of_priors,
                                     const int num_loc_classes) {
        const bool share_location = instance.argument->share_location;
        auto input_location = instance.location_memory();
        auto location_layout = input_location->get_layout();
        const int num_of_images = static_cast<int>(locations.size());
        mem_lock<dtype, mem_lock_type::read> lock{input_location, stream};
        auto location_data = lock.begin();
        assert(num_of_priors * num_loc_classes * PRIOR_BOX_SIZE == input_location->get_layout().feature());

        const auto& input_buffer_size = location_layout.get_padded_dims();
        const int input_buffer_size_x = input_buffer_size[3];
        const int input_buffer_size_y = input_buffer_size[2];
        const int input_buffer_size_f = input_buffer_size[1];
        const auto& input_padding = location_layout.data_padding;
        const int input_padding_lower_x = input_padding._lower_size[2];
        const int input_padding_lower_y = input_padding._lower_size[3];

        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<bounding_box>>& label_to_bbox = locations[image];
            label_to_bbox.resize(num_loc_classes);
            for (int cls = 0; cls < num_loc_classes; ++cls) {
                int label = share_location ? 0 : cls;
                auto& bboxes = label_to_bbox[label];
                bboxes.resize(num_of_priors);
                for (int prior = 0; prior < num_of_priors; ++prior) {
                    int idx = prior * num_loc_classes * PRIOR_BOX_SIZE;
                    bboxes[prior].xmin = static_cast<float>((location_data[get_linear_feature_index(image,
                                                                                        idx + cls * PRIOR_BOX_SIZE,
                                                                                        input_buffer_size_f,
                                                                                        input_buffer_size_y,
                                                                                        input_buffer_size_x,
                                                                                        input_padding_lower_y,
                                                                                        input_padding_lower_x)]));
                    bboxes[prior].ymin = static_cast<float>((location_data[get_linear_feature_index(image,
                                                                                        idx + cls * PRIOR_BOX_SIZE + 1,
                                                                                        input_buffer_size_f,
                                                                                        input_buffer_size_y,
                                                                                        input_buffer_size_x,
                                                                                        input_padding_lower_y,
                                                                                        input_padding_lower_x)]));
                    bboxes[prior].xmax = static_cast<float>((location_data[get_linear_feature_index(image,
                                                                                        idx + cls * PRIOR_BOX_SIZE + 2,
                                                                                        input_buffer_size_f,
                                                                                        input_buffer_size_y,
                                                                                        input_buffer_size_x,
                                                                                        input_padding_lower_y,
                                                                                        input_padding_lower_x)]));
                    bboxes[prior].ymax = static_cast<float>((location_data[get_linear_feature_index(image,
                                                                                        idx + cls * PRIOR_BOX_SIZE + 3,
                                                                                        input_buffer_size_f,
                                                                                        input_buffer_size_y,
                                                                                        input_buffer_size_x,
                                                                                        input_padding_lower_y,
                                                                                        input_padding_lower_x)]));
                }
            }
        }
    }

    template <typename dtype>
    void extract_prior_boxes_and_variances(stream& stream, const detection_output_inst& instance,
                                           const bool variance_encoded_in_target,
                                           const int32_t prior_info_size,
                                           const int32_t prior_coordinates_offset,
                                           const int32_t images_count,
                                           std::vector<bounding_box>& prior_bboxes,
                                           std::vector<std::array<float, PRIOR_BOX_SIZE>>& prior_variances) {
        auto input_prior_box = instance.prior_box_memory();
        const int num_of_priors = static_cast<int>(prior_bboxes.size()) / images_count;
        mem_lock<dtype, mem_lock_type::read> lock{std::move(input_prior_box), stream};
        for (int i = 0; i < images_count; i++) {
            auto prior_box_data =
                lock.begin() + i * num_of_priors * prior_info_size * (variance_encoded_in_target ? 1 : 2);

            for (int prior = 0; prior < num_of_priors; ++prior) {
                int idx = prior * prior_info_size + prior_coordinates_offset;
                prior_bboxes[i * num_of_priors + prior] = bounding_box(static_cast<float>(prior_box_data[idx]),
                                                                       static_cast<float>(prior_box_data[idx + 1]),
                                                                       static_cast<float>(prior_box_data[idx + 2]),
                                                                       static_cast<float>(prior_box_data[idx + 3]));
                idx += num_of_priors * prior_info_size;
            }
            if (!variance_encoded_in_target) {
                for (int prior = 0; prior < num_of_priors; ++prior) {
                    int start_idx = prior * 4;
                    std::array<float, PRIOR_BOX_SIZE> var = {0.f, 0.f, 0.f, 0.f};
                    for (int j = 0; j < PRIOR_BOX_SIZE; ++j) {
                        var[j] = (prior_box_data[start_idx + j + num_of_priors * prior_info_size]);
                    }
                    prior_variances[i * num_of_priors + prior] = var;
                }
            }
        }
    }

    template <typename dtype>
    void extract_confidences_per_image_caffe(stream& stream, const detection_output_inst& instance,
                                             std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences,
                                             const int num_of_priors, const int num_classes) {
        const int num_of_images = static_cast<int>(confidences.size());
        auto input_confidence = instance.confidence_memory();
        const float confidence_threshold = instance.argument->confidence_threshold;

        mem_lock<dtype, mem_lock_type::read> lock{input_confidence, stream};
        auto confidence_data = lock.begin();

        assert(num_of_priors * num_classes == input_confidence->get_layout().feature());

        const auto& input_buffer_layout = input_confidence->get_layout();
        const int input_buffer_size_x = input_buffer_layout.spatial(0);
        const int input_buffer_size_y = input_buffer_layout.spatial(1);
        const int input_buffer_size_f = input_buffer_layout.feature();
        const auto& input_padding = input_confidence->get_layout().data_padding;
        const int input_padding_lower_x = input_padding._lower_size[2];
        const int input_padding_lower_y = input_padding._lower_size[3];
        const int stride = input_buffer_size_y * input_buffer_size_x;

        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<std::pair<float, int>>>& label_to_scores = confidences[image];
            std::vector<std::pair<float, std::pair<int, int>>> score_index_per_prior;
            label_to_scores.resize(num_classes);
            int idx = get_linear_feature_index(image,
                                               0,
                                               input_buffer_size_f,
                                               input_buffer_size_y,
                                               input_buffer_size_x,
                                               input_padding_lower_y,
                                               input_padding_lower_x);
            if (stride == 1 && std::is_same<dtype, float>::value) {
                float const* confidence_ptr_float = (float const*)(&(*confidence_data));
                confidence_ptr_float += idx;
#ifdef HAVE_SSE
                __m128 threshold = _mm_load_ps1(&confidence_threshold);
#endif // HAVE_SSE
                for (int prior = 0; prior < num_of_priors; ++prior) {
                    int cls = 0;
#ifdef HAVE_SSE
                    for (; cls + 3 < num_classes; cls += 4) {
                        __m128 scores = _mm_loadu_ps(confidence_ptr_float);
                        confidence_ptr_float += 4;
                        __m128i mask128 = _mm_castps_si128(_mm_cmpgt_ps(scores, threshold));
                        if (_mm_testz_si128(mask128, mask128)) {
                            continue;
                        }
                        int mask = _mm_movemask_ps(_mm_castsi128_ps(mask128));
                        if (mask & 1) {
                            label_to_scores[cls + 0].emplace_back(_mm_cvtss_f32(scores), prior);
                        }
                        if (mask & 2) {
                            int score = _mm_extract_ps(scores, 1);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 1].emplace_back(s, prior);
                        }
                        if (mask & 4) {
                            int score = _mm_extract_ps(scores, 2);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 2].emplace_back(s, prior);
                        }
                        if (mask & 8) {
                            int score = _mm_extract_ps(scores, 3);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 3].emplace_back(s, prior);
                        }
                    }
#endif // HAVE_SSE
                    for (; cls < num_classes; ++cls) {
                        float score = *confidence_ptr_float;
                        if (score > confidence_threshold) {
                            label_to_scores[cls].emplace_back(score, prior);
                        }
                        ++confidence_ptr_float;
                    }
                }
            } else {
                for (int prior = 0; prior < num_of_priors; ++prior) {
                    for (int cls = 0; cls < num_classes; ++cls) {
                        float score = static_cast<float>(confidence_data[idx]);
                        if (score > confidence_threshold) {
                            label_to_scores[cls].emplace_back(score, prior);
                        }
                        idx += stride;
                    }
                }
            }
        }
    }

    template <typename dtype>
    void extract_confidences_per_image_mxnet(stream& stream, const detection_output_inst& instance,
                                             std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences,
                                             const int num_of_priors, const int num_classes,
                                             std::vector<std::vector<std::pair<float, std::pair<int, int>>>>& scoreIndexPairs) {
        const int background_label_id = instance.argument->background_label_id;
        const int num_of_images = static_cast<int>(confidences.size());
        auto input_confidence = instance.confidence_memory();
        const float confidence_threshold = instance.argument->confidence_threshold;
        auto confidence_layout = input_confidence->get_layout();

        mem_lock<dtype, mem_lock_type::read> lock{input_confidence, stream};
        auto confidence_data = lock.begin();

        assert(num_of_priors * num_classes == confidence_layout.feature());

        const int input_buffer_size_x = confidence_layout.spatial(0);
        const int input_buffer_size_y = confidence_layout.spatial(1);
        const int input_buffer_size_f = confidence_layout.feature();
        const auto& input_padding = confidence_layout.data_padding;
        const int input_padding_lower_x = input_padding._lower_size[2];
        const int input_padding_lower_y = input_padding._lower_size[3];
        const int stride = input_buffer_size_y * input_buffer_size_x;

        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<std::pair<float, int>>>& label_to_scores = confidences[image];
            std::vector<std::pair<float, std::pair<int, int>>> score_index_per_prior;
            label_to_scores.resize(num_classes);
            int idx = get_linear_feature_index(image,
                                               0,
                                               input_buffer_size_f,
                                               input_buffer_size_y,
                                               input_buffer_size_x,
                                               input_padding_lower_y,
                                               input_padding_lower_x);
            if (stride == 1 && std::is_same<dtype, float>::value) {
                float const* confidence_ptr_float = (float const*)(&(*confidence_data));
                confidence_ptr_float += idx;
#ifdef HAVE_SSE
                __m128 threshold = _mm_load_ps1(&confidence_threshold);
#endif // HAVE_SSE
                for (int prior = 0; prior < num_of_priors; ++prior) {
                    int idx_start = (background_label_id == 0 ? 1 : 0);
                    int cls = idx_start;
                    float max_score = 0;
                    int max_cls = 0;
#ifdef HAVE_SSE
                    for (; cls + 3 < num_classes; cls += 4) {
                        if ((background_label_id == 0) && (cls == idx_start)) {
                            confidence_ptr_float += 1;
                        }
                        __m128 scores = _mm_loadu_ps(confidence_ptr_float);
                        confidence_ptr_float += 4;
                        __m128i mask128 = _mm_castps_si128(_mm_cmpgt_ps(scores, threshold));
                        if (_mm_testz_si128(mask128, mask128)) {
                            continue;
                        }
                        int mask = _mm_movemask_ps(_mm_castsi128_ps(mask128));
                        if (mask & 1) {
                            float s = _mm_cvtss_f32(scores);
                            label_to_scores[cls + 0].emplace_back(s, prior);
                            if ((cls == idx_start) || (s > max_score)) {
                                max_score = s; max_cls = cls + 0;
                            }
                        }
                        if (mask & 2) {
                            int score = _mm_extract_ps(scores, 1);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 1].emplace_back(s, prior);
                            if (s > max_score) {
                                max_score = s; max_cls = cls + 1;
                            }
                        }
                        if (mask & 4) {
                            int score = _mm_extract_ps(scores, 2);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 2].emplace_back(s, prior);
                            if (s > max_score) {
                                max_score = s; max_cls = cls + 2;
                            }
                        }
                        if (mask & 8) {
                            int score = _mm_extract_ps(scores, 3);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 3].emplace_back(s, prior);
                            if (s > max_score) {
                                max_score = s; max_cls = cls + 3;
                            }
                        }
                    }
#endif // HAVE_SSE
                    for (; cls < num_classes; ++cls) {
                        float score = *confidence_ptr_float;
                        if (score > confidence_threshold) {
                            label_to_scores[cls].emplace_back(score, prior);
                            if (score > max_score) {
                                max_score = score;  max_cls = cls;
                            }
                        }
                        ++confidence_ptr_float;
                    }
                    score_index_per_prior.emplace_back(std::make_pair(max_score, std::make_pair(max_cls, prior)));
                }
                scoreIndexPairs.push_back(score_index_per_prior);
            } else {
                for (int prior = 0; prior < num_of_priors; ++prior) {
                    int idx_start = (background_label_id == 0 ? 1 : 0);
                    float max_score = 0;
                    int max_cls = 0;
                    for (int cls = idx_start; cls < num_classes; ++cls) {
                        float score = static_cast<float>(confidence_data[idx]);
                        if (score > confidence_threshold) {
                            label_to_scores[cls].emplace_back(score, prior);
                            if ((cls == idx_start) || score > max_score) {
                                max_score = score; max_cls = cls;
                            }
                        }
                        idx += stride;
                    }
                    score_index_per_prior.emplace_back(std::make_pair(max_score, std::make_pair(max_cls, prior)));
                }
                scoreIndexPairs.push_back(score_index_per_prior);
            }
        }
    }

    template <typename dtype>
    void prepare_data(stream& stream, const detection_output_inst& instance,
                      std::vector<std::vector<std::vector<bounding_box>>>& bboxes,
                      std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences,
                      std::vector<std::vector<std::pair<float, std::pair<int, int>>>>& scoreIndexPairs) {
        assert(bboxes.size() == confidences.size());

        const auto& args = instance.argument;

        auto confidence_layout = instance.confidence_memory()->get_layout();
        auto priors_layout = instance.prior_box_memory()->get_layout();

        const int num_of_images = static_cast<int>(bboxes.size());
        const int num_of_priors = priors_layout.spatial(1) / args->prior_info_size;
        const int num_classes = (args->num_classes == -1) ? confidence_layout.feature() / num_of_priors : args->num_classes;
        const int num_loc_classes = args->share_location ? 1 : num_classes;

        // Extract locations per image.
        std::vector<std::vector<std::vector<bounding_box>>> locations(
            num_of_images);  // Per image : label -> bounding boxes.
        extract_locations_per_image<dtype>(stream, instance, locations, num_of_priors, num_loc_classes);

        int32_t batches_in_prior_boxes = priors_layout.batch();
        std::vector<bounding_box> prior_bboxes(batches_in_prior_boxes *
                                               num_of_priors);  // Prior-Boxes (identical for all images since we assume
                                                                // all images in a batch are of same dimension).
        std::vector<std::array<float, PRIOR_BOX_SIZE>> prior_variances(
            batches_in_prior_boxes * num_of_priors);  // Variances per prior-box (identical for all images since we
                                                      // assume all images in a batch are of same dimension).
        extract_prior_boxes_and_variances<dtype>(stream,
                                                 instance,
                                                 args->variance_encoded_in_target,
                                                 args->prior_info_size,
                                                 args->prior_coordinates_offset,
                                                 batches_in_prior_boxes,
                                                 prior_bboxes,
                                                 prior_variances);

        // Create the decoded bounding boxes according to locations predictions and prior-boxes.
        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<bounding_box>>& bboxes_per_image = bboxes[image];
            bboxes_per_image.resize(num_loc_classes);
            locations[image].resize(num_loc_classes);

            for (int cls = 0; cls < num_loc_classes; ++cls) {
                const int label = args->share_location ? 0 : cls;
                if (!args->share_location && label == args->background_label_id) {
                    continue;  // Skip background class.
                }
                const std::vector<bounding_box>& label_loc_preds = locations[image][label];
                int label_loc_preds_size = static_cast<int>(label_loc_preds.size());
                bboxes_per_image[label].clear();

                for (int i = 0; i < label_loc_preds_size; ++i) {
                    bounding_box decoded_bbox;
                    int32_t pb_offset = (batches_in_prior_boxes > 1) ? (image * num_of_priors + i) : i;
                    int32_t var_offset = (batches_in_prior_boxes > 1) ? (image * num_of_priors + i) : i;
                    decode_bounding_box(prior_bboxes[pb_offset],
                                        prior_variances[var_offset],
                                        args->code_type,
                                        args->variance_encoded_in_target,
                                        label_loc_preds[i],
                                        &decoded_bbox,
                                        args->prior_is_normalized,
                                        args->input_width,
                                        args->input_height,
                                        args->clip_before_nms);
                    bboxes_per_image[label].emplace_back(decoded_bbox);
                }
            }
        }
        // Extract confidences per image.
        if (nms_type == NMSType::CAFFE) {
            extract_confidences_per_image_caffe<dtype>(stream, instance, confidences, num_of_priors, num_classes);
        } else {
            extract_confidences_per_image_mxnet<dtype>(stream, instance, confidences, num_of_priors, num_classes, scoreIndexPairs);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, detection_output_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }

        const int num_of_images = instance.location_memory()->get_layout().batch();  // batch size
        // Per image : label -> decoded bounding boxes.
        std::vector<std::vector<std::vector<bounding_box>>> bboxes(num_of_images);
        // Per image : class -> confidences per bounding box.
        std::vector<std::vector<std::vector<std::pair<float, int>>>> confidences(num_of_images);

        std::vector<std::vector<std::pair<float, std::pair<int, int>>>> scoreIndexPairs;
        if (instance.location_memory()->get_layout().data_type == data_types::f32) {
            prepare_data<ov::element_type_traits<data_types::f32>::value_type>(stream, instance, bboxes, confidences, scoreIndexPairs);
            generate_detections<ov::element_type_traits<data_types::f32>::value_type>(stream, instance, num_of_images, bboxes, confidences, scoreIndexPairs);
        } else {
            prepare_data<ov::element_type_traits<data_types::f16>::value_type>(stream, instance, bboxes, confidences, scoreIndexPairs);
            generate_detections<ov::element_type_traits<data_types::f16>::value_type>(stream, instance, num_of_images, bboxes, confidences, scoreIndexPairs);
        }

        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }

        return stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    static std::unique_ptr<primitive_impl> create(const detection_output_node& arg, const kernel_impl_params&) {
        return make_unique<detection_output_impl>(arg);
    }
};

namespace detail {

attach_detection_output_impl::attach_detection_output_impl() {
    auto formats = {
        format::bfyx
    };

    auto types = {
        data_types::f32,
        data_types::f16
    };

    implementation_map<detection_output>::add(impl_types::cpu, shape_types::any, detection_output_impl::create, types, formats);
}

}  // namespace detail

}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::detection_output_impl)
