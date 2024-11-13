// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proposal_inst.h"
#include "intel_gpu/runtime/engine.hpp"
#include "impls/registry/implementation_map.hpp"
#include "register.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <utility>

#define EPSILON 0.00001f

namespace cldnn {
namespace cpu {

namespace {

/****************************************************************************
 *                                                                          *
 *                              Common Utils                                *
 *                                                                          *
 ****************************************************************************/

inline const float& clamp(const float& v, const float& lower, const float& upper) {
    return std::max(lower, std::min(v, upper));
}

struct roi_t {
    float x0, y0, x1, y1;
};

struct delta_t {
    float shift_x, shift_y, log_w, log_h;
};

struct proposal_t {
    proposal_t() = default;
    proposal_t(const roi_t& r, const float c, const size_t& o) : roi(r), confidence(c), ord(o) {}

    roi_t roi;
    float confidence;
    size_t ord;
};

inline float float_read_helper(const float* mem) { return *mem; }

inline float float_read_helper(const ov::float16* mem) { return static_cast<float>(*mem); }

inline void float_write_helper(float* mem, float f) { *mem = f; }

inline void float_write_helper(ov::float16* mem, float f) { *mem = static_cast<ov::float16>(f); }

/****************************************************************************
 *                                                                          *
 *                              Impl Details                                *
 *                                                                          *
 ****************************************************************************/

void sort_and_keep_n_items(std::vector<proposal_t>& proposals, size_t n) {
    auto cmp_fn = [](const proposal_t& a, const proposal_t& b) { return (a.confidence > b.confidence); };

    if (proposals.size() > n) {
        std::partial_sort(proposals.begin(), proposals.begin() + n, proposals.end(), cmp_fn);
        proposals.resize(n);
    } else {
        std::sort(proposals.begin(), proposals.end(), cmp_fn);
    }
}

roi_t gen_bbox(const proposal_inst::anchor& box,
               const delta_t& delta,
               int anchor_shift_x,
               int anchor_shift_y,
               int img_w,
               int img_h,
               float coordinates_offset,
               bool initial_clip,
               bool clip_before_nms,
               bool for_deformable) {
    float x0 = box.start_x + anchor_shift_x;
    float y0 = box.start_y + anchor_shift_y;
    float x1 = box.end_x + anchor_shift_x;
    float y1 = box.end_y + anchor_shift_y;

    if (initial_clip) {
        x0 = clamp(x0, 0.0f, static_cast<float>(img_w));
        y0 = clamp(y0, 0.0f, static_cast<float>(img_h));
        x1 = clamp(x1, 0.0f, static_cast<float>(img_w));
        y1 = clamp(y1, 0.0f, static_cast<float>(img_h));
    }

    const float anchor_w = x1 - x0 + coordinates_offset;
    const float anchor_h = y1 - y0 + coordinates_offset;
    const float center_x = for_deformable ? x0 + 0.5f * (anchor_w - 1) :
                           x0 + 0.5f * anchor_w;
    const float center_y = for_deformable ? y0 + 0.5f * (anchor_h - 1) :
                           y0 + 0.5f * anchor_h;

    const float pred_center_x = delta.shift_x * anchor_w + center_x;
    const float pred_center_y = delta.shift_y * anchor_h + center_y;
    const float half_pred_w = for_deformable ? (std::exp(delta.log_w) * anchor_w - 1) * .5f :
                              std::exp(delta.log_w) * anchor_w * .5f;
    const float half_pred_h = for_deformable ? (std::exp(delta.log_h) * anchor_h - 1) * .5f :
                              std::exp(delta.log_h) * anchor_h * .5f;

    float new_x0 = pred_center_x - half_pred_w;
    float new_y0 = pred_center_y - half_pred_h;
    float new_x1 = pred_center_x + half_pred_w;
    float new_y1 = pred_center_y + half_pred_h;

    if (clip_before_nms) {
        new_x0 = clamp(new_x0, 0.f, img_w - coordinates_offset);
        new_y0 = clamp(new_y0, 0.f, img_h - coordinates_offset);
        new_x1 = clamp(new_x1, 0.f, img_w - coordinates_offset);
        new_y1 = clamp(new_y1, 0.f, img_h - coordinates_offset);
    }

    return {new_x0, new_y0, new_x1, new_y1};
}

std::vector<roi_t> perform_nms(const std::vector<proposal_t>& proposals,
                               float iou_threshold,
                               size_t top_n,
                               float coordinates_offset) {
    std::vector<roi_t> res;
    res.reserve(top_n);

    for (const auto& prop : proposals) {
        const roi_t& bbox = prop.roi;

        bool overlaps = std::any_of(res.begin(), res.end(), [&](const roi_t& res_bbox) {
            bool intersecting =
                (bbox.x0 < res_bbox.x1) & (res_bbox.x0 < bbox.x1) & (bbox.y0 < res_bbox.y1) & (res_bbox.y0 < bbox.y1);
            float overlap = 0.0f;
            if (intersecting) {
                const float x0 = std::max(bbox.x0, res_bbox.x0);
                const float y0 = std::max(bbox.y0, res_bbox.y0);
                const float x1 = std::min(bbox.x1, res_bbox.x1);
                const float y1 = std::min(bbox.y1, res_bbox.y1);

                const float intersect_width = std::max(0.0f, x1 - x0 + coordinates_offset);
                const float intersect_height = std::max(0.0f, y1 - y0 + coordinates_offset);
                const float intersect_size = intersect_width * intersect_height;

                const float A_area =
                    (bbox.x1 - bbox.x0 + coordinates_offset) * (bbox.y1 - bbox.y0 + coordinates_offset);
                const float B_area =
                    (res_bbox.x1 - res_bbox.x0 + coordinates_offset) * (res_bbox.y1 - res_bbox.y0 + coordinates_offset);

                overlap = intersect_size / (A_area + B_area - intersect_size);
            }
            return overlap > iou_threshold;
        });

        if (!overlaps) {
            res.push_back(bbox);
            if (res.size() == top_n)
                break;
        }
    }

    res.resize(top_n);
    return res;
}
}  // anonymous namespace

/****************************************************************************
 *                                                                          *
 *                              Proposal Layer                              *
 *                                                                          *
 ****************************************************************************/

struct im_info_t {
    int img_w;
    int img_h;
    int img_z;
    int min_bbox_x;
    int min_bbox_y;
};

struct proposal_impl : typed_primitive_impl<proposal> {
    using parent = typed_primitive_impl<proposal>;
    using parent::parent;

    proposal_impl() : parent() {}

    explicit proposal_impl(const proposal_node& arg) {}

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::proposal_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<proposal_impl>(*this);
    }

    template <typename dtype>
    void read_image_info(stream& stream, proposal_inst& instance, im_info_t& im_info) {
        auto image_info = instance.dep_memory_ptr(proposal_inst::image_info_index);
        mem_lock<dtype, mem_lock_type::read> image_info_ptr{image_info, stream};
        const dtype* image_info_mem = image_info_ptr.data();
        const auto& primitive = instance.get_typed_desc<proposal>();
        bool swap_xy = primitive->swap_xy;

        // original input image to the graph (after possible scaling etc.) so that coordinates are valid for it
        int img_w = 1;
        int img_h = 1;
        int img_z = 1;
        int min_bbox_x = 1;
        int min_bbox_y = 1;

        auto image_info_size = image_info->get_layout().get_tensor();
        auto image_info_count = image_info_size.feature[0] == 1 ? image_info_size.batch[0] : image_info_size.feature[0];

        int scaled_min_bbox_size = primitive->min_bbox_size;

        if (image_info_count == 4) {
            img_w =
                static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_width_index) + EPSILON);
            img_h =
                static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_height_index) + EPSILON);
            min_bbox_x = static_cast<int>(scaled_min_bbox_size * float_read_helper(image_info_mem + 3));
            min_bbox_y = static_cast<int>(scaled_min_bbox_size * float_read_helper(image_info_mem + 2));
        } else {
            img_w =
                static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_width_index) + EPSILON);
            img_h =
                static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_height_index) + EPSILON);
            img_z =
                static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_depth_index) + EPSILON);

            scaled_min_bbox_size *= img_z;

            min_bbox_x = scaled_min_bbox_size;
            if (image_info_count > proposal_inst::image_info_scale_min_bbox_x) {
                min_bbox_x = static_cast<int>(
                    min_bbox_x * float_read_helper(image_info_mem + proposal_inst::image_info_scale_min_bbox_x));
            }

            min_bbox_y = scaled_min_bbox_size;
            if (image_info_count > proposal_inst::image_info_scale_min_bbox_y) {
                min_bbox_y = static_cast<int>(
                    min_bbox_y * float_read_helper(image_info_mem + proposal_inst::image_info_scale_min_bbox_y));
            }
        }

        if (swap_xy) {
            std::swap(img_w, img_h);
        }

        im_info.img_h = img_h;
        im_info.img_w = img_w;
        im_info.img_z = img_z;
        im_info.min_bbox_x = min_bbox_x;
        im_info.min_bbox_y = min_bbox_y;
    }

    template <typename dtype>
    void execute(stream& stream, proposal_inst& instance, im_info_t im_info, dtype* proposal_prob_ptr = nullptr) {
        const std::vector<proposal_inst::anchor>& anchors = instance.get_anchors();

        size_t anchors_num = anchors.size();

        auto cls_scores = instance.dep_memory_ptr(proposal_inst::cls_scores_index);
        auto bbox_pred = instance.dep_memory_ptr(proposal_inst::bbox_pred_index);
        const auto& primitive = instance.get_typed_desc<proposal>();
        bool swap_xy = primitive->swap_xy;
        bool initial_clip = primitive->initial_clip;
        bool clip_before_nms = primitive->clip_before_nms;
        bool clip_after_nms = primitive->clip_after_nms;
        float coordinates_offset = primitive->coordinates_offset;
        float box_coordinate_scale = primitive->box_coordinate_scale;
        float box_size_scale = primitive->box_size_scale;
        bool for_deformable = primitive->for_deformable;

        // feat map sizes
        const auto& score_layout = cls_scores->get_layout();
        int fm_h = score_layout.spatial(1);
        int fm_w = score_layout.spatial(0);

        int fm_sz = fm_w * fm_h;

        mem_lock<dtype, mem_lock_type::read> cls_scores_ptr{cls_scores, stream};
        mem_lock<dtype, mem_lock_type::read> bbox_pred_ptr{std::move(bbox_pred), stream};
        const dtype* cls_scores_mem = cls_scores_ptr.data();
        const dtype* bbox_pred_mem = bbox_pred_ptr.data();

        for (int n = 0; n < score_layout.batch(); n++) {
            std::vector<proposal_t> sorted_proposals_confidence;
            size_t num_proposals = fm_h * fm_w * anchors_num;
            sorted_proposals_confidence.reserve(num_proposals);
            for (int y = 0; y < fm_h; ++y) {
                for (int x = 0; x < fm_w; ++x) {
                    const int anchor_shift_x = (swap_xy ? y : x) * primitive->feature_stride;
                    const int anchor_shift_y = (swap_xy ? x : y) * primitive->feature_stride;
                    const int location_index = y * fm_w + x;

                    // we assume proposals are grouped by window location
                    for (unsigned int anchor_index = 0; anchor_index < anchors_num; anchor_index++) {
                        float dx0 = float_read_helper(bbox_pred_mem + n * num_proposals * 4 + location_index +
                                                      fm_sz * (anchor_index * 4 + 0)) /
                                    box_coordinate_scale;
                        float dy0 = float_read_helper(bbox_pred_mem + n * num_proposals * 4 + location_index +
                                                      fm_sz * (anchor_index * 4 + 1)) /
                                    box_coordinate_scale;
                        float dx1 = float_read_helper(bbox_pred_mem + n * num_proposals * 4 + location_index +
                                                      fm_sz * (anchor_index * 4 + 2)) /
                                    box_size_scale;
                        float dy1 = float_read_helper(bbox_pred_mem + n * num_proposals * 4 + location_index +
                                                      fm_sz * (anchor_index * 4 + 3)) /
                                    box_size_scale;

                        delta_t bbox_delta{dx0, dy0, dx1, dy1};

                        const roi_t& roi = gen_bbox(anchors[anchor_index],
                                                    bbox_delta,
                                                    anchor_shift_x,
                                                    anchor_shift_y,
                                                    im_info.img_w,
                                                    im_info.img_h,
                                                    coordinates_offset,
                                                    initial_clip,
                                                    clip_before_nms,
                                                    for_deformable);

                        int bbox_w = static_cast<int>((roi.x1 - roi.x0 + coordinates_offset));
                        int bbox_h = static_cast<int>((roi.y1 - roi.y0 + coordinates_offset));

                        size_t scores_index =
                            n * num_proposals * 2 + location_index + fm_sz * (anchor_index + anchors_num);
                        float proposal_confidence = (im_info.min_bbox_x <= bbox_w) * (im_info.min_bbox_y <= bbox_h) *
                                                    float_read_helper(cls_scores_mem + scores_index);
                        sorted_proposals_confidence.emplace_back(roi,
                                                                 proposal_confidence,
                                                                 sorted_proposals_confidence.size());
                    }
                }
            }

            size_t pre_nms = std::min(primitive->pre_nms_topn, static_cast<int>(sorted_proposals_confidence.size()));
            sort_and_keep_n_items(sorted_proposals_confidence, pre_nms);
            std::vector<roi_t> res = perform_nms(sorted_proposals_confidence,
                                                 primitive->iou_threshold,
                                                 primitive->post_nms_topn,
                                                 coordinates_offset);

            auto output = instance.output_memory_ptr();

            mem_lock<dtype, mem_lock_type::write> output_ptr{output, stream};
            dtype* top_data = output_ptr.data() + n * primitive->post_nms_topn * 5;

            dtype* top_data_prob = proposal_prob_ptr == nullptr ? nullptr : proposal_prob_ptr + n * primitive->post_nms_topn;

            size_t res_num_rois = res.size();

            for (size_t i = 0; i < res_num_rois; ++i) {
                if (clip_after_nms) {
                    res[i].x0 = clamp(res[i].x0, 0.0f, static_cast<float>(im_info.img_w));
                    res[i].y0 = clamp(res[i].y0, 0.0f, static_cast<float>(im_info.img_h));
                    res[i].x1 = clamp(res[i].x1, 0.0f, static_cast<float>(im_info.img_w));
                    res[i].y1 = clamp(res[i].y1, 0.0f, static_cast<float>(im_info.img_h));
                }

                float_write_helper(top_data + 5 * i + 0, static_cast<float>(n));
                float_write_helper(top_data + 5 * i + 1, res[i].x0 / (primitive->normalize ? im_info.img_w : 1.0f));
                float_write_helper(top_data + 5 * i + 2, res[i].y0 / (primitive->normalize ? im_info.img_h : 1.0f));
                float_write_helper(top_data + 5 * i + 3, res[i].x1 / (primitive->normalize ? im_info.img_w : 1.0f));
                float_write_helper(top_data + 5 * i + 4, res[i].y1 / (primitive->normalize ? im_info.img_h : 1.0f));
                if (top_data_prob != nullptr && i < sorted_proposals_confidence.size()) {
                    float_write_helper(top_data_prob + i, sorted_proposals_confidence[i].confidence);
                }
            }

            for (size_t i = res_num_rois; i < (size_t)primitive->post_nms_topn; i++) {
                float_write_helper(top_data + 5 * i + 0, -1.0f);
                float_write_helper(top_data + 5 * i + 1, 0.0f);
                float_write_helper(top_data + 5 * i + 2, 0.0f);
                float_write_helper(top_data + 5 * i + 3, 0.0f);
                float_write_helper(top_data + 5 * i + 4, 0.0f);
                if (top_data_prob != nullptr)
                    float_write_helper(top_data_prob + i, 0.0f);
            }
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, proposal_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }

        im_info_t im_info;
        if (instance.dep_memory(proposal_inst::image_info_index).get_layout().data_type == data_types::f16) {
            read_image_info<ov::element_type_traits<data_types::f16>::value_type>(stream, instance, im_info);
        } else {
            read_image_info<ov::element_type_traits<data_types::f32>::value_type>(stream, instance, im_info);
        }

        if (instance.dep_memory(proposal_inst::cls_scores_index).get_layout().data_type !=
            instance.dep_memory(proposal_inst::bbox_pred_index).get_layout().data_type)
            throw std::runtime_error("clDNN: proposal primitive doesn't support mixed bbox and scores types");

        if (instance.dependencies().size() == 4) {
            auto proposal_probabilities = instance.dep_memory_ptr(proposal_inst::proposal_probabilities_out);
            if (instance.dep_memory(proposal_inst::cls_scores_index).get_layout().data_type == data_types::f16) {
                mem_lock<ov::element_type_traits<data_types::f16>::value_type, mem_lock_type::read> proposal_prob_ptr{proposal_probabilities, stream};
                execute<ov::element_type_traits<data_types::f16>::value_type>(stream, instance, im_info, proposal_prob_ptr.data());
            } else {
                mem_lock<ov::element_type_traits<data_types::f32>::value_type, mem_lock_type::read> proposal_prob_ptr{proposal_probabilities, stream};
                execute<ov::element_type_traits<data_types::f32>::value_type>(stream, instance, im_info, proposal_prob_ptr.data());
            }
        } else if (instance.outputs_memory_count() == 2) {
            auto proposal_probabilities = instance.output_memory_ptr(1);
            if (instance.dep_memory(proposal_inst::cls_scores_index).get_layout().data_type == data_types::f16) {
                mem_lock<ov::element_type_traits<data_types::f16>::value_type, mem_lock_type::write> proposal_prob_ptr{proposal_probabilities, stream};
                execute<ov::element_type_traits<data_types::f16>::value_type>(stream, instance, im_info, proposal_prob_ptr.data());
            } else {
                mem_lock<ov::element_type_traits<data_types::f32>::value_type, mem_lock_type::write> proposal_prob_ptr{proposal_probabilities, stream};
                execute<ov::element_type_traits<data_types::f32>::value_type>(stream, instance, im_info, proposal_prob_ptr.data());
            }
        } else {
            if (instance.dep_memory(proposal_inst::cls_scores_index).get_layout().data_type == data_types::f16) {
                execute<ov::element_type_traits<data_types::f16>::value_type>(stream, instance, im_info);
            } else {
                execute<ov::element_type_traits<data_types::f32>::value_type>(stream, instance, im_info);
            }
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

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    static std::unique_ptr<primitive_impl> create(const proposal_node& arg, const kernel_impl_params& impl_param) {
        const layout& l = impl_param.input_layouts[2];
        if (l.is_static() && l.get_partial_shape().size() >= 2) {
            const size_t count = l.get_partial_shape()[1].get_length() == 1 ? l.get_partial_shape()[0].get_length() :
                                 l.get_partial_shape()[1].get_length();

            // Supported image_info sizes and components meaning:
            // - image_info[3] = { img_height, img_width, img_depth }
            // - image_info[4] = { img_height, img_width, scale_min_bbox_y, scale_min_bbox_x }
            // - image_info[6] = { img_height, img_width, img_depth, scale_min_bbox_y, scale_min_bbox_x, scale_depth_index }
            OPENVINO_ASSERT(one_of(count, {3, 4, 6}), arg.id(), "image_info must have either 3, 4 or 6 items");
        }

        return make_unique<proposal_impl>(arg);
    }
};

namespace detail {

attach_proposal_impl::attach_proposal_impl() {
    auto formats = {
        format::bfyx
    };

    auto types = {
        data_types::f32,
        data_types::f16
    };

    implementation_map<proposal>::add(impl_types::cpu, shape_types::static_shape, proposal_impl::create, types, formats);
    implementation_map<proposal>::add(impl_types::cpu, shape_types::dynamic_shape, proposal_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::proposal_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::proposal)
