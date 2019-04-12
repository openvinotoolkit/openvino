/*
// Copyright (c) 2016-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "proposal_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "network_impl.h"
#include "engine_impl.h"
#include "math_utils.h"
#include "error_handler.h"

#include <algorithm>
#include <string>

#define EPSILON 0.00001f

namespace cldnn { namespace gpu {

namespace {

    /****************************************************************************
    *                                                                          *
    *                              Common Utils                                *
    *                                                                          *
    ****************************************************************************/

    inline const float& clamp(const float & v, const float & lower, const float & upper)
    {
        return std::max(lower, std::min(v, upper));
    }

    inline bool hasSingleBatchOutput(const program_node & node)
    {
        const auto & batch = node.get_output_layout().size.batch;

        return batch.empty() || (batch.size() == 1 && batch[0] == 1);
    }

    struct roi_t
    {
        float x0, y0, x1, y1;

        inline float area() const
        {
            return std::max(0.f, y1 - y0 + 1.f) * std::max(0.f, x1 - x0 + 1.f);
        }
    };

    struct delta_t { float shift_x, shift_y, log_w, log_h; };

    struct proposal_t
    {
        proposal_t() = default;
        proposal_t(const roi_t& r, const float c, const size_t& o) : roi(r), confidence(c), ord(o) {}

        roi_t roi;
        float confidence;
        size_t ord;
    };

    inline float float_read_helper(const float* mem)
    {
        return *mem;
    }

    inline float float_read_helper(const half_t* mem)
    {
        return float16_to_float32(*((uint16_t*)(mem)));
    }

    inline void float_write_helper(float* mem, float f)
    {
        *mem = f;
    }

    inline void float_write_helper(half_t* mem, float f)
    {
        *mem = (half_t)float32_to_float16(f);
    }

    /****************************************************************************
     *                                                                          *
     *                              Impl Details                                *
     *                                                                          *
     ****************************************************************************/

    void sort_and_keep_n_items(std::vector<proposal_t>& proposals, size_t n)
    {
        auto cmp_fn = [](const proposal_t& a, const proposal_t& b)
        {
            return (a.confidence > b.confidence);
        };

        if (proposals.size() > n)
        {
            std::partial_sort(proposals.begin(), proposals.begin() + n, proposals.end(), cmp_fn);
            proposals.resize(n);
        }
        else
        {
            std::sort(proposals.begin(), proposals.end(), cmp_fn);
        }
    }

    roi_t gen_bbox(
            const proposal_inst::anchor& box,
            const delta_t& delta,
            int anchor_shift_x,
            int anchor_shift_y,
            int img_w,
            int img_h,
            float coordinates_offset,
            bool initial_clip,
            bool clip_before_nms)
    {
        float x0 = box.start_x + anchor_shift_x;
        float y0 = box.start_y + anchor_shift_y;
        float x1 = box.end_x + anchor_shift_x;
        float y1 = box.end_y + anchor_shift_y;

        if (initial_clip)
        {
            x0 = clamp(x0, 0.0f, static_cast<float>(img_w));
            y0 = clamp(y0, 0.0f, static_cast<float>(img_h));
            x1 = clamp(x1, 0.0f, static_cast<float>(img_w));
            y1 = clamp(y1, 0.0f, static_cast<float>(img_h));
        }

        const float anchor_w = x1 - x0 + coordinates_offset;
        const float anchor_h = y1 - y0 + coordinates_offset;
        const float center_x = x0 + 0.5f * anchor_w;
        const float center_y = y0 + 0.5f * anchor_h;

        const float pred_center_x = delta.shift_x * anchor_w + center_x;
        const float pred_center_y = delta.shift_y * anchor_h + center_y;
        const float half_pred_w = std::exp(delta.log_w) * anchor_w * .5f;
        const float half_pred_h = std::exp(delta.log_h) * anchor_h * .5f;

        float new_x0 = pred_center_x - half_pred_w;
        float new_y0 = pred_center_y - half_pred_h;
        float new_x1 = pred_center_x + half_pred_w;
        float new_y1 = pred_center_y + half_pred_h;

        if (clip_before_nms)
        {
            new_x0 = clamp(new_x0, 0.f, img_w - coordinates_offset);
            new_y0 = clamp(new_y0, 0.f, img_h - coordinates_offset);
            new_x1 = clamp(new_x1, 0.f, img_w - coordinates_offset);
            new_y1 = clamp(new_y1, 0.f, img_h - coordinates_offset);
        }

        return { new_x0, new_y0, new_x1, new_y1 };
    }

    std::vector<roi_t> perform_nms(
            const std::vector<proposal_t>& proposals,
            float iou_threshold,
            size_t top_n,
            float coordinates_offset)
    {
        std::vector<roi_t> res;
        res.reserve(top_n);

        for (const auto & prop : proposals)
        {
            const roi_t& bbox = prop.roi;

            bool overlaps = std::any_of(res.begin(), res.end(), [&](const roi_t& res_bbox)
            {
                bool intersecting = (bbox.x0 < res_bbox.x1) & (res_bbox.x0 < bbox.x1) & (bbox.y0 < res_bbox.y1) & (res_bbox.y0 < bbox.y1);
                float overlap = 0.0f;
                if (intersecting)
                {
                    const float x0 = std::max(bbox.x0, res_bbox.x0);
                    const float y0 = std::max(bbox.y0, res_bbox.y0);
                    const float x1 = std::min(bbox.x1, res_bbox.x1);
                    const float y1 = std::min(bbox.y1, res_bbox.y1);

                    const float intersect_width = std::max(0.0f, x1 - x0 + coordinates_offset);
                    const float intersect_height = std::max(0.0f, y1 - y0 + coordinates_offset);
                    const float intersect_size = intersect_width * intersect_height;

                    const float A_area = (bbox.x1 - bbox.x0 + coordinates_offset) * (bbox.y1 - bbox.y0 + coordinates_offset);
                    const float B_area = (res_bbox.x1 - res_bbox.x0 + coordinates_offset) * (res_bbox.y1 - res_bbox.y0 + coordinates_offset);

                    overlap = intersect_size / (A_area + B_area - intersect_size);
                }
                return overlap > iou_threshold;
            });

            if (!overlaps)
            {
                res.push_back(bbox);
                if (res.size() == top_n) break;
            }
        }

        res.resize(top_n);
        return res;
    }
} // anonymous namespace


/****************************************************************************
*                                                                          *
*                              Proposal Layer                              *
*                                                                          *
****************************************************************************/

struct proposal_gpu : typed_primitive_impl<proposal>
{
    const proposal_node& outer;

    proposal_gpu(const proposal_node& arg)
            : outer(arg)
    {}

    template<typename dtype>
    void execute(proposal_inst& instance)
    {
        const std::vector<proposal_inst::anchor>& anchors = instance.get_anchors();

        size_t anchors_num = anchors.size();

        auto& cls_scores = instance.dep_memory(proposal_inst::cls_scores_index);
        auto& bbox_pred  = instance.dep_memory(proposal_inst::bbox_pred_index);
        auto& image_info = instance.dep_memory(proposal_inst::image_info_index);

        // original input image to the graph (after possible scaling etc.) so that coordinates are valid for it
        mem_lock<dtype> image_info_ptr{ image_info };
        const dtype* image_info_mem = image_info_ptr.data();

        int img_w = 1;
        int img_h = 1;
        int img_z = 1;
        int min_bbox_x = 1;
        int min_bbox_y = 1;
        int scaled_min_bbox_size = instance.argument.min_bbox_size;

        bool swap_xy = instance.argument.swap_xy;
        bool initial_clip = instance.argument.initial_clip;
        bool clip_before_nms = instance.argument.clip_before_nms;
        bool clip_after_nms = instance.argument.clip_after_nms;
        float coordinates_offset = instance.argument.coordinates_offset;
        float box_coordinate_scale = instance.argument.box_coordinate_scale;
        float box_size_scale = instance.argument.box_size_scale;

        if (image_info.get_layout().size.feature[0] == 4)
        {
            img_w = static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_width_index) + EPSILON);
            img_h = static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_height_index) + EPSILON);
            min_bbox_x = static_cast<int>(scaled_min_bbox_size * float_read_helper(image_info_mem + 3));
            min_bbox_y = static_cast<int>(scaled_min_bbox_size * float_read_helper(image_info_mem + 2));
        }
        else
        {
            img_w = static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_width_index) + EPSILON);
            img_h = static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_height_index) + EPSILON);
            img_z = static_cast<int>(float_read_helper(image_info_mem + proposal_inst::image_info_depth_index) + EPSILON);

            scaled_min_bbox_size *= img_z;

            min_bbox_x = scaled_min_bbox_size;
            if (image_info.get_layout().size.feature[0] > proposal_inst::image_info_scale_min_bbox_x)
            {
                min_bbox_x = static_cast<int>(min_bbox_x * float_read_helper(image_info_mem + proposal_inst::image_info_scale_min_bbox_x));
            }

            min_bbox_y = scaled_min_bbox_size;
            if (image_info.get_layout().size.feature[0] > proposal_inst::image_info_scale_min_bbox_y)
            {
                min_bbox_y = static_cast<int>(min_bbox_y * float_read_helper(image_info_mem + proposal_inst::image_info_scale_min_bbox_y));
            }
        }

        if (swap_xy)
        {
            std::swap(img_w, img_h);
        }

        // feat map sizes
        const auto& score_size = cls_scores.get_layout().size;
        int fm_h = score_size.spatial[1];
        int fm_w = score_size.spatial[0];

        int fm_sz = fm_w * fm_h;

        mem_lock<dtype> cls_scores_ptr{ cls_scores };
        mem_lock<dtype> bbox_pred_ptr{ bbox_pred };
        const dtype* cls_scores_mem = cls_scores_ptr.data();
        const dtype* bbox_pred_mem  = bbox_pred_ptr.data();

        for (int n = 0; n < score_size.batch[0]; n++)
        {
            std::vector<proposal_t> sorted_proposals_confidence;
            size_t num_proposals = fm_h * fm_w * anchors_num;
            sorted_proposals_confidence.reserve(num_proposals);
            for (int y = 0; y < fm_h; ++y)
            {
                for (int x = 0; x < fm_w; ++x)
                {
                    const int anchor_shift_x = (swap_xy ? y : x) * instance.argument.feature_stride;
                    const int anchor_shift_y = (swap_xy ? x : y) * instance.argument.feature_stride;
                    const int location_index = y * fm_w + x;

                    // we assume proposals are grouped by window location
                    for (unsigned int anchor_index = 0; anchor_index < anchors_num ; anchor_index++)
                    {
                        float dx0 = float_read_helper(bbox_pred_mem + n*num_proposals*4 + location_index + fm_sz * (anchor_index * 4 + 0)) / box_coordinate_scale;
                        float dy0 = float_read_helper(bbox_pred_mem + n*num_proposals*4 + location_index + fm_sz * (anchor_index * 4 + 1)) / box_coordinate_scale;
                        float dx1 = float_read_helper(bbox_pred_mem + n*num_proposals*4 + location_index + fm_sz * (anchor_index * 4 + 2)) / box_size_scale;
                        float dy1 = float_read_helper(bbox_pred_mem + n*num_proposals*4 + location_index + fm_sz * (anchor_index * 4 + 3)) / box_size_scale;

                        delta_t bbox_delta { dx0, dy0, dx1, dy1 };

                        const roi_t& roi = gen_bbox(anchors[anchor_index], bbox_delta, anchor_shift_x, anchor_shift_y,
                                                    img_w, img_h, coordinates_offset, initial_clip, clip_before_nms);

                        int bbox_w = (int)(roi.x1 - roi.x0 + coordinates_offset);
                        int bbox_h = (int)(roi.y1 - roi.y0 + coordinates_offset);

                        size_t scores_index = n*num_proposals * 2 + location_index + fm_sz * (anchor_index + anchors_num);
                        float proposal_confidence = (min_bbox_x <= bbox_w)* (min_bbox_y <= bbox_h) * float_read_helper(cls_scores_mem + scores_index);
                        sorted_proposals_confidence.emplace_back(roi, proposal_confidence, sorted_proposals_confidence.size());
                    }
                }
            }

            size_t pre_nms = std::min(instance.argument.pre_nms_topn, (int)sorted_proposals_confidence.size());
            sort_and_keep_n_items(sorted_proposals_confidence, pre_nms);
            std::vector<roi_t> res = perform_nms(sorted_proposals_confidence, instance.argument.iou_threshold,
                                                 instance.argument.post_nms_topn, coordinates_offset);

            auto& output = instance.output_memory();

            mem_lock<dtype> output_ptr{ output };
            dtype* top_data = output_ptr.data() + n*instance.argument.post_nms_topn*5;

            size_t res_num_rois = res.size();


            for (size_t i = 0; i < res_num_rois; ++i)
            {
                if (clip_after_nms)
                {
                    res[i].x0 = clamp(res[i].x0, 0.0f, float(img_w));
                    res[i].y0 = clamp(res[i].y0, 0.0f, float(img_h));
                    res[i].x1 = clamp(res[i].x1, 0.0f, float(img_w));
                    res[i].y1 = clamp(res[i].y1, 0.0f, float(img_h));
                }

                float_write_helper(top_data + 5 * i + 0, float(n));
                float_write_helper(top_data + 5 * i + 1, res[i].x0 / (instance.argument.normalize ? img_w : 1.0f));
                float_write_helper(top_data + 5 * i + 2, res[i].y0 / (instance.argument.normalize ? img_h : 1.0f));
                float_write_helper(top_data + 5 * i + 3, res[i].x1 / (instance.argument.normalize ? img_w : 1.0f));
                float_write_helper(top_data + 5 * i + 4, res[i].y1 / (instance.argument.normalize ? img_h : 1.0f));
            }

            for (size_t i = res_num_rois; i < (size_t)instance.argument.post_nms_topn; i++)
            {
                float_write_helper(top_data + 5*i + 0, -1.0f);
                float_write_helper(top_data + 5*i + 1,  0.0f);
                float_write_helper(top_data + 5*i + 2,  0.0f);
                float_write_helper(top_data + 5*i + 3,  0.0f);
                float_write_helper(top_data + 5*i + 4,  0.0f);
            }
        }
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, proposal_inst& instance) override
    {
        for (auto& a : events)
        {
            a->wait();
        }

        auto ev = instance.get_network().get_engine().create_user_event(false);

        if (instance.dep_memory(proposal_inst::cls_scores_index).get_layout().data_type == data_types::f16)
        {
            execute<data_type_to_type<data_types::f16>::type>(instance);
        }
        else
        {
            execute<data_type_to_type<data_types::f32>::type>(instance);
        }

        dynamic_cast<cldnn::user_event*>(ev.get())->set(); // set as complete
        return ev;
    }

    static primitive_impl* create(const proposal_node& arg)
    {
        const layout & l = arg.image_info().get_output_layout();
        const size_t count = static_cast<size_t>(l.size.feature[0]);

        //Supported image_info sizes and components meaning:
        // - image_info[3] = { img_height, img_width, img_depth }
        // - image_info[4] = { img_height, img_width, scale_min_bbox_y, scale_min_bbox_x }
        // - image_info[6] = { img_height, img_width, img_depth, scale_min_bbox_y, scale_min_bbox_x, scale_depth_index }
        if (count != 3 && count != 4 && count != 6) {
            CLDNN_ERROR_MESSAGE(arg.id(), "image_info must have either 3, 4 or 6 items");
        }

        return new proposal_gpu(arg);
    }
};

namespace {
    struct attach {
        attach()
        {
            implementation_map<proposal>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), proposal_gpu::create);
            implementation_map<proposal>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), proposal_gpu::create);
        }

        ~attach() {}
    };
    attach attach_impl;
}
} }
