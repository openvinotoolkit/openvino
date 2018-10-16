/*
// Copyright (c) 2016 Intel Corporation
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
#include "kd_selector.h"
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

    inline const float & clamp(const float & v, const float & lower, const float & upper)
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
            return (a.confidence > b.confidence) || (a.confidence == b.confidence && a.ord > b.ord);
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
            int img_h)
    {
        const float anchor_w = box.end_x - box.start_x + 1.0f;
        const float anchor_h = box.end_y - box.start_y + 1;
        const float center_x = box.start_x + anchor_w * .5f;
        const float center_y = box.start_y + anchor_h *.5f;

        const float pred_center_x = delta.shift_x * anchor_w + center_x + anchor_shift_x;
        const float pred_center_y = delta.shift_y * anchor_h + center_y + anchor_shift_y;
        const float half_pred_w = std::exp(delta.log_w) * anchor_w * .5f;
        const float half_pred_h = std::exp(delta.log_h) * anchor_h * .5f;

        return
        {
            clamp(pred_center_x - half_pred_w, 0.f, img_w - 1.f), clamp(pred_center_y - half_pred_h, 0.f, img_h - 1.f),
            clamp(pred_center_x + half_pred_w, 0.f, img_w - 1.f), clamp(pred_center_y + half_pred_h, 0.f, img_h - 1.f)
        };
    }
        
    std::vector<roi_t> perform_nms(
            const std::vector<proposal_t>& proposals,
            float iou_threshold,
            size_t top_n)
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
                    const float intersect_width = std::min(bbox.x1, res_bbox.x1) - std::max(bbox.x0, res_bbox.x0) + 1.f;
                    const float intersect_height = std::min(bbox.y1, res_bbox.y1) - std::max(bbox.y0, res_bbox.y0) + 1.f;
                    const float intersect_size = intersect_width * intersect_height;
                    overlap = intersect_size / (bbox.area() + res_bbox.area() - intersect_size);
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

        // feat map sizes
        const auto& score_size = cls_scores.get_layout().size;
        int fm_h = score_size.spatial[1];
        int fm_w = score_size.spatial[0];
        
        int fm_sz = fm_w * fm_h;

        // original input image to the graph (after possible scaling etc.) so that coordinates are valid for it
        mem_lock<dtype> image_info_ptr{ image_info };
        const dtype* image_info_mem = image_info_ptr.data();

        int img_w = 1;
        int img_h = 1;
        int img_z = 1;
        int min_bbox_x = 1;
        int min_bbox_y = 1;
        int scaled_min_bbox_size = instance.argument.min_bbox_size;

        if (image_info.get_layout().count() == 4)
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
            if (image_info.get_layout().count() > proposal_inst::image_info_scale_min_bbox_x)
            {
                min_bbox_x = static_cast<int>(min_bbox_x * float_read_helper(image_info_mem + proposal_inst::image_info_scale_min_bbox_x));
            }

            min_bbox_y = scaled_min_bbox_size;
            if (image_info.get_layout().count() > proposal_inst::image_info_scale_min_bbox_y)
            {
                min_bbox_y = static_cast<int>(min_bbox_y * float_read_helper(image_info_mem + proposal_inst::image_info_scale_min_bbox_y));
            }
        }

        mem_lock<dtype> cls_scores_ptr{ cls_scores };
        mem_lock<dtype> bbox_pred_ptr{ bbox_pred };
        const dtype* cls_scores_mem = cls_scores_ptr.data();
        const dtype* bbox_pred_mem  = bbox_pred_ptr.data();

        std::vector<proposal_t> sorted_proposals_confidence;
        sorted_proposals_confidence.reserve(fm_h * fm_w * anchors_num);
        for (int y = 0; y < fm_h; ++y)
        {
            const int anchor_shift_y = y * instance.argument.feature_stride;

            for (int x = 0; x < fm_w; ++x)
            {
                const int anchor_shift_x = x * instance.argument.feature_stride;
                const int location_index = y * fm_w + x;

                // we assume proposals are grouped by window location
                for (unsigned int anchor_index = 0; anchor_index < anchors_num ; anchor_index++)
                {
                    float dx0 = float_read_helper(bbox_pred_mem + location_index + fm_sz * (anchor_index * 4 + 0));
                    float dy0 = float_read_helper(bbox_pred_mem + location_index + fm_sz * (anchor_index * 4 + 1));
                    float dx1 = float_read_helper(bbox_pred_mem + location_index + fm_sz * (anchor_index * 4 + 2));
                    float dy1 = float_read_helper(bbox_pred_mem + location_index + fm_sz * (anchor_index * 4 + 3));

                    delta_t bbox_delta { dx0, dy0, dx1, dy1 };

                    unsigned int scores_index = location_index + fm_sz * (anchor_index + (unsigned int)anchors_num);
                    float proposal_confidence = float_read_helper(cls_scores_mem + scores_index);

                    const roi_t& roi = gen_bbox(anchors[anchor_index], bbox_delta, anchor_shift_x, anchor_shift_y, img_w, img_h);

                    int bbox_w = (int)roi.x1 - (int)roi.x0 + 1;
                    int bbox_h = (int)roi.y1 - (int)roi.y0 + 1;

                    if (bbox_w >= min_bbox_x && bbox_h >= min_bbox_y && proposal_confidence > 0)
                    {
                        sorted_proposals_confidence.emplace_back(roi, proposal_confidence, sorted_proposals_confidence.size());
                    }
                }
            }
        }

        sort_and_keep_n_items(sorted_proposals_confidence, instance.argument.pre_nms_topn);
        const std::vector<roi_t>& res = perform_nms(sorted_proposals_confidence, instance.argument.iou_threshold, instance.argument.post_nms_topn);

        auto& output = instance.output_memory();
        
        mem_lock<dtype> output_ptr{ output };
        dtype* top_data = output_ptr.data();        

        size_t res_num_rois = res.size();
        
        for (size_t i = 0; i < res_num_rois; ++i)
        {
            float_write_helper(top_data + 5 * i    , 0.0f);
            float_write_helper(top_data + 5 * i + 1, res[i].x0);
            float_write_helper(top_data + 5 * i + 2, res[i].y0);
            float_write_helper(top_data + 5 * i + 3, res[i].x1);
            float_write_helper(top_data + 5 * i + 4, res[i].y1);
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
        const size_t count = l.size.count();

        //Supported image_info sizes and components meaning:
        // - image_info[3] = { img_height, img_width, img_depth }
        // - image_info[4] = { img_height, img_width, scale_min_bbox_y, scale_min_bbox_x }
        // - image_info[6] = { img_height, img_width, img_depth, scale_min_bbox_y, scale_min_bbox_x, scale_depth_index }
        if ((size_t)l.size.spatial[0] != count || (count != 3 && count != 4 && count != 6)) {
            CLDNN_ERROR_MESSAGE(arg.id(), "image_info must have either 3, 4 or 6 items");
        }
        CLDNN_ERROR_BOOL(arg.id(), "Batching", !hasSingleBatchOutput(arg.bbox_pred()), "Proposal doesn't support batching.");
        CLDNN_ERROR_BOOL(arg.id(), "Batching", !hasSingleBatchOutput(arg.cls_score()), "Proposal doesn't support batching.");

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
