// Copyright (C) 2018-2020 Intel Corporation
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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void prior_box_clustered(
    const __global INPUT0_TYPE*  input0,
    const __global INPUT0_TYPE*  input1,
    __global OUTPUT0_TYPE* output)
{
    const int num_priors_ = sizeof(width_)/sizeof(width_[0]);
    const int var_size = sizeof(variance_)/sizeof(variance_[0]);

    const float img_width  = (img_w_ == 0) ? INPUT1_DIMS[3] : img_w_;
    const float img_height = (img_h_ == 0) ? INPUT1_DIMS[2] : img_h_;

    const float r_img_width  = 1.f/img_width;
    const float r_img_height = 1.f/img_height;

    float step_w = (step_w_ == 0) ? step_ : step_w_;
    float step_h = (step_h_ == 0) ? step_ : step_h_;

    if ((step_w == 0) & (step_h == 0))
    {
        step_w = img_width / INPUT0_DIMS[3];
        step_h = img_height / INPUT0_DIMS[2];
    }

    int h = get_global_id(0);
    int w = get_global_id(1);

    __global OUTPUT0_TYPE* top_data = output + h*INPUT0_DIMS[3]*num_priors_*4 + w*num_priors_*4;
    __global OUTPUT0_TYPE* top_data_var = output + OUTPUT0_DIMS[2] + h*INPUT0_DIMS[3]*num_priors_*var_size + w * num_priors_ * var_size;

    const float center_x = (w + offset_) * step_w;
    const float center_y = (h + offset_) * step_h;

    int idx = 0;
    for (int s = 0; s < num_priors_; ++s)
    {
        const float box_width = width_[s];
        const float box_height = height_[s];

        OUTPUT0_TYPE xmin = (center_x - box_width*0.5f)  * r_img_width;
        OUTPUT0_TYPE ymin = (center_y - box_height*0.5f) * r_img_height;
        OUTPUT0_TYPE xmax = (center_x + box_width*0.5f)  * r_img_width;
        OUTPUT0_TYPE ymax = (center_y + box_height*0.5f) * r_img_height;

        if (clip_)
        {
            xmin = min(max(xmin, (OUTPUT0_TYPE)(0.0f)), (OUTPUT0_TYPE)(1.0f));
            ymin = min(max(ymin, (OUTPUT0_TYPE)(0.0f)), (OUTPUT0_TYPE)(1.0f));
            xmax = min(max(xmax, (OUTPUT0_TYPE)(0.0f)), (OUTPUT0_TYPE)(1.0f));
            ymax = min(max(ymax, (OUTPUT0_TYPE)(0.0f)), (OUTPUT0_TYPE)(1.0f));
        }

        top_data[idx++] = xmin;
        top_data[idx++] = ymin;
        top_data[idx++] = xmax;
        top_data[idx++] = ymax;

        for (int i = 0; i < var_size; i++)
        {
            top_data_var[s * var_size + i] = variance_[i];
        }
    }
}
