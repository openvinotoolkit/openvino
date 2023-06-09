// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <oneapi/dnnl/dnnl.hpp>
#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/depth_to_space.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "activation_inst.h"
#include "reorder_inst.h"
#include "eltwise_inst.h"
#include "quantize_inst.h"

namespace cldnn {

    void fused_primitive_desc::save(cldnn::BinaryOutputBuffer& ob) const {
        // ob << desc;
        ob << input_layout;
        ob << output_layout;

        if (f_param->type() == activation::type_id()) {
            ob << std::string("activation");
            auto casted = std::dynamic_pointer_cast<ActivationFuseParams>(f_param);
            ob << casted->_desc;
        } else if (f_param->type() == depth_to_space::type_id()) {
            ob << std::string("depth_to_space");
        } else if (f_param->type() == reorder::type_id()) {
            ob << std::string("reorder");
            auto casted = std::dynamic_pointer_cast<ReorderFuseParams>(f_param);
            ob << casted->_in;
            ob << casted->_out;
        } else if (f_param->type() == eltwise::type_id()) {
            ob << std::string("eltwise");
            auto casted = std::dynamic_pointer_cast<EltwiseFuseParams>(f_param);
            ob << casted->_desc;
        } else if (f_param->type() == quantize::type_id()) {
            ob << std::string("quantize");
            auto casted = std::dynamic_pointer_cast<QuantizeFuseParams>(f_param);
            ob << casted->_out_layout;
            ob << casted->_scale_shift_opt;
            ob << casted->_need_post_scale;
            ob << casted->_need_post_shift;
            ob << casted->_need_pre_shift;
            ob << casted->_need_clamp;
            ob << casted->_need_min_clamp;
            ob << casted->_need_max_clamp;
            ob << casted->_per_tensor_input_range;
            ob << casted->_per_tensor_input_scale;
            ob << casted->_per_tensor_input_shift;
            ob << casted->_per_tensor_output_range;
            ob << casted->_per_tensor_output_scale;
            ob << casted->_per_tensor_output_shift;
            ob << casted->_in_lo;
            ob << casted->_in_hi;
            ob << casted->_in_scale;
            ob << casted->_in_shift;
            ob << casted->_out_lo;
            ob << casted->_out_hi;
            ob << casted->_out_scale;
            ob << casted->_out_shift;
        } else {
            OPENVINO_THROW("[GPU] Unknown type of NodeFuseParams");
        }

        ob << deps.size();
        for (auto& dep : deps) {
            ob << dep.first;
            ob << dep.second;
        }
        ob << fused_deps;
        ob << outer_dep_start_idx;
        ob << total_num_deps;
    }

    void fused_primitive_desc::load(cldnn::BinaryInputBuffer& ib) {
        // ib >> desc;
        ib >> input_layout;
        ib >> output_layout;
        std::string f_param_type;
        ib >> f_param_type;
        if (f_param_type.compare("activation") == 0) {
            std::shared_ptr<activation> desc;
            ib >> desc;
            f_param = std::make_shared<ActivationFuseParams>(desc);
        } else if (f_param_type.compare("depth_to_space") == 0) {
            f_param = std::make_shared<NodeFuseParams>(depth_to_space::type_id());
        } else if (f_param_type.compare("reorder") == 0) {
            layout in, out;
            ib >> in;
            ib >> out;
            f_param = std::make_shared<ReorderFuseParams>(in, out);
        } else if (f_param_type.compare("eltwise") == 0) {
            std::shared_ptr<eltwise> desc;
            ib >> desc;
            f_param = std::make_shared<EltwiseFuseParams>(desc);
        } else if (f_param_type.compare("quantize") == 0) {
            layout out_layout;
            bool scale_shift_opt;
            bool need_post_scale;
            bool need_post_shift;
            bool need_pre_shift;
            bool need_clamp;
            bool need_min_clamp;
            bool need_max_clamp;
            bool per_tensor_input_range;
            bool per_tensor_input_scale;
            bool per_tensor_input_shift;
            bool per_tensor_output_range;
            bool per_tensor_output_scale;
            bool per_tensor_output_shift;
            float in_lo;
            float in_hi;
            float in_scale;
            float in_shift;
            float out_lo;
            float out_hi;
            float out_scale;
            float out_shift;

            ib >> out_layout;
            ib >> scale_shift_opt;
            ib >> need_post_scale;
            ib >> need_post_shift;
            ib >> need_pre_shift;
            ib >> need_clamp;
            ib >> need_min_clamp;
            ib >> need_max_clamp;
            ib >> per_tensor_input_range;
            ib >> per_tensor_input_scale;
            ib >> per_tensor_input_shift;
            ib >> per_tensor_output_range;
            ib >> per_tensor_output_scale;
            ib >> per_tensor_output_shift;
            ib >> in_lo;
            ib >> in_hi;
            ib >> in_scale;
            ib >> in_shift;
            ib >> out_lo;
            ib >> out_hi;
            ib >> out_scale;
            ib >> out_shift;

            f_param = std::make_shared<QuantizeFuseParams>(out_layout,
                        scale_shift_opt, need_post_scale, need_post_shift, need_pre_shift,
                        need_clamp, need_min_clamp, need_max_clamp,
                        per_tensor_input_range, per_tensor_input_scale, per_tensor_input_shift,
                        per_tensor_output_range, per_tensor_output_scale, per_tensor_output_shift,
                        in_lo, in_hi, in_scale, in_shift, out_lo, out_hi, out_scale, out_shift);
        } else {
            OPENVINO_THROW("[GPU] Unknown type of NodeFuseParams");
        }

        size_t deps_size;
        ib >> deps_size;
        for (size_t i = 0; i < deps_size; ++i) {
            primitive_id prim_id;
            size_t idx;
            ib >> prim_id;
            ib >> idx;
            deps.emplace_back(std::make_pair(prim_id, idx));
        }
        ib >> fused_deps;
        ib >> outer_dep_start_idx;
        ib >> total_num_deps;
    }


}  // namespace cldnn
