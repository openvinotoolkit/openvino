// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/quantize.hpp"
#include "api/binary_convolution.hpp"
#include "api/scale.hpp"
#include "api/pooling.hpp"

#include "quantize_inst.h"
#include "binary_convolution_inst.h"
#include "scale_inst.h"
#include "eltwise_inst.h"
#include "data_inst.h"
#include "pass_manager.h"
#include "program_helpers.h"
#include <algorithm>
#include "to_string_utils.h"
#include "error_handler.h"

#include <string>
#include <memory>
#include <vector>

void prepare_quantization::prepare_packed_quantize(program_impl& p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto &node = (*node_itr);

        program_helpers::do_for_types<quantize>(*node, [&](quantize_node& quantize_node) {
            if (quantize_node.is_output())
                return;

            auto levels = quantize_node.get_primitive()->levels;

            program_node &input_low_node = quantize_node.get_dependency(1);
            program_node &input_high_node = quantize_node.get_dependency(2);

            if (!input_low_node.is_type<data>() || !input_high_node.is_type<data>()) {
                return;
            }

            auto &input_low = input_low_node.as<data>();
            auto &input_high = input_high_node.as<data>();

            auto &mem_input_low = input_low.get_attached_memory();
            auto &mem_input_high = input_high.get_attached_memory();

            auto output_dt = quantize_node.get_output_layout().data_type;

            if (levels == 2) {
                bool is_binarization = true;
                switch (mem_input_high.get_layout().data_type) {
                    case data_types::f32: {
                        auto data_input_low = static_cast<float*>(mem_input_low.lock());
                        auto data_input_high = static_cast<float*>(mem_input_high.lock());

                        for (size_t i = 0; i < mem_input_high.get_layout().count(); i++) {
                            if (data_input_high[i] != data_input_low[i]) {
                                is_binarization = false;
                                break;
                            }
                        }
                        break;
                    }
                    case data_types::f16: {
                        auto data_input_low = static_cast<uint16_t*>(mem_input_low.lock());
                        auto data_input_high = static_cast<uint16_t*>(mem_input_high.lock());

                        for (size_t i = 0; i < mem_input_high.get_layout().count(); i++) {
                            if (data_input_high[i] != data_input_low[i]) {
                                is_binarization = false;
                                break;
                            }
                        }
                        break;
                    }
                    default:
                        CLDNN_ERROR_MESSAGE(node->id(), "prepare_quantization: Unsupported precision of quantize inputs");
                }
                mem_input_low.unlock();
                mem_input_high.unlock();

                if (is_binarization) {
                    output_dt = data_types::bin;
                }
            }

            quantize_node.typed_desc()->output_data_type = optional_data_type{output_dt};
            quantize_node.recalc_output_layout();
        });
    }
}

void prepare_quantization::prepare_scale_shift_opt(program_impl &p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto &node = (*node_itr);

        program_helpers::do_for_types<quantize>(*node, [&](quantize_node& quantize_node) {
            auto levels = quantize_node.get_primitive()->levels;
            if (levels == 2 || levels > 256 || quantize_node.get_scale_shift_opt() || quantize_node.is_constant())
                return;

            program_node &input_low_node = quantize_node.get_dependency(1);
            program_node &input_high_node = quantize_node.get_dependency(2);
            program_node &output_low_node = quantize_node.get_dependency(3);
            program_node &output_high_node = quantize_node.get_dependency(4);

            if (!input_low_node.is_type<data>() || !input_high_node.is_type<data>() ||
                !output_low_node.is_type<data>() || !output_high_node.is_type<data>()) {
                return;
            }

            auto &input_low = input_low_node.as<data>();
            auto &input_high = input_high_node.as<data>();
            auto &output_low = output_low_node.as<data>();
            auto &output_high = output_high_node.as<data>();

            auto &mem_input_low = input_low.get_attached_memory();
            auto &mem_input_high = input_high.get_attached_memory();
            auto &mem_output_low = output_low.get_attached_memory();
            auto &mem_output_high = output_high.get_attached_memory();

            auto scales_layout = mem_input_low.get_layout();
            scales_layout.size = tensor::max(scales_layout.size, mem_input_high.get_layout().size);
            scales_layout.size = tensor::max(scales_layout.size, mem_output_low.get_layout().size);
            scales_layout.size = tensor::max(scales_layout.size, mem_output_high.get_layout().size);

            auto mem_input_scale  = p.get_engine().allocate_memory(scales_layout, mem_input_low.get_net_id(), false);
            auto mem_input_shift  = p.get_engine().allocate_memory(scales_layout, mem_input_high.get_net_id(), false);
            auto mem_output_scale = p.get_engine().allocate_memory(scales_layout, mem_output_low.get_net_id(), false);
            auto mem_output_shift = p.get_engine().allocate_memory(scales_layout, mem_output_high.get_net_id(), false);

            auto get_offset_safe = [](layout l, tensor idx) -> int {
                auto sizes = l.size;
                auto pitches = l.get_pitches();

                int offset = (idx.batch[0] % sizes.batch[0])*pitches.batch[0]
                             + (idx.feature[0] % sizes.feature[0])*pitches.feature[0]
                             + (idx.spatial[1] % sizes.spatial[1])*pitches.spatial[1]
                             + (idx.spatial[0] % sizes.spatial[0])*pitches.spatial[0];
                return offset;
            };

            bool has_negative_scales = false;
            bool need_post_scale = false;
            bool need_post_shift = false;
            bool need_pre_shift = false;
            auto out_dt = quantize_node.get_output_layout().data_type;
            bool need_clamp = levels != 256 || (out_dt != data_types::u8 && out_dt != data_types::i8);
            bool per_tensor_in_scale = true;
            bool per_tensor_in_shift = true;
            bool per_tensor_in_range = true;
            bool per_tensor_out_scale = true;
            bool per_tensor_out_shift = true;
            float in_scale_val = 0.0f;
            float in_shift_val = 0.0f;
            float out_scale_val = 0.0f;
            float out_shift_val = 0.0f;
            float in_lo_val = 0.0f;
            float in_hi_val = 0.0f;
            switch (mem_output_high.get_layout().data_type) {
                case data_types::f32: {
                    // TODO [LOW PRECISION]: Output low/high values can be removed.
                    auto data_input_low = static_cast<float*>(mem_input_low.lock());
                    auto data_input_high = static_cast<float*>(mem_input_high.lock());
                    auto data_output_low = static_cast<float*>(mem_output_low.lock());
                    auto data_output_high = static_cast<float*>(mem_output_high.lock());
                    auto data_input_scale = static_cast<float*>(mem_input_scale->lock());
                    auto data_input_shift = static_cast<float*>(mem_input_shift->lock());
                    auto data_output_scale = static_cast<float*>(mem_output_scale->lock());
                    auto data_output_shift = static_cast<float*>(mem_output_shift->lock());

                    for (int b = 0; b < scales_layout.size.batch[0]; b++) {
                        for (int f = 0; f < scales_layout.size.feature[0]; f++) {
                            for (int y = 0; y < scales_layout.size.spatial[1]; y++) {
                                for (int x = 0; x < scales_layout.size.spatial[0]; x++) {
                                    auto idx = cldnn::tensor(format::bfyx, {b, f, y, x}, 0);
                                    auto s_offset = scales_layout.get_linear_offset(idx);
                                    auto in_lo = data_input_low[get_offset_safe(mem_input_low.get_layout(), idx)];
                                    auto in_hi = data_input_high[get_offset_safe(mem_input_high.get_layout(), idx)];

                                    auto out_lo = data_output_low[get_offset_safe(mem_output_low.get_layout(), idx)];
                                    auto out_hi = data_output_high[get_offset_safe(mem_output_high.get_layout(), idx)];
                                    data_input_scale[s_offset] = (static_cast<float>(levels) - 1) / (in_hi - in_lo);
                                    data_input_shift[s_offset] = - in_lo * (static_cast<float>(levels) - 1) / (in_hi - in_lo);
                                    data_output_scale[s_offset] = (out_hi - out_lo) / (static_cast<float>(levels) - 1);
                                    data_output_shift[s_offset] = out_lo;

                                    if (data_output_scale[s_offset] != 1.0f) {
                                        need_post_scale = true;
                                    }
                                    if (data_output_shift[s_offset] != 0.0f) {
                                        need_post_shift = true;
                                    }
                                    if (data_input_scale[s_offset] < 0.0f) {
                                        has_negative_scales = true;
                                    }
                                }
                            }
                        }
                    }

                    in_scale_val = data_input_scale[0];
                    in_shift_val = data_input_shift[0];
                    out_scale_val = data_output_scale[0];
                    out_shift_val = data_output_shift[0];
                    in_lo_val = data_input_low[0];
                    in_hi_val = data_input_high[0];

                    for (size_t i = 0; i < scales_layout.count(); i++) {
                        if (in_scale_val != data_input_scale[i])
                            per_tensor_in_scale = false;
                        if (in_shift_val != data_input_shift[i])
                            per_tensor_in_shift = false;
                        if (out_scale_val != data_output_scale[i])
                            per_tensor_out_scale = false;
                        if (out_shift_val != data_output_shift[i])
                            per_tensor_out_shift = false;
                        if (data_input_shift[i] != 0.0f)
                            need_pre_shift = true;

                        if (in_lo_val != data_input_low[i % mem_input_low.get_layout().count()] ||
                            in_hi_val != data_input_high[i % mem_input_high.get_layout().count()])
                            per_tensor_in_range = false;
                    }
                    break;
                }
                case data_types::f16: {
                    auto data_input_low = static_cast<uint16_t*>(mem_input_low.lock());
                    auto data_input_high = static_cast<uint16_t*>(mem_input_high.lock());
                    auto data_output_low = static_cast<uint16_t*>(mem_output_low.lock());
                    auto data_output_high = static_cast<uint16_t*>(mem_output_high.lock());
                    auto data_input_scale = static_cast<uint16_t*>(mem_input_scale->lock());
                    auto data_input_shift = static_cast<uint16_t*>(mem_input_shift->lock());
                    auto data_output_scale = static_cast<uint16_t*>(mem_output_scale->lock());
                    auto data_output_shift = static_cast<uint16_t*>(mem_output_shift->lock());

                    for (int b = 0; b < scales_layout.size.batch[0]; b++) {
                        for (int f = 0; f < scales_layout.size.feature[0]; f++) {
                            for (int y = 0; y < scales_layout.size.spatial[1]; y++) {
                                for (int x = 0; x < scales_layout.size.spatial[0]; x++) {
                                    auto idx = cldnn::tensor(format::bfyx, {b, f, y, x}, 0);
                                    auto s_offset = scales_layout.get_linear_offset(idx);
                                    auto in_lo = half_to_float(data_input_low[get_offset_safe(mem_input_low.get_layout(), idx)]);
                                    auto in_hi = half_to_float(data_input_high[get_offset_safe(mem_input_high.get_layout(), idx)]);

                                    auto out_lo = half_to_float(data_output_low[get_offset_safe(mem_output_low.get_layout(), idx)]);
                                    auto out_hi = half_to_float(data_output_high[get_offset_safe(mem_output_high.get_layout(), idx)]);
                                    data_input_scale[s_offset] = float_to_half((static_cast<float>(levels) - 1) / (in_hi - in_lo));
                                    data_input_shift[s_offset] = float_to_half(- in_lo * (static_cast<float>(levels) - 1) / (in_hi - in_lo));
                                    data_output_scale[s_offset] = float_to_half((out_hi - out_lo) / (static_cast<float>(levels) - 1));
                                    data_output_shift[s_offset] = float_to_half(out_lo);

                                    if (half_to_float(data_output_scale[s_offset]) != 1.0f) {
                                        need_post_scale = true;
                                    }
                                    if (half_to_float(data_output_shift[s_offset]) != 0.0f) {
                                        need_post_shift = true;
                                    }
                                    if (half_to_float(data_input_scale[s_offset]) < 0.0f) {
                                        has_negative_scales = true;
                                    }
                                }
                            }
                        }
                    }

                    in_scale_val = half_to_float(data_input_scale[0]);
                    in_shift_val = half_to_float(data_input_shift[0]);
                    out_scale_val = half_to_float(data_output_scale[0]);
                    out_shift_val = half_to_float(data_output_shift[0]);
                    in_lo_val = half_to_float(data_input_low[0]);
                    in_hi_val = half_to_float(data_input_high[0]);
                    for (size_t i = 0; i < scales_layout.count(); i++) {
                        if (in_scale_val != half_to_float(data_input_scale[i]))
                            per_tensor_in_scale = false;
                        if (in_shift_val != half_to_float(data_input_shift[i]))
                            per_tensor_in_shift = false;
                        if (out_scale_val != half_to_float(data_output_scale[i]))
                            per_tensor_out_scale = false;
                        if (out_shift_val != half_to_float(data_output_shift[i]))
                            per_tensor_out_shift = false;
                        if (half_to_float(data_input_shift[i]) != 0.0f)
                            need_pre_shift = true;

                        if (in_lo_val != half_to_float(data_input_low[i % mem_input_low.get_layout().count()]) ||
                            in_hi_val != half_to_float(data_input_high[i % mem_input_high.get_layout().count()]))
                            per_tensor_in_range = false;
                    }
                    break;
                }
                default:
                    throw std::runtime_error("prepare_quantization: Unsupported precision of quantize output values");
            }

            if (has_negative_scales) {
                return;
            }

            layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));
            float zero = 0.f;
            auto in_scale_prim = std::make_shared<data>(quantize_node.id() + "_in_scale", memory::attach(dummy_layout, &zero, 1));
            auto in_shift_prim = std::make_shared<data>(quantize_node.id() + "_in_shift", memory::attach(dummy_layout, &zero, 1));
            auto out_scale_prim = std::make_shared<data>(quantize_node.id() + "_output_scale", memory::attach(dummy_layout, &zero, 1));
            auto out_shift_prim = std::make_shared<data>(quantize_node.id() + "_output_shift", memory::attach(dummy_layout, &zero, 1));
            auto& in_scale_node = p.get_or_create(in_scale_prim);
            auto& in_shift_node = p.get_or_create(in_shift_prim);
            auto& out_scale_node = p.get_or_create(out_scale_prim);
            auto& out_shift_node = p.get_or_create(out_shift_prim);

            in_scale_node.as<data>().attach_memory(*mem_input_scale);
            in_shift_node.as<data>().attach_memory(*mem_input_shift);
            out_scale_node.as<data>().attach_memory(*mem_output_scale);
            out_shift_node.as<data>().attach_memory(*mem_output_shift);

            auto& inputs = p.get_inputs();

            inputs.push_back(&in_scale_node);
            inputs.push_back(&in_shift_node);
            inputs.push_back(&out_scale_node);
            inputs.push_back(&out_shift_node);

            p.add_connection(in_scale_node, quantize_node);
            p.add_connection(in_shift_node, quantize_node);
            p.add_connection(out_scale_node, quantize_node);
            p.add_connection(out_shift_node, quantize_node);
            quantize_node.add_memory_dependency(in_scale_node.id());
            quantize_node.add_memory_dependency(in_shift_node.id());
            quantize_node.add_memory_dependency(out_scale_node.id());
            quantize_node.add_memory_dependency(out_shift_node.id());
            p.get_processing_order().insert(&quantize_node, &in_shift_node);
            p.get_processing_order().insert(&quantize_node, &in_scale_node);
            p.get_processing_order().insert(&quantize_node, &out_shift_node);
            p.get_processing_order().insert(&quantize_node, &out_scale_node);

            quantize_node.set_scale_shift_opt();

            if (need_post_scale) {
                quantize_node.set_need_post_scale();
            }

            if (need_post_shift) {
                quantize_node.set_need_post_shift();
            }

            if (need_pre_shift) {
                quantize_node.set_need_pre_shift();
            }

            if (per_tensor_in_scale) {
                quantize_node.set_per_tensor_input_scale();
                quantize_node.set_input_scale_val(in_scale_val);
            }

            if (per_tensor_in_shift && need_pre_shift) {
                quantize_node.set_per_tensor_input_shift();
                quantize_node.set_input_shift_val(in_shift_val);
            }

            if (need_clamp) {
                quantize_node.set_need_clamp();
            }

            if (per_tensor_in_range) {
                quantize_node.set_per_tensor_input_range();
                quantize_node.set_input_lo_val(in_lo_val);
                quantize_node.set_input_hi_val(in_hi_val);
            }
            if (per_tensor_out_scale) {
                quantize_node.set_per_tensor_output_scale();
                quantize_node.set_output_scale_val(out_scale_val);
            }

            if (per_tensor_out_shift) {
                quantize_node.set_per_tensor_output_shift();
                quantize_node.set_output_shift_val(out_shift_val);
            }

            mem_input_low.unlock();
            mem_input_high.unlock();
            mem_output_low.unlock();
            mem_output_high.unlock();
            mem_input_scale->unlock();
            mem_input_shift->unlock();
            mem_output_scale->unlock();
            mem_output_shift->unlock();
        });
    }
}

void prepare_quantization::remove_fake_reorders(program_impl& p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto &node = (*node_itr);

        if (!node->is_type<reorder>() || !node->is_in_data_flow())
            continue;

        if (node->get_users().size() != 1 || node->get_dependencies().size() != 1)
            continue;

        auto &usr = node->get_users().front();
        auto &dep = node->get_dependency(0);
        if (!(usr->is_type<convolution>() && usr->get_dependency(1).get_output_layout().data_type == data_types::i8) ||
            !dep.is_input() ||
            dep.get_output_layout().data_type != data_types::u8 ||
            (node->get_output_layout().data_type != data_types::f32 && node->get_output_layout().data_type != data_types::f16) ||
            dep.get_output_layout().format != node->get_output_layout().format ||
            dep.get_output_layout().size != node->get_output_layout().size)
            continue;

        p.replace_all_usages(*node, dep);
        p.add_optimized_primitive_info(node->id());
        p.remove_all_connections(*node);
        p.remove_if_dangling(*node);
    }
}

template<typename W_T, typename AZP_T>
void fill_compensation_typed(W_T* w, AZP_T* azp, W_T* wzp, float* comp, int GS, int OC, int IC, int KS) {
    for (int g = 0; g < GS; g++) {
        for (int oc = 0; oc < OC; oc++) {
            float c = 0.f;
            for (int ic = 0; ic < IC; ic++) {
                for (int k = 0; k < KS; k++) {
                    // zero points don't depend on group size and in case of per-channel zp
                    // we have groups * feature_maps_in_group elements in the buffer
                    int azp_idx = g*IC + ic;
                    int wzp_idx = g*OC + oc;

                    c += w[g*OC*IC*KS + oc*IC*KS + ic*KS + k] * azp[azp_idx];
                    if (wzp) {
                        c -= azp[azp_idx] * wzp[wzp_idx];
                    }
                }
            }

            comp[g*OC + oc] = -c;
        }
    }
}

void prepare_quantization::prepare_asymmetric_quantization(program_impl &p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        // Detects if given eltwise node performs zero point subtraction
        auto is_zero_point_node = [](eltwise_node& node) -> bool {
            auto prim = node.get_primitive();

            if (node.get_dependencies().size() != 2 || prim->mode != eltwise_mode::sub)
                return false;

            if (node.get_users().size() != 1)
                return false;

            auto in0_layout = node.get_dependency(0).get_output_layout();
            auto in1_layout = node.get_dependency(1).get_output_layout();

            if (!node.get_dependency(1).is_type<data>())
                return false;

            // Check if sub inputs are quantized
            if (in0_layout.data_type != data_types::u8 && in0_layout.data_type != data_types::i8)
                return false;

            // Zero point must have the same type as quantized data
            if (in0_layout.data_type != in1_layout.data_type)
                return false;

            return true;
        };

        auto fill_compensation = [&](int groups, memory_impl* w, memory_impl* azp, memory_impl* wzp,
                                     memory_impl::ptr compensation) {
            auto wl = w->get_layout();

            int GS = groups;
            int OC = wl.size.batch[0] / GS;
            int IC = wl.size.feature[0];  // already divided by GS
            int KS = wl.size.spatial[0]*wl.size.spatial[1]*wl.size.spatial[2];

            auto w_dt = wl.data_type;
            auto azp_dt = azp->get_layout().data_type;

            mem_lock<float> comp_lock{compensation};

            if (w_dt == data_types::u8 && azp_dt == data_types::u8) {
                mem_lock<uint8_t> w_lock(*w);
                mem_lock<uint8_t> azp_lock(*azp);
                if (wzp) {
                    mem_lock<uint8_t> wzp_lock(*wzp);
                    fill_compensation_typed(w_lock.data(), azp_lock.data(), wzp_lock.data(), comp_lock.data(), GS, OC, IC, KS);
                } else {
                    fill_compensation_typed(w_lock.data(), azp_lock.data(), static_cast<uint8_t*>(nullptr), comp_lock.data(), GS, OC, IC, KS);
                }
            } else if (w_dt == data_types::i8 && azp_dt == data_types::u8) {
                mem_lock<int8_t> w_lock(*w);
                mem_lock<uint8_t> azp_lock(*azp);
                if (wzp) {
                    mem_lock<int8_t> wzp_lock(*wzp);
                    fill_compensation_typed(w_lock.data(), azp_lock.data(), wzp_lock.data(), comp_lock.data(), GS, OC, IC, KS);
                } else {
                    fill_compensation_typed(w_lock.data(), azp_lock.data(), static_cast<int8_t*>(nullptr), comp_lock.data(), GS, OC, IC, KS);
                }
            } else if (w_dt == data_types::i8 && azp_dt == data_types::i8) {
                mem_lock<int8_t> w_lock(*w);
                mem_lock<int8_t> azp_lock(*azp);
                if (wzp) {
                    mem_lock<int8_t> wzp_lock(*wzp);
                    fill_compensation_typed(w_lock.data(), azp_lock.data(), wzp_lock.data(), comp_lock.data(), GS, OC, IC, KS);
                } else {
                    fill_compensation_typed(w_lock.data(), azp_lock.data(), static_cast<int8_t*>(nullptr), comp_lock.data(), GS, OC, IC, KS);
                }
            } else if (w_dt == data_types::u8 && azp_dt == data_types::i8) {
                mem_lock<uint8_t> w_lock(*w);
                mem_lock<int8_t> azp_lock(*azp);
                if (wzp) {
                    mem_lock<uint8_t> wzp_lock(*wzp);
                    fill_compensation_typed(w_lock.data(), azp_lock.data(), wzp_lock.data(), comp_lock.data(), GS, OC, IC, KS);
                } else {
                    fill_compensation_typed(w_lock.data(), azp_lock.data(), static_cast<uint8_t*>(nullptr), comp_lock.data(), GS, OC, IC, KS);
                }
            }
        };

        auto asymmetric_convolution_f = [&](convolution_node& convolution_node) {
            auto& in0 = convolution_node.get_dependency(0);
            auto& in1 = convolution_node.get_dependency(1);

            bool asymmetric_data = in0.is_type<eltwise>() && is_zero_point_node(in0.as<eltwise>());
            bool asymmetric_weights = in1.is_type<eltwise>() && is_zero_point_node(in1.as<eltwise>());

            if (!asymmetric_data && !asymmetric_weights)
                return;

            // Input that doesn't match asymmetric pattern should be quantized
            // Cases like fp32 input + i8 weights + i8 w_zp are not supported
            if (!asymmetric_data &&
                in0.get_output_layout().data_type != data_types::u8 &&
                in0.get_output_layout().data_type != data_types::i8)
                return;

            if (!asymmetric_weights &&
                in1.get_output_layout().data_type != data_types::u8 &&
                in1.get_output_layout().data_type != data_types::i8)
                return;

            auto old_conv_prim = convolution_node.get_primitive();

            // Split is not supported
            if (old_conv_prim->weights.size() > 1)
                return;


            primitive_id input = old_conv_prim->input[0];
            std::vector<primitive_id> weights = old_conv_prim->weights;
            std::vector<primitive_id> w_zero_points = {};
            std::vector<primitive_id> a_zero_points = {};
            std::vector<primitive_id> compensation = {};

            cldnn::program_node* new_input = &in0;
            cldnn::program_node* new_weights = &in1;
            cldnn::program_node* new_bias = !old_conv_prim->bias.empty() ? &convolution_node.get_dependency(2) : nullptr;
            cldnn::program_node* new_a_zp = nullptr;
            cldnn::program_node* new_w_zp = nullptr;
            cldnn::program_node* new_compenstation = nullptr;

            bool need_compensation = false;

            auto output_size = convolution_node.get_output_layout().size;
            int ofm = in1.get_output_layout().size.batch[0];
            int ifm = in0.get_output_layout().size.feature[0];
            int ofm_aligned = ((ofm + 31) / 32) * 32;
            int ifm_aligned = ((ifm + 31) / 32) * 32;
            int groups = static_cast<int>(convolution_node.get_groups());

            if (asymmetric_data) {
                new_input = &in0.get_dependency(0);
                new_a_zp = &in0.get_dependency(1);

                auto l = layout{new_a_zp->get_output_layout().data_type, format::bfyx, tensor{1, ifm_aligned, 1, 1}};
                int s = new_a_zp->get_output_layout().size.feature[0];
                auto azp_aligned = p.get_engine().allocate_memory(l, 0, false);
                mem_lock<int8_t> new_data{azp_aligned};
                mem_lock<int8_t> old_data{new_a_zp->as<data>().get_attached_memory()};
                for (int i = 0; i < ifm_aligned; i++) {
                    new_data.data()[i] = old_data.data()[i % s];
                }
                new_a_zp->as<data>().attach_memory(*azp_aligned);

                input = new_input->id();
                a_zero_points.push_back(new_a_zp->id());
                need_compensation = true;
            }

            if (asymmetric_weights) {
                new_weights = &in1.get_dependency(0);
                new_w_zp = &in1.get_dependency(1);

                auto l = layout{new_w_zp->get_output_layout().data_type, format::bfyx, tensor{ofm_aligned, 1, 1, 1}};
                int s = new_w_zp->get_output_layout().size.batch[0];
                auto wzp_aligned = p.get_engine().allocate_memory(l, 0, false);
                mem_lock<int8_t> new_data{wzp_aligned};
                mem_lock<int8_t> old_data{new_w_zp->as<data>().get_attached_memory()};
                for (int i = 0; i < ofm_aligned; i++) {
                    new_data.data()[i] = old_data.data()[i % s];
                }
                new_w_zp->as<data>().attach_memory(*wzp_aligned);

                weights = { new_weights->id() };
                w_zero_points.push_back(new_w_zp->id());
            }

            if (need_compensation) {
                auto l = layout{data_types::f32, format::bfyx, tensor{1, ofm_aligned, 1, 1}};
                auto data_to_allocate = p.get_engine().allocate_memory(l, 0, false);
                auto w = &new_weights->as<data>().get_attached_memory();
                auto azp = asymmetric_data ? &new_a_zp->as<data>().get_attached_memory() : nullptr;
                auto wzp = asymmetric_weights ? &new_w_zp->as<data>().get_attached_memory() : nullptr;
                fill_compensation(groups, w, azp, wzp, data_to_allocate);
                layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));
                float zero = 0.f;

                auto compensation_prim = std::make_shared<data>(convolution_node.id() + "_compensation", memory::attach(dummy_layout, &zero, 1));
                new_compenstation = &p.get_or_create(compensation_prim);
                p.get_inputs().push_back(new_compenstation);
                compensation.push_back(new_compenstation->id());
                new_compenstation->as<data>().attach_memory(*data_to_allocate);
            }

            // Collect dependencies of a new convolution node
            std::vector<program_node*> dependencies = {new_input, new_weights};
            if (new_bias)
                dependencies.push_back(new_bias);
            if (new_w_zp)
                dependencies.push_back(new_w_zp);
            if (new_a_zp)
                dependencies.push_back(new_a_zp);
            if (new_compenstation)
                dependencies.push_back(new_compenstation);

            auto new_conv_prim = std::make_shared<convolution>(
                        convolution_node.id() + "_asymmetric",
                        input,
                        weights,
                        old_conv_prim->bias,
                        w_zero_points,
                        a_zero_points,
                        compensation,
                        old_conv_prim->groups,
                        *old_conv_prim->output_data_type,
                        old_conv_prim->stride,
                        old_conv_prim->input_offset,
                        old_conv_prim->dilation,
                        output_size,
                        old_conv_prim->grouped_weights_shape,
                        old_conv_prim->output_padding);

            auto& new_conv_node = p.get_or_create(new_conv_prim);

            // Replace old conv node with the new one. New node has correct users, but dependencies don't match primitive parameters,
            // so replace it with the vector collected earlier
            p.replace(convolution_node, new_conv_node);
            if (need_compensation) {
                p.get_processing_order().insert(&new_conv_node, new_compenstation);
                new_compenstation->users.push_back(&new_conv_node);
            }
            new_conv_node.dependencies = dependencies;

            // Remove sub operations from the graph and set correct users for zero points and inputs
            if (asymmetric_data) {
                if (!new_a_zp || !new_input)
                    CLDNN_ERROR_MESSAGE(new_conv_node.id(), "Unexpected nullptr in asymmetric quantization for activations optimization");

                auto& zp_users = new_a_zp->users;
                auto& in_users = new_input->users;
                // Erase sub node from input and zero point users...
                zp_users.erase(std::remove(zp_users.begin(), zp_users.end(), &in0), zp_users.end());
                in_users.erase(std::remove(in_users.begin(), in_users.end(), &in0), in_users.end());

                // ...because now the user is new convolution node
                new_a_zp->users.push_back(&new_conv_node);
                new_input->users.push_back(&new_conv_node);

                p.add_optimized_primitive_info(in0.id(), {new_conv_node.id()});

                // Remove sub node on activations
                in0.dependencies.clear();
                in0.users.clear();
                p.remove_if_dangling(in0);
            }

            if (asymmetric_weights) {
                if (!new_w_zp || !new_weights)
                    CLDNN_ERROR_MESSAGE(new_conv_node.id(), "Unexpected nullptr in asymmetric quantization for weights optimization");

                auto& zp_users = new_w_zp->users;
                auto& wei_users = new_weights->users;
                // Erase sub node from weights and zero point users...
                zp_users.erase(std::remove(zp_users.begin(), zp_users.end(), &in1), zp_users.end());
                wei_users.erase(std::remove(wei_users.begin(), wei_users.end(), &in1), wei_users.end());

                // ...because now the user is new convolution node
                new_weights->users.push_back(&new_conv_node);
                new_w_zp->users.push_back(&new_conv_node);

                p.add_optimized_primitive_info(in1.id(), {new_conv_node.id()});

                // Remove sub node on weights
                in1.dependencies.clear();
                in1.users.clear();
                p.remove_if_dangling(in1);
            }

            new_conv_node.recalc_output_layout();
        };

        program_helpers::do_for_types<convolution>(*node,
                                                    asymmetric_convolution_f);
    }
}

void prepare_quantization::prepare_dequantize_merge(program_impl &p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto &node = (*itr++);

        if (node->is_output())
            continue;

        program_helpers::do_for_types<eltwise>(*node, [&p](eltwise_node& node) {
            for (size_t i = 1; i < node.get_dependencies().size(); i++) {
                if (!node.get_dependency(i).is_type<data>()) {
                    return;
                }
            }

            auto get_scale_shift_mem = [](const eltwise_node& eltw, size_t dep_id) -> memory_impl& {
                if (dep_id >= eltw.get_dependencies().size())
                    CLDNN_ERROR_MESSAGE(eltw.id(), "Invalid dependency id in dequantize optimization");

                return eltw.get_dependency(dep_id).as<data>().get_attached_memory();
            };

            auto eltw_mode = node.get_primitive()->mode;
            if (eltw_mode != eltwise_mode::sum && eltw_mode != eltwise_mode::prod)
                return;

            auto& input = node.input();

            for (auto& user : input.get_users()) {
                if (user == &node)
                    continue;

                if (!user->is_type<eltwise>() || user->get_dependencies().size() != node.get_dependencies().size())
                    continue;

                auto& eltwise_dep = user->as<eltwise>();
                if (eltwise_dep.get_primitive()->mode != node.get_primitive()->mode)
                    continue;

                bool valid_scale_node = true;
                for (size_t i = 1; i < eltwise_dep.get_dependencies().size(); i++) {
                    if (!eltwise_dep.get_dependency(i).is_type<data>()) {
                        valid_scale_node = false;
                    }
                }

                if (!valid_scale_node)
                    continue;

                bool same_params = true;
                for (size_t i = 1; i < node.get_dependencies().size(); i++) {
                    auto& mem0 = get_scale_shift_mem(eltwise_dep, i);
                    auto& mem1 = get_scale_shift_mem(node, i);

                    auto ptr0 = static_cast<uint8_t*>(mem0.lock());
                    auto ptr1 = static_cast<uint8_t*>(mem1.lock());

                    for (size_t j = 0; j < mem0.get_layout().bytes_count(); j++) {
                        if (ptr0[j] != ptr1[j]) {
                            same_params = false;
                            break;
                        }
                    }
                    mem0.unlock();
                    mem1.unlock();
                }

                if (same_params) {
                    while (!node.get_dependencies().empty()) {
                        auto& dep = node.get_dependency(0);
                        p.remove_connection(dep, node);
                        p.remove_if_dangling(dep);
                    }
                    p.add_optimized_primitive_info(node.id(), {user->id()});
                    p.replace_all_usages(node, *user);
                }
            }
        });
    }
}

void prepare_quantization::run(program_impl& p) {
    prepare_packed_quantize(p);
    prepare_scale_shift_opt(p);
    prepare_dequantize_merge(p);
    remove_fake_reorders(p);
    prepare_asymmetric_quantization(p);
}
