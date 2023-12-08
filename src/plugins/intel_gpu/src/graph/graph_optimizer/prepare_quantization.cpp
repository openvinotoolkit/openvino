// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_inst.h"
#include "pooling_inst.h"
#include "quantize_inst.h"
#include "reorder_inst.h"
#include "eltwise_inst.h"
#include "data_inst.h"
#include "pass_manager.h"
#include "program_helpers.h"
#include "to_string_utils.h"

#include <algorithm>
#include <string>
#include <memory>
#include <vector>

using namespace cldnn;

namespace {

inline float clamp(float val) {
    return std::max(std::numeric_limits<float>::lowest(), std::min(std::numeric_limits<float>::max(), val));
}

}  // namespace


void prepare_quantization::prepare_scale_shift_opt(program &p, quantize_node& quantize_node) {
    const auto& stream = p.get_stream();

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

    auto mem_input_low = input_low.get_attached_memory_ptr();
    auto mem_input_high = input_high.get_attached_memory_ptr();
    auto mem_output_low = output_low.get_attached_memory_ptr();
    auto mem_output_high = output_high.get_attached_memory_ptr();

    auto scales_layout = mem_input_low->get_layout();
    auto max_size = tensor(0);
    max_size = tensor::max(max_size, mem_input_high->get_layout().get_tensor());
    max_size = tensor::max(max_size, mem_output_low->get_layout().get_tensor());
    max_size = tensor::max(max_size, mem_output_high->get_layout().get_tensor());

    scales_layout.set_tensor(max_size);

    auto mem_input_scale  = p.get_engine().allocate_memory(scales_layout, false);
    auto mem_input_shift  = p.get_engine().allocate_memory(scales_layout, false);
    auto mem_output_scale = p.get_engine().allocate_memory(scales_layout, false);
    auto mem_output_shift = p.get_engine().allocate_memory(scales_layout, false);

    auto get_offset_safe = [](const layout& l, const tensor& idx) -> int {
        auto sizes = l.get_tensor();
        auto pitches = l.get_pitches();

        return (idx.batch[0] % sizes.batch[0])*pitches.batch[0]
                        + (idx.feature[0] % sizes.feature[0])*pitches.feature[0]
                        + (idx.spatial[1] % sizes.spatial[1])*pitches.spatial[1]
                        + (idx.spatial[0] % sizes.spatial[0])*pitches.spatial[0];
    };

    auto lock_memory = [&stream] (memory::ptr memory, std::function<void(std::size_t, float)>& set_data,
                                  std::function<float(size_t)>& get_data) {
        using float_mem_lock = mem_lock<float, mem_lock_type::write>;
        using float16_mem_lock = mem_lock<ov::float16, mem_lock_type::write>;
        switch (memory->get_layout().data_type) {
            case data_types::f32: {
                std::shared_ptr<float_mem_lock> data_lock_ptr = std::make_shared<float_mem_lock>(memory, stream);
                float* data = data_lock_ptr->data();
                set_data = [data] (size_t idx, float value) {
                    data[idx] = value;
                };
                get_data = [data] (size_t idx) {
                    return data[idx];
                };
                return std::pair<std::shared_ptr<float_mem_lock>, std::shared_ptr<float16_mem_lock>>(data_lock_ptr, nullptr);
            }
            case data_types::f16: {
                std::shared_ptr<float16_mem_lock> data_lock_ptr = std::make_shared<float16_mem_lock>(memory, stream);
                ov::float16* data = data_lock_ptr->data();
                set_data = [data] (size_t idx, float value) {
                    data[idx] = ov::float16(value);
                };
                get_data = [data] (size_t idx) {
                    return static_cast<float>(data[idx]);
                };
                return std::pair<std::shared_ptr<float_mem_lock>, std::shared_ptr<float16_mem_lock>>(nullptr, data_lock_ptr);
            }
            default:
                throw std::runtime_error("prepare_quantization: Unsupported precision of quantize output values");
        }
    };

    std::function<void(size_t, float)> set_data_input_low;
    std::function<float(size_t)> get_data_input_low;
    auto input_low_locked_memory = lock_memory(mem_input_low, set_data_input_low, get_data_input_low);

    std::function<void(size_t, float)> set_data_input_high;
    std::function<float(size_t)> get_data_input_high;
    auto input_high_locked_memory = lock_memory(mem_input_high, set_data_input_high, get_data_input_high);

    std::function<void(size_t, float)> set_data_output_low;
    std::function<float(size_t)> get_data_output_low;
    auto output_low_locked_memory = lock_memory(mem_output_low, set_data_output_low, get_data_output_low);

    std::function<void(size_t, float)> set_data_output_high;
    std::function<float(size_t)> get_data_output_high;
    auto output_high_locked_memory = lock_memory(mem_output_high, set_data_output_high, get_data_output_high);

    std::function<void(std::size_t, float)> set_data_input_scale;
    std::function<float(size_t)> get_data_input_scale;
    auto input_scale_locked_memory = lock_memory(mem_input_scale, set_data_input_scale, get_data_input_scale);

    std::function<void(size_t, float)> set_data_input_shift;
    std::function<float(size_t)> get_data_input_shift;
    auto input_shift_locked_memory = lock_memory(mem_input_shift, set_data_input_shift, get_data_input_shift);

    std::function<void(size_t, float)> set_data_output_scale;
    std::function<float(size_t)> get_data_output_scale;
    auto output_scale_locked_memory = lock_memory(mem_output_scale, set_data_output_scale, get_data_output_scale);

    std::function<void(size_t, float)> set_data_output_shift;
    std::function<float(size_t)> get_data_output_shift;
    auto output_shift_locked_memory = lock_memory(mem_output_shift, set_data_output_shift, get_data_output_shift);

    bool has_negative_scales = false;
    bool need_post_scale = false;
    bool need_post_shift = false;
    auto primitive = quantize_node.get_primitive();
    int levels = primitive->levels;

    for (int b = 0; b < scales_layout.batch(); b++) {
        for (int f = 0; f < scales_layout.feature(); f++) {
            for (int y = 0; y < scales_layout.spatial(1); y++) {
                for (int x = 0; x < scales_layout.spatial(0); x++) {
                    auto idx = cldnn::tensor(format::bfyx, {b, f, y, x}, 0);
                    auto s_offset = scales_layout.get_linear_offset(idx);
                    float in_lo = get_data_input_low(get_offset_safe(mem_input_low->get_layout(), idx));
                    float in_hi = get_data_input_high(get_offset_safe(mem_input_high->get_layout(), idx));

                    float out_lo = get_data_output_low(get_offset_safe(mem_output_low->get_layout(), idx));
                    float out_hi = get_data_output_high(get_offset_safe(mem_output_high->get_layout(), idx));
                    float in_shift_basic = (static_cast<float>(levels) - 1.f) / (in_hi - in_lo);
                    set_data_input_scale(s_offset, in_shift_basic);
                    set_data_input_shift(s_offset, -in_lo * in_shift_basic);
                    set_data_output_scale(s_offset, (out_hi - out_lo) / (static_cast<float>(levels) - 1.f));
                    set_data_output_shift(s_offset, out_lo);

                    if (get_data_output_scale(s_offset) != 1.0f) {
                        need_post_scale = true;
                    }
                    if (get_data_output_shift(s_offset) != 0.0f) {
                        need_post_shift = true;
                    }
                    if (get_data_input_scale(s_offset) < 0.0f) {
                        has_negative_scales = true;
                    }
                }
            }
        }
    }

    bool need_pre_shift = false;
    bool per_tensor_in_scale = true;
    bool per_tensor_in_shift = true;
    bool per_tensor_in_range = true;
    bool per_tensor_out_scale = true;
    bool per_tensor_out_shift = true;
    bool per_tensor_out_range = true;
    float in_scale_val = get_data_input_scale(0);
    float in_shift_val = get_data_input_shift(0);
    float out_scale_val = get_data_output_scale(0);
    float out_shift_val = get_data_output_shift(0);
    float in_lo_val = get_data_input_low(0);
    float in_hi_val = get_data_input_high(0);
    float out_lo_val = get_data_output_low(0);
    float out_hi_val = get_data_output_high(0);
    for (size_t i = 0; i < scales_layout.count(); i++) {
        if (in_scale_val != get_data_input_scale(i))
            per_tensor_in_scale = false;
        if (in_shift_val != get_data_input_shift(i))
            per_tensor_in_shift = false;
        if (out_scale_val != get_data_output_scale(i))
            per_tensor_out_scale = false;
        if (out_shift_val != get_data_output_shift(i))
            per_tensor_out_shift = false;
        if (get_data_input_shift(i) != 0.0f)
            need_pre_shift = true;

        if (in_lo_val != get_data_input_low(i % mem_input_low->get_layout().count()) ||
            in_hi_val != get_data_input_high(i % mem_input_high->get_layout().count()))
            per_tensor_in_range = false;
        if (out_lo_val != get_data_output_low(i % mem_output_low->get_layout().count()) ||
            out_hi_val != get_data_output_high(i % mem_output_high->get_layout().count()))
            per_tensor_out_range = false;
    }

    auto out_is_int8 = quantize_node.get_output_layout().data_type == data_types::i8;
    auto out_is_uint8 = quantize_node.get_output_layout().data_type == data_types::u8;
    auto out_is_fp = !(out_is_int8 || out_is_uint8);
    bool need_clamp = levels != 256 || out_is_fp;
    bool need_min_clamp = need_clamp;
    bool need_max_clamp = need_clamp;

    // Check that we can optimize clamp operation for int8 data using saturation clamp only
    if (per_tensor_out_range && !out_is_fp && levels != 256) {
        if ((out_is_int8 && out_lo_val == -128.f) || (out_is_uint8 && out_lo_val == 0.f))
            need_min_clamp = false;
        if ((out_is_int8 && out_hi_val == 127.f) || (out_is_uint8 && out_hi_val == 255.f))
            need_max_clamp = false;
    }

    if (has_negative_scales) {
        return;
    }

    auto in_scale_prim = std::make_shared<data>(quantize_node.id() + "_in_scale", mem_input_scale);
    auto in_shift_prim = std::make_shared<data>(quantize_node.id() + "_in_shift", mem_input_shift);
    auto out_scale_prim = std::make_shared<data>(quantize_node.id() + "_output_scale", mem_output_scale);
    auto out_shift_prim = std::make_shared<data>(quantize_node.id() + "_output_shift", mem_output_shift);

    std::vector<input_info> quantize_inputs = quantize_node.get_primitive()->input;
    quantize_inputs.push_back(in_scale_prim->id);
    quantize_inputs.push_back(in_shift_prim->id);
    quantize_inputs.push_back(out_scale_prim->id);
    quantize_inputs.push_back(out_shift_prim->id);

    data_types out_dt = primitive->output_data_types.size() ? primitive->output_data_types[0].value_or(data_types::f32) : data_types::f32;
    auto new_quantize_prim = std::make_shared<quantize>(quantize_node.id() + "_opt", quantize_inputs, primitive->levels, out_dt);
    new_quantize_prim->origin_op_name = primitive->origin_op_name;
    new_quantize_prim->origin_op_type_name = primitive->origin_op_type_name;

    new_quantize_prim->scale_shift_opt = true;
    new_quantize_prim->need_post_scale = need_post_scale;
    new_quantize_prim->need_post_shift = need_post_shift;
    new_quantize_prim->need_pre_shift = need_pre_shift;
    new_quantize_prim->per_tensor_input_scale = per_tensor_in_scale;
    new_quantize_prim->per_tensor_input_shift = per_tensor_in_shift && need_pre_shift;
    new_quantize_prim->need_clamp = need_clamp;
    new_quantize_prim->need_min_clamp = need_min_clamp;
    new_quantize_prim->need_max_clamp = need_max_clamp;
    new_quantize_prim->per_tensor_input_range = per_tensor_in_range;
    new_quantize_prim->per_tensor_output_range = per_tensor_out_range;
    new_quantize_prim->per_tensor_output_scale = per_tensor_out_scale;
    new_quantize_prim->per_tensor_output_shift = per_tensor_out_shift;

    // Clamp is needed to avoid inf and -inf which are converted to undefined "inf" constant in opencl
    new_quantize_prim->in_scale  = clamp(in_scale_val);
    new_quantize_prim->in_shift  = clamp(in_shift_val);
    new_quantize_prim->in_lo     = clamp(in_lo_val);
    new_quantize_prim->in_hi     = clamp(in_hi_val);
    new_quantize_prim->out_lo    = clamp(out_lo_val);
    new_quantize_prim->out_hi    = clamp(out_hi_val);
    new_quantize_prim->out_scale = clamp(out_scale_val);
    new_quantize_prim->out_shift = clamp(out_shift_val);

    auto& in_scale_node = p.get_or_create(in_scale_prim);
    auto& in_shift_node = p.get_or_create(in_shift_prim);
    auto& out_scale_node = p.get_or_create(out_scale_prim);
    auto& out_shift_node = p.get_or_create(out_shift_prim);
    auto& new_quantize_node = p.get_or_create(new_quantize_prim);

    auto& inputs = p.get_inputs();

    inputs.push_back(&in_scale_node);
    inputs.push_back(&in_shift_node);
    inputs.push_back(&out_scale_node);
    inputs.push_back(&out_shift_node);

    p.replace(quantize_node, new_quantize_node);

    p.add_connection(in_scale_node, new_quantize_node);
    p.add_connection(in_shift_node, new_quantize_node);
    p.add_connection(out_scale_node, new_quantize_node);
    p.add_connection(out_shift_node, new_quantize_node);
    new_quantize_node.add_memory_dependency(in_scale_node.id());
    new_quantize_node.add_memory_dependency(in_shift_node.id());
    new_quantize_node.add_memory_dependency(out_scale_node.id());
    new_quantize_node.add_memory_dependency(out_shift_node.id());
    p.get_processing_order().insert(&new_quantize_node, &in_shift_node);
    p.get_processing_order().insert(&new_quantize_node, &in_scale_node);
    p.get_processing_order().insert(&new_quantize_node, &out_shift_node);
    p.get_processing_order().insert(&new_quantize_node, &out_scale_node);
}

void prepare_quantization::handle_quantize_node(program& p, quantize_node& quantize_node) {
    if (optimize_quantize(p, quantize_node))
        return;

    auto l = quantize_node.get_primitive()->levels;
    if (l > 2 && l <= 256 && !quantize_node.get_scale_shift_opt() && !quantize_node.is_constant()) {
        prepare_scale_shift_opt(p, quantize_node);
    }
}

void prepare_quantization::prepare_dequantize_merge(program& p, eltwise_node& eltwise_node) {
    for (size_t i = 1; i < eltwise_node.get_dependencies().size(); i++) {
        if (!eltwise_node.get_dependency(i).is_type<data>()) {
            return;
        }
    }

    auto get_scale_shift_mem = [](const cldnn::eltwise_node& eltw, size_t dep_id) -> memory::ptr {
        if (dep_id >= eltw.get_dependencies().size())
            CLDNN_ERROR_MESSAGE(eltw.id(), "Invalid dependency id in dequantize optimization");

        return eltw.get_dependency(dep_id).as<data>().get_attached_memory_ptr();
    };

    const auto& eltw_mode = eltwise_node.get_primitive()->mode;
    if (eltw_mode != eltwise_mode::sum && eltw_mode != eltwise_mode::prod)
        return;

    auto& input = eltwise_node.input();
    const auto& stream = p.get_stream();

    for (auto& user : input.get_users()) {
        if (user == &eltwise_node)
            continue;

        if (!user->is_type<eltwise>() || user->get_dependencies().size() != eltwise_node.get_dependencies().size())
            continue;

        auto& eltwise_dep = user->as<eltwise>();
        if (eltwise_dep.get_primitive()->mode != eltwise_node.get_primitive()->mode)
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
        for (size_t i = 1; i < eltwise_node.get_dependencies().size(); i++) {
            auto mem0 = get_scale_shift_mem(eltwise_dep, i);
            auto mem1 = get_scale_shift_mem(eltwise_node, i);

            mem_lock<uint8_t, mem_lock_type::read> mem0_lock{mem0, stream};
            mem_lock<uint8_t, mem_lock_type::read> mem1_lock{mem1, stream};
            auto ptr0 = mem0_lock.data();
            auto ptr1 = mem1_lock.data();

            for (size_t j = 0; j < mem0->get_layout().bytes_count(); j++) {
                if (ptr0[j] != ptr1[j]) {
                    same_params = false;
                    break;
                }
            }
        }

        if (same_params) {
            while (!eltwise_node.get_dependencies().empty()) {
                auto& dep = eltwise_node.get_dependency(0);
                p.remove_connection(dep, eltwise_node);
                p.remove_if_dangling(dep);
            }
            p.add_optimized_primitive_info(eltwise_node.id(), {user->id()});
            p.replace_all_usages(eltwise_node, *user);
        }
    }
}

void prepare_quantization::remove_fake_reorders(program& p, reorder_node& reorder_node) {
    if (!reorder_node.is_in_data_flow() || reorder_node.get_users().size() != 1 || reorder_node.get_dependencies().size() != 1) {
        return;
    }

    auto &usr = reorder_node.get_users().front();
    auto &dep = reorder_node.get_dependency(0);
    if (!(usr->is_type<convolution>() && usr->get_input_layout(1).data_type == data_types::i8) ||
        !dep.is_input() ||
        dep.get_output_layout().data_type != data_types::u8 ||
        (reorder_node.get_output_layout().data_type != data_types::f32 && reorder_node.get_output_layout().data_type != data_types::f16) ||
        dep.get_output_layout().format != reorder_node.get_output_layout().format ||
        dep.get_output_layout().get_tensor() != reorder_node.get_output_layout().get_tensor())
        return;

    p.replace_all_usages(reorder_node, dep);
    p.add_optimized_primitive_info(reorder_node.id());
    p.remove_all_connections(reorder_node);
    p.remove_if_dangling(reorder_node);
}

template<typename W_T, typename AZP_T>
void fill_compensation_typed(W_T* w, AZP_T* azp, W_T* wzp, float* comp, const int GS, const int OC, const int IC, const int KS) {
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

void prepare_quantization::prepare_asymmetric_quantization(program &p, convolution_node& convolution_node) {
    // Detects if given eltwise node performs zero point subtraction
    auto is_zero_point_node = [](const eltwise_node& node) -> bool {
        auto prim = node.get_primitive();

        if (node.get_dependencies().size() != 2 || prim->mode != eltwise_mode::sub)
            return false;

        if (node.get_users().size() != 1)
            return false;

        auto in0_layout = node.get_input_layout(0);
        auto in1_layout = node.get_input_layout(1);

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

    const auto& stream = p.get_stream();
    auto fill_compensation = [&](int groups, const memory::ptr w, const memory::ptr azp, const memory::ptr wzp, memory::ptr compensation,
                                 bool grouped_weights_shape) {
        auto wl = w->get_layout();
        if (!format::is_weights_format(wl.format)) {
            wl = wl.convert_to_weights_layout(grouped_weights_shape);
        }

        const int GS = groups;
        const int OC = grouped_weights_shape ? wl.ofm() : (wl.ofm() / GS);
        const int IC = wl.ifm();
        const int KS = wl.spatial(0)*wl.spatial(1)*wl.spatial(2);

        const auto& w_dt = wl.data_type;
        const auto& azp_dt = azp->get_layout().data_type;

        mem_lock<float, mem_lock_type::write> comp_lock{compensation, stream};

        if (w_dt == data_types::u8 && azp_dt == data_types::u8) {
            mem_lock<uint8_t, mem_lock_type::read> w_lock(w, stream);
            mem_lock<uint8_t, mem_lock_type::read> azp_lock(azp, stream);
            if (wzp) {
                mem_lock<uint8_t, mem_lock_type::read> wzp_lock(wzp, stream);
                fill_compensation_typed(w_lock.data(), azp_lock.data(), wzp_lock.data(), comp_lock.data(), GS, OC, IC, KS);
            } else {
                fill_compensation_typed(w_lock.data(), azp_lock.data(), static_cast<uint8_t*>(nullptr), comp_lock.data(), GS, OC, IC, KS);
            }
        } else if (w_dt == data_types::i8 && azp_dt == data_types::u8) {
            mem_lock<int8_t, mem_lock_type::read> w_lock(w, stream);
            mem_lock<uint8_t, mem_lock_type::read> azp_lock(azp, stream);
            if (wzp) {
                mem_lock<int8_t, mem_lock_type::read> wzp_lock(wzp, stream);
                fill_compensation_typed(w_lock.data(), azp_lock.data(), wzp_lock.data(), comp_lock.data(), GS, OC, IC, KS);
            } else {
                fill_compensation_typed(w_lock.data(), azp_lock.data(), static_cast<int8_t*>(nullptr), comp_lock.data(), GS, OC, IC, KS);
            }
        } else if (w_dt == data_types::i8 && azp_dt == data_types::i8) {
            mem_lock<int8_t, mem_lock_type::read> w_lock(w, stream);
            mem_lock<int8_t, mem_lock_type::read> azp_lock(azp, stream);
            if (wzp) {
                mem_lock<int8_t, mem_lock_type::read> wzp_lock(wzp, stream);
                fill_compensation_typed(w_lock.data(), azp_lock.data(), wzp_lock.data(), comp_lock.data(), GS, OC, IC, KS);
            } else {
                fill_compensation_typed(w_lock.data(), azp_lock.data(), static_cast<int8_t*>(nullptr), comp_lock.data(), GS, OC, IC, KS);
            }
        } else if (w_dt == data_types::u8 && azp_dt == data_types::i8) {
            mem_lock<uint8_t, mem_lock_type::read> w_lock(w, stream);
            mem_lock<int8_t, mem_lock_type::read> azp_lock(azp, stream);
            if (wzp) {
                mem_lock<uint8_t, mem_lock_type::read> wzp_lock(wzp, stream);
                fill_compensation_typed(w_lock.data(), azp_lock.data(), wzp_lock.data(), comp_lock.data(), GS, OC, IC, KS);
            } else {
                fill_compensation_typed(w_lock.data(), azp_lock.data(), static_cast<uint8_t*>(nullptr), comp_lock.data(), GS, OC, IC, KS);
            }
        }
    };

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

    const size_t feature_idx = 1;
    if (asymmetric_data && in0.get_output_layout().get_partial_shape()[feature_idx].is_dynamic())
        return;

    auto old_conv_prim = convolution_node.get_primitive();

    primitive_id input = old_conv_prim->input[0].pid;
    primitive_id a_zero_points = "";

    cldnn::program_node* new_input = &in0;
    cldnn::program_node* new_a_zp = nullptr;
    cldnn::program_node* new_w_zp = nullptr;

    bool need_compensation = false;

    auto wl = in1.get_output_layout();
    if (!format::is_weights_format(wl.format)) {
        wl = wl.convert_to_weights_layout(convolution_node.typed_desc()->grouped_weights_shape);
    }
    int ofm = wl.group() * wl.ofm();
    int ifm = in0.get_output_layout().get_partial_shape()[feature_idx].get_length();
    int ofm_aligned = align_to(ofm, 32);
    int ifm_aligned = align_to(ifm, 32);

    if (asymmetric_data) {
        new_input = &in0.get_dependency(0);
        new_a_zp = &in0.get_dependency(1);

        auto l = layout{new_a_zp->get_output_layout().data_type, format::bfyx, tensor{1, ifm_aligned, 1, 1}};
        int s = new_a_zp->get_output_layout().feature();
        auto azp_aligned = p.get_engine().allocate_memory(l);
        auto old_ptr = new_a_zp->as<data>().get_attached_memory_ptr();
        mem_lock<int8_t, mem_lock_type::write> new_data{azp_aligned, stream};
        mem_lock<int8_t, mem_lock_type::read> old_data{old_ptr, stream};
        for (int i = 0; i < ifm_aligned; i++) {
            new_data.data()[i] = old_data.data()[i % s];
        }
        new_a_zp->as<data>().attach_memory(azp_aligned, false); // don't invalidate users as those will be reconnected to new node

        input = new_input->id();
        a_zero_points = new_a_zp->id();
        need_compensation = true;
    }

    primitive_id w_zero_points = "";
    primitive_id weights = old_conv_prim->weights;
    cldnn::program_node* new_weights = &in1;
    if (asymmetric_weights) {
        new_weights = &in1.get_dependency(0);
        new_w_zp = &in1.get_dependency(1);

        auto l = layout{new_w_zp->get_output_layout().data_type, format::bfyx, tensor{ofm_aligned, 1, 1, 1}};
        int s = new_w_zp->get_output_layout().batch();
        auto wzp_aligned = p.get_engine().allocate_memory(l);
        auto old_ptr = new_w_zp->as<data>().get_attached_memory_ptr();
        mem_lock<int8_t, mem_lock_type::write> new_data{wzp_aligned, stream};
        mem_lock<int8_t, mem_lock_type::read> old_data{old_ptr, stream};
        for (int i = 0; i < ofm_aligned; i++) {
            new_data.data()[i] = old_data.data()[i % s];
        }
        new_w_zp->as<data>().attach_memory(wzp_aligned, false);  // don't invalidate users as those will be reconnected to new node

        weights = new_weights->id();
        w_zero_points = new_w_zp->id();
    }

    if (!new_weights->is_type<data>()) {
        need_compensation = false;
    }

    primitive_id compensation = "";
    cldnn::program_node* new_compenstation = nullptr;
    if (need_compensation) {
        auto l = layout{data_types::f32, format::bfyx, tensor{1, ofm_aligned, 1, 1}};
        auto data_to_allocate = p.get_engine().allocate_memory(l);
        auto w = new_weights->as<data>().get_attached_memory_ptr();
        auto azp = asymmetric_data ? new_a_zp->as<data>().get_attached_memory_ptr() : nullptr;
        auto wzp = asymmetric_weights ? new_w_zp->as<data>().get_attached_memory_ptr() : nullptr;
        int groups = static_cast<int>(convolution_node.get_groups());
        fill_compensation(groups, w, azp, wzp, data_to_allocate, convolution_node.typed_desc()->grouped_weights_shape);

        auto compensation_prim = std::make_shared<data>(convolution_node.id() + "_compensation", data_to_allocate);
        new_compenstation = &p.get_or_create(compensation_prim);
        p.get_inputs().push_back(new_compenstation);
        compensation = new_compenstation->id();
    }

    // Collect dependencies of a new convolution node
    std::vector<std::pair<program_node*, int32_t>> dependencies = {{new_input, 0}, {new_weights, 0}};
    cldnn::program_node* new_bias = !old_conv_prim->bias.empty() ? &convolution_node.get_dependency(2) : nullptr;
    if (new_bias)
        dependencies.push_back({new_bias, 0});
    if (new_w_zp)
        dependencies.push_back({new_w_zp, 0});
    if (new_a_zp)
        dependencies.push_back({new_a_zp, 0});
    if (new_compenstation)
        dependencies.push_back({new_compenstation, 0});

    auto new_conv_prim = std::make_shared<convolution>(
                convolution_node.id() + "_asymmetric",
                input,
                weights,
                old_conv_prim->bias,
                w_zero_points,
                a_zero_points,
                compensation,
                old_conv_prim->groups,
                old_conv_prim->stride,
                old_conv_prim->dilation,
                old_conv_prim->padding_begin,
                old_conv_prim->padding_end,
                old_conv_prim->grouped_weights_shape,
                !old_conv_prim->output_data_types.empty() ? old_conv_prim->output_data_types[0].value_or(data_types::f32) : data_types::f32,
                old_conv_prim->auto_pad,
                old_conv_prim->output_paddings[0]);

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
}

bool prepare_quantization::optimize_quantize(program &p, quantize_node& quantize_node) {
    const auto& stream = p.get_stream();

    auto& input = quantize_node.get_dependency(0);
    auto parallel_quantizes_num = 0;
    for (auto& usr : input.get_users()) {
        if (usr->is_type<quantize>())
            parallel_quantizes_num++;
    }

    if (parallel_quantizes_num < 2)
        return false;

    auto quantize_prim_first = quantize_node.get_primitive();

    program_node &input_low_node_first = quantize_node.get_dependency(1);
    program_node &input_high_node_first = quantize_node.get_dependency(2);
    program_node &output_low_node_first = quantize_node.get_dependency(3);
    program_node &output_high_node_first = quantize_node.get_dependency(4);

    if (!input_low_node_first.is_type<data>() || !input_high_node_first.is_type<data>() ||
        !output_low_node_first.is_type<data>() || !output_high_node_first.is_type<data>()) {
        return false;
    }

    auto mem_input_low_first = input_low_node_first.as<data>().get_attached_memory_ptr();
    auto mem_input_high_first = input_high_node_first.as<data>().get_attached_memory_ptr();
    auto mem_output_low_first = output_low_node_first.as<data>().get_attached_memory_ptr();
    auto mem_output_high_first = output_high_node_first.as<data>().get_attached_memory_ptr();

    mem_lock<uint8_t, mem_lock_type::read> mem_input_low_lock_first{mem_input_low_first, stream};
    mem_lock<uint8_t, mem_lock_type::read> mem_input_high_lock_first{mem_input_high_first, stream};
    mem_lock<uint8_t, mem_lock_type::read> mem_output_low_lock_first{mem_output_low_first, stream};
    mem_lock<uint8_t, mem_lock_type::read> mem_output_high_lock_first{mem_output_high_first, stream};

    program_node* same_quantize = nullptr;
    for (auto& usr : input.get_users()) {
        if (!usr->is_type<quantize>() || usr == &quantize_node)
            continue;

        auto quantize_prim_second = usr->as<quantize>().get_primitive();

        program_node &input_low_node_second = usr->get_dependency(1);
        program_node &input_high_node_second = usr->get_dependency(2);
        program_node &output_low_node_second = usr->get_dependency(3);
        program_node &output_high_node_second = usr->get_dependency(4);

        if (!input_low_node_second.is_type<data>() || !input_high_node_second.is_type<data>() ||
            !output_low_node_second.is_type<data>() || !output_high_node_second.is_type<data>())
            continue;

        auto mem_input_low_second = input_low_node_second.as<data>().get_attached_memory_ptr();
        auto mem_input_high_second = input_high_node_second.as<data>().get_attached_memory_ptr();
        auto mem_output_low_second = output_low_node_second.as<data>().get_attached_memory_ptr();
        auto mem_output_high_second = output_high_node_second.as<data>().get_attached_memory_ptr();

        mem_lock<uint8_t, mem_lock_type::read> mem_input_low_lock_second{mem_input_low_second, stream};
        mem_lock<uint8_t, mem_lock_type::read> mem_input_high_lock_second{mem_input_high_second, stream};
        mem_lock<uint8_t, mem_lock_type::read> mem_output_low_lock_second{mem_output_low_second, stream};
        mem_lock<uint8_t, mem_lock_type::read> mem_output_high_lock_second{mem_output_high_second, stream};

        if (mem_input_low_first->count() != mem_input_low_second->count() || mem_input_high_first->count() != mem_input_high_second->count() ||
            mem_output_low_first->count() != mem_output_low_second->count() || mem_output_high_first->count() != mem_output_high_second->count())
            continue;

        if (memcmp(mem_input_low_lock_first.data(), mem_input_low_lock_second.data(), mem_input_low_first->size()) != 0 ||
            memcmp(mem_input_high_lock_first.data(), mem_input_high_lock_second.data(), mem_input_high_first->size()) != 0 ||
            memcmp(mem_output_low_lock_first.data(), mem_output_low_lock_second.data(), mem_output_low_first->size()) != 0 ||
            memcmp(mem_output_high_lock_first.data(), mem_output_high_lock_second.data(), mem_output_high_first->size()) != 0)
            continue;

        if (quantize_prim_first->output_data_types[0] != quantize_prim_second->output_data_types[0] ||
            quantize_prim_first->levels != quantize_prim_second->levels)
            continue;

        same_quantize = usr;
        break;
    }

    if (!same_quantize)
        return false;

    while (!quantize_node.get_dependencies().empty()) {
        auto& dep = quantize_node.get_dependency(0);
        p.remove_connection(dep, quantize_node);
        p.remove_if_dangling(dep);
    }

    p.add_optimized_primitive_info(quantize_node.id(), {same_quantize->id()});
    p.replace_all_usages(quantize_node, *same_quantize);

    return true;
}

static void optimize_weights_decompression_parameters(fully_connected_node& fc_node, program& p) {
    auto fc_prim = fc_node.get_primitive();
    if (!fc_prim->compressed_weights)
        return;

    auto reorder_bfyx_to_fbyx = [&](size_t dep_id) {
        auto& dep = fc_node.get_dependency(dep_id);
        auto target_layout = dep.get_output_layout();
        target_layout.format = format::fbyx;
        auto reorder_prim = std::make_shared<reorder>(dep.id() + "_reorder", dep.id(), target_layout);
        p.add_intermediate(reorder_prim, fc_node, dep_id, true);
        fc_node.get_dependency(dep_id).recalc_output_layout(false);
    };

    auto need_reorder = [&](size_t dep_id) {
        auto dep_layout = fc_node.get_input_layout(dep_id);
        auto dep_pshape = dep_layout.get_partial_shape();

        auto groups_count = dep_pshape[dep_pshape.size() - 1].get_length();

        return groups_count > 1;
    };

    auto decompression_scale_idx = !fc_node.bias_term() ? 2 : 3;
    if (need_reorder(decompression_scale_idx)) {
        reorder_bfyx_to_fbyx(decompression_scale_idx);
    }

    if (!fc_prim->decompression_zero_point.empty()) {
        auto decompression_zp_idx = decompression_scale_idx + 1;
        if (need_reorder(decompression_zp_idx)) {
            reorder_bfyx_to_fbyx(decompression_zp_idx);
        }
    }
}

void prepare_quantization::run(program& p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto &node = (*itr++);
        if (node->is_type<quantize>()) {
            handle_quantize_node(p, node->as<quantize>());
        } else if (node->is_type<eltwise>()) {
            prepare_dequantize_merge(p, node->as<eltwise>());
        } else if (node->is_type<reorder>()) {
            remove_fake_reorders(p, node->as<reorder>());
        } else if (node->is_type<convolution>()) {
            prepare_asymmetric_quantization(p, node->as<convolution>());
        } else if (node->is_type<fully_connected>()) {
            optimize_weights_decompression_parameters(node->as<fully_connected>(), p);
        }
    }
}
