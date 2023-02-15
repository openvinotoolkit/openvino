// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define ONEDNN_PRIMITIVE_SERIALIZATION

#include "primitive_inst.h"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "to_string_utils.h"
#include "register.hpp"
#include "utils.hpp"
#include "openvino/util/file_util.hpp"

#include "quantize_inst.h"
#include "reorder_inst.h"

#include "reorder/reorder_weights_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"

#include <vector>
#include <list>
#include <utility>

#include <oneapi/dnnl/dnnl.hpp>

namespace cldnn {
namespace onednn {

static std::mutex cacheAccessMutex;

template <class PType, class PrimDescType = dnnl::primitive_desc, class PrimType = dnnl::primitive>
struct typed_primitive_onednn_impl : public typed_primitive_impl<PType> {
    const engine* _engine;
    std::shared_ptr<dnnl::primitive_attr> _attrs;
    PrimDescType _pd;
    PrimType _prim;
    std::unordered_map<uint32_t, std::unordered_map<int, dnnl::memory>> _args;
    bool _enable_profiling = false;

    typed_primitive_onednn_impl(const engine& engine,
            const ExecutionConfig& config,
            std::shared_ptr<dnnl::primitive_attr> attrs,
            const PrimDescType& pd,
            kernel_selector::WeightsReorderParams weights_reorder = {})
        : typed_primitive_impl<PType>(weights_reorder, pd.impl_info_str()),
        _engine(&engine),
        _attrs(attrs),
        _pd(pd) {
            _enable_profiling = config.get_property(ov::enable_profiling);
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
                _enable_profiling = true;
            }
            build_primitive(config);
        }

    typed_primitive_onednn_impl(const engine& engine, const ExecutionConfig& config = {})
        : typed_primitive_impl<PType>({}, "undef"),
        _engine(&engine),
        _pd(),
        _prim() {
            _enable_profiling = config.get_property(ov::enable_profiling);
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
                _enable_profiling = true;
            }
        }

    typed_primitive_onednn_impl()
        : typed_primitive_impl<PType>({}, "undef"),
          _pd(), _prim() {
        _attrs = std::make_shared<dnnl::primitive_attr>();
    }

    bool is_cpu() const override { return false; }

    // Cache blob format:
    //     [ dnnl::primitive_attr ]
    //     [ dnnl::primitive_desc ]
    //     [ dnnl::cache_blob ]
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        if (_attrs->get() == nullptr) {
            ob << false;
        } else {
            ob << true;
        }

        if (_attrs->get() != nullptr) {
            {
                dnnl::scratchpad_mode _scratchpad_mode = _attrs->get_scratchpad_mode();
                ob << make_data(&_scratchpad_mode, sizeof(dnnl::scratchpad_mode));
            }
            {
                dnnl::fpmath_mode _fmath_mode = _attrs->get_fpmath_mode();
                ob << make_data(&_fmath_mode, sizeof(dnnl::fpmath_mode));
            }
            {
                const dnnl::post_ops _post_ops = _attrs->get_post_ops();

                ob << _post_ops.len();
                for (int idx = 0; idx < _post_ops.len(); ++idx) {
                    dnnl::primitive::kind _kind = _post_ops.kind(idx);

                    ob << make_data(&_kind, sizeof(dnnl::primitive::kind));

                    if (_kind == dnnl::primitive::kind::sum) {
                        float scale;
                        int32_t zero_point;
                        dnnl::memory::data_type data_type;

                        _post_ops.get_params_sum(idx, scale, zero_point, data_type);

                        ob << scale;
                        ob << zero_point;
                        ob << make_data(&data_type, sizeof(dnnl::memory::data_type));
                    } else if (_kind == dnnl::primitive::kind::eltwise) {
                        dnnl::algorithm aalgorithm;
                        float alpha;
                        float beta;

                        _post_ops.get_params_eltwise(idx, aalgorithm, alpha, beta);
                        ob << make_data(&aalgorithm, sizeof(dnnl::algorithm));
                        ob << alpha;
                        ob << beta;
                    } else if (_kind == dnnl::primitive::kind::convolution) {
                        dnnl::memory::data_type weights_data_type;
                        dnnl::memory::data_type bias_data_type;
                        dnnl::memory::data_type dst_data_type;
                        dnnl::memory::dim kernel_size;
                        dnnl::memory::dim stride_size;
                        dnnl::memory::dim padding_l_size;

                        _post_ops.get_params_dw(idx, weights_data_type, bias_data_type, dst_data_type, kernel_size, stride_size, padding_l_size);

                        ob << make_data(&weights_data_type, sizeof(dnnl::memory::data_type));
                        ob << make_data(&bias_data_type, sizeof(dnnl::memory::data_type));
                        ob << make_data(&dst_data_type, sizeof(dnnl::memory::data_type));
                        ob << kernel_size << stride_size << padding_l_size;
                    } else if (_kind == dnnl::primitive::kind::binary) {
                        dnnl::algorithm aalgorithm;
                        dnnl::memory::desc src1_desc;

                        _post_ops.get_params_binary(idx, aalgorithm, src1_desc);

                        ob << make_data(&aalgorithm, sizeof(dnnl::algorithm));
                    } else if (_kind == dnnl::primitive::kind::prelu) {
                        int mask;

                        _post_ops.get_params_prelu(idx, mask);

                        ob << mask;
                    }
                }
            }
            {
                float scale, shift;
                _attrs->get_rnn_data_qparams(scale, shift);
                ob << scale << shift;
            }
            {
                int mask;
                std::vector<float> scales;

                _attrs->get_rnn_weights_qparams(mask, scales);

                ob << mask;
                ob << scales;
            }
            {
                int mask;
                std::vector<float> scales;

                _attrs->get_rnn_weights_projection_qparams(mask, scales);

                ob << mask;
                ob << scales;
            }
        }
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        bool has_attrs;
        ib >> has_attrs;

        if (has_attrs) {
            {
                dnnl::scratchpad_mode _scratchpad_mode = dnnl::scratchpad_mode::library;
                ib >> make_data(&_scratchpad_mode, sizeof(dnnl::scratchpad_mode));
                _attrs->set_scratchpad_mode(_scratchpad_mode);
            }
            {
                dnnl::fpmath_mode _fmath_mode = dnnl::fpmath_mode::any;
                ib >> make_data(&_fmath_mode, sizeof(dnnl::fpmath_mode));
                _attrs->set_fpmath_mode(_fmath_mode);
            }
            {
                const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernlImplParams());
                const std::vector<cldnn::fused_primitive_desc_onednn>& fused_desc = impl_params->fused_desc_onednn;
                dnnl::post_ops _post_ops;
                int post_ops_len;

                ib >> post_ops_len;
                for (int idx = 0; idx < post_ops_len; ++idx) {
                    dnnl::primitive::kind _kind = dnnl::primitive::kind::undef;

                    ib >> make_data(&_kind, sizeof(dnnl::primitive::kind));

                    if (_kind == dnnl::primitive::kind::sum) {
                        float scale;
                        int32_t zero_point;
                        dnnl::memory::data_type data_type = dnnl::memory::data_type::undef;

                        ib >> scale;
                        ib >> zero_point;
                        ib >> make_data(&data_type, sizeof(dnnl::memory::data_type));

                        _post_ops.append_sum(scale, zero_point, data_type);
                    } else if (_kind == dnnl::primitive::kind::eltwise) {
                        dnnl::algorithm aalgorithm = dnnl::algorithm::undef;
                        float alpha;
                        float beta;

                        ib >> make_data(&aalgorithm, sizeof(dnnl::algorithm));
                        ib >> alpha;
                        ib >> beta;
                        _post_ops.append_eltwise(aalgorithm, alpha, beta);
                    } else if (_kind == dnnl::primitive::kind::convolution) {
                        dnnl::memory::data_type weights_data_type = dnnl::memory::data_type::undef;
                        dnnl::memory::data_type bias_data_type = dnnl::memory::data_type::undef;
                        dnnl::memory::data_type dst_data_type = dnnl::memory::data_type::undef;
                        dnnl::memory::dim kernel_size;
                        dnnl::memory::dim stride_size;
                        dnnl::memory::dim padding_l_size;

                        ib >> make_data(&weights_data_type, sizeof(dnnl::memory::data_type));
                        ib >> make_data(&bias_data_type, sizeof(dnnl::memory::data_type));
                        ib >> make_data(&dst_data_type, sizeof(dnnl::memory::data_type));
                        ib >> kernel_size >> stride_size >> padding_l_size;

                        _post_ops.append_dw(weights_data_type, bias_data_type, dst_data_type,
                                            kernel_size, stride_size, padding_l_size);
                    } else if (_kind == dnnl::primitive::kind::binary) {
                        dnnl::algorithm aalgorithm = dnnl::algorithm::undef;
                        ib >> make_data(&aalgorithm, sizeof(dnnl::algorithm));

                        dnnl::memory::desc md = onednn::layout_to_memory_desc(
                                                        impl_params->get_input_layout(fused_desc.at(idx).mem_dep),
                                                        fused_desc.at(idx).tag, fused_desc.at(idx).flatten);

                        _post_ops.append_binary(aalgorithm, md);
                    } else if (_kind == dnnl::primitive::kind::prelu) {
                        int mask;
                        ib >> mask;
                        _post_ops.append_prelu(mask);
                    }
                }

                _attrs->set_post_ops(_post_ops);
            }
            {
                float scale;
                float shift;

                ib >> scale >> shift;
                _attrs->set_rnn_data_qparams(scale, shift);
            }
            {
                int mask;
                std::vector<float> scales;

                ib >> mask;
                ib >> scales;

                _attrs->set_rnn_weights_qparams(mask, scales);
            }
            {
                int mask;
                std::vector<float> scales;

                ib >> mask;
                ib >> scales;

                _attrs->set_rnn_weights_projection_qparams(mask, scales);
            }

            _engine = &ib.get_engine();
        }
#endif
    }

private:
    using primitive_impl::get_arguments;

    std::string get_cache_directory(const ExecutionConfig& config) const {
        auto path = config.get_property(ov::cache_dir);
        if (path.empty()) {
            return {};
        }

        if (path.back() != '/' && path.back() != '\\') {
            path += "/";
        }
        return path;
    }

    std::string generate_cache_path_from_key(const ExecutionConfig& config, std::vector<uint8_t> key) const {
        auto path = get_cache_directory(config);
        if (path.empty()) {
            return {};
        }

        std::string key_str(key.begin(), key.end());
        size_t hash = std::hash<std::string>()(key_str);
        return path + std::to_string(hash) + ".onednn.cl_cache";
    }

    void build_primitive(const ExecutionConfig& config) {
        auto cache_outpath = get_cache_directory(config);

        if (const char* env_p = std::getenv("OV_GPU_CACHE_MODEL")) {
            if (env_p[0] == '1') {
                cache_outpath = "";
            }
        }

        if (cache_outpath.empty()) {
            _prim = PrimType(_pd);
        } else {
            std::vector<uint8_t> key = _pd.get_cache_blob_id();
            assert(!key.empty());

            std::vector<uint8_t> cache;
            {
                std::lock_guard<std::mutex> lock(cacheAccessMutex);
                cache = ov::util::load_binary(generate_cache_path_from_key(config, key));
            }

            if (cache.empty()) {
                _prim = PrimType(_pd);
                cache = _prim.get_cache_blob();

                {
                    std::lock_guard<std::mutex> lock(cacheAccessMutex);
                    ov::util::save_binary(generate_cache_path_from_key(config, key), cache);
                }
            } else {
                _prim = PrimType(_pd, cache);
            }
        }
    }

protected:
    virtual bool optimized_out(typed_primitive_inst<PType>&) const { return false; }

    void configure_post_ops_arguments(typed_primitive_inst<PType>& instance, std::unordered_map<int, dnnl::memory>& args) const {
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        // Get current post-ops info
        dnnl::post_ops post_ops = _attrs->get_post_ops();

        // Create onednn memory buffers for post-ops
        auto& cur_post_ops = instance.get_fused_primitives_onednn();
        auto post_ops_size = cur_post_ops.size();
        for (size_t post_op_idx = 0, num_of_optimized_post_ops = 0; post_op_idx < post_ops_size; post_op_idx++) {
            auto post_op_type = cur_post_ops[post_op_idx].op_type;
            auto memory_offset = cur_post_ops[post_op_idx].mem_offset;
            auto onednn_post_op_idx = post_op_idx;
            onednn_post_op_idx -= num_of_optimized_post_ops;

            switch (post_op_type) {
                case onednn_post_op_type::eltwise_act:
                case onednn_post_op_type::eltwise_clip:
                case onednn_post_op_type::eltwise_linear:
                case onednn_post_op_type::eltwise_round:
                case onednn_post_op_type::eltwise_hardsigmoid:
                {
                    // onednn elwise doesn't need any data from memory buffers
                    break;
                }

                case onednn_post_op_type::binary_add:
                case onednn_post_op_type::binary_sub:
                case onednn_post_op_type::binary_mul:
                case onednn_post_op_type::binary_max:
                case onednn_post_op_type::binary_min:
                {
                    auto binary_op_mem = instance.fused_memory(memory_offset);
                    dnnl::algorithm alg;
                    dnnl::memory::desc desc;
                    post_ops.get_params_binary(static_cast<int>(onednn_post_op_idx), alg, desc);
                    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(onednn_post_op_idx)) | DNNL_ARG_SRC_1,
                                 binary_op_mem->get_onednn_memory(desc)});
                    break;
                }

                case onednn_post_op_type::binary_relu:
                {
                    auto binary_op_mem = instance.fused_memory(memory_offset);
                    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(onednn_post_op_idx)) | DNNL_ARG_WEIGHTS,
                                 binary_op_mem->get_onednn_memory(_pd.dnnl::primitive_desc_base::weights_desc(0))});
                    break;
                }

                case onednn_post_op_type::sum:
                case onednn_post_op_type::optimized_sum:
                case onednn_post_op_type::optimized_eltwise_linear:
                case onednn_post_op_type::optimized_eltwise_act:
                case onednn_post_op_type::optimized_eltwise_round:
                case onednn_post_op_type::optimized_eltwise_clip:
                {
                    break;
                }

                case onednn_post_op_type::optimized:
                {
                    // Optimized post-op, count it to respect onednn_post_op_idx in the next operations
                    num_of_optimized_post_ops++;
                    break;
                }

                default:
                    throw std::runtime_error("Unsupported onednn post-operation type");
            }
        }
    }

    virtual std::unordered_map<int, dnnl::memory> get_arguments(typed_primitive_inst<PType>& instance) const {
        std::unordered_map<int, dnnl::memory> args;
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        {
            auto& input = instance.input_memory(0);
            auto offset = onednn::get_f_offset(instance.get_input_layout(), _pd.dnnl::primitive_desc_base::src_desc(0));
            args.insert({DNNL_ARG_SRC, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(0), offset)});
        }

        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_f_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }

        configure_post_ops_arguments(instance, args);

        return args;
    }

    void init_kernels(const kernels_cache&) override { }

    event::ptr aggregate_events(const std::vector<event::ptr>& events, stream& stream, bool group = false, bool is_output = false) const {
        if (events.size() == 1 && !is_output)
            return events[0];

        if (group && !is_output)
            return stream.group_events(events);

        return stream.enqueue_marker(events, is_output);
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (instance.can_be_optimized())
            return;
        uint32_t net_id = instance.get_network().get_id();
        _args[net_id] = get_arguments(instance);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<PType>& instance) override {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        auto net_id = network.get_id();
        event::ptr event;

        if (_enable_profiling) {
            stream.finish();
            event = stream.create_user_event(false);
        }

        if (!instance.can_be_optimized()) {
            _prim.execute(stream.get_onednn_stream(), _args[net_id]);
        }

        if (_enable_profiling) {
            stream.finish();
            event->set();
        }

        return event;
    }
};

}  // namespace onednn
}  // namespace cldnn
