// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define ONEDNN_PRIMITIVE_SERIALIZATION

#include "primitive_inst.h"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/file_util.hpp"
#include "to_string_utils.h"
#include "register.hpp"
#include "utils.hpp"
#include "runtime/ocl/ocl_event.hpp"

#include "quantize_inst.h"
#include "reorder_inst.h"

#include "reorder/reorder_weights_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"
#include "impls/ocl/kernel_selector_helper.h"

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
    dnnl::memory::desc _scratchpad_md;
    bool _enable_profiling = false;

    typed_primitive_onednn_impl(const engine& engine,
            const ExecutionConfig& config,
            std::shared_ptr<dnnl::primitive_attr> attrs,
            const PrimDescType& pd,
            std::shared_ptr<WeightsReorderParams> weights_reorder = {})
        : typed_primitive_impl<PType>(weights_reorder, pd.impl_info_str()),
        _engine(&engine),
        _attrs(attrs),
        _pd(pd) {
            _enable_profiling = config.get_property(ov::enable_profiling);

            _scratchpad_md = _pd.scratchpad_desc();

            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
                _enable_profiling = true;
            }

            GPU_DEBUG_IF(debug_config->verbose >= 4) {
                if (_scratchpad_md.get_size() > 0) {
                    static std::atomic_llong total{0};
                    int64_t size = _scratchpad_md.get_size() / 1048576;
                    total += size;
                    GPU_DEBUG_TRACE_DETAIL << " [scratchpad] kind: " << static_cast<int>(_pd.get_kind())
                        << ", " << size << "MB, total " << total << "MB" << std::endl;
                }
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
          _engine(nullptr),
          _pd(), _prim() {
        _attrs = std::make_shared<dnnl::primitive_attr>();
    }

    bool is_cpu() const override { return false; }
    bool is_onednn() const override { return true; }

    // Cache blob format:
    //     [ dnnl::primitive_attr ]
    //     [ dnnl::primitive_desc ]
    //     [ dnnl::cache_blob ]
    void save(BinaryOutputBuffer& ob) const override {
        primitive_impl::save(ob);
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
                dnnl::fpmath_mode _fmath_mode;
                bool _apply_to_int;
                _attrs->get_fpmath_mode(_fmath_mode, _apply_to_int);
                ob << make_data(&_fmath_mode, sizeof(dnnl::fpmath_mode));
                ob << _apply_to_int;
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
        primitive_impl::load(ib);
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        bool has_attrs;
        ib >> has_attrs;

        if (has_attrs) {
            {
                dnnl::scratchpad_mode _scratchpad_mode = dnnl::scratchpad_mode::user;
                ib >> make_data(&_scratchpad_mode, sizeof(dnnl::scratchpad_mode));
                _attrs->set_scratchpad_mode(_scratchpad_mode);
            }
            {
                dnnl::fpmath_mode _fmath_mode = dnnl::fpmath_mode::any;
                bool _apply_to_int = false;
                ib >> make_data(&_fmath_mode, sizeof(dnnl::fpmath_mode));
                ib >> _apply_to_int;
                _attrs->set_fpmath_mode(_fmath_mode, _apply_to_int);
            }
            {
                const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
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

                        if (fused_desc.at(idx).dims.size() > 0) {
                            _post_ops.append_binary(aalgorithm,
                                dnnl::memory::desc(fused_desc.at(idx).dims, fused_desc.at(idx).dt, fused_desc.at(idx).tag));
                        } else {
                            dnnl::memory::desc md = onednn::layout_to_memory_desc(
                                                            impl_params->get_input_layout(fused_desc.at(idx).mem_dep),
                                                            fused_desc.at(idx).tag, fused_desc.at(idx).flatten);

                            _post_ops.append_binary(aalgorithm, md);
                        }
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

        if (!config.get_property(ov::intel_gpu::allow_new_shape_infer)) {
            cache_outpath = "";
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
                    ov::intel_gpu::save_binary(generate_cache_path_from_key(config, key), cache);
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
            auto offset = onednn::get_offset(instance.get_input_layout(0), _pd.dnnl::primitive_desc_base::src_desc(0));
            args.insert({DNNL_ARG_SRC, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(0), offset)});
        }

        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }

        if (_scratchpad_md.get_size() != 0) {
            // onednn primitive can have only 1 scratchpad memory.
            auto scratchpad = instance.get_intermediates_memories()[0];
            args.insert({DNNL_ARG_SCRATCHPAD, scratchpad->get_onednn_memory(_scratchpad_md, 0)});
        }

        configure_post_ops_arguments(instance, args);

        return args;
    }

    virtual std::unordered_map<int, dnnl::memory> get_arguments(typed_primitive_inst<PType>& instance, kernel_arguments_data& mem_args) const {
        std::unordered_map<int, dnnl::memory> args;
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        OPENVINO_ASSERT(mem_args.inputs.size() == 1);
        OPENVINO_ASSERT(mem_args.outputs.size() == 1);
        OPENVINO_ASSERT(_scratchpad_md.get_size() == 0);
        OPENVINO_ASSERT(instance.get_fused_primitives_onednn().empty());

        {
            auto input = mem_args.inputs[0];
            layout l = input->get_layout();
            auto offset = onednn::get_offset(std::move(l), _pd.dnnl::primitive_desc_base::src_desc(0));
            args.insert({DNNL_ARG_SRC, input->get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(0), offset)});
        }

        {
            auto output = mem_args.outputs[0];
            layout l = output->get_layout();
            auto offset = onednn::get_offset(std::move(l), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST, output->get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }

        return args;
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override { }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (instance.can_be_optimized())
            return;
        uint32_t net_id = instance.get_network().get_id();
        _args[net_id] = get_arguments(instance);
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override {
        if (instance.can_be_optimized()) {
            return;
        }

        _args[instance.get_network().get_id()] = get_arguments(instance, args);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<PType>& instance) override {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        auto net_id = network.get_id();
        event::ptr event;

        if (_enable_profiling) {
            if (instance.can_be_optimized()) {
                event = stream.create_user_event(true);
            } else {
                dnnl::reset_profiling(stream.get_onednn_stream());
            }
        }

        if (!instance.can_be_optimized()) {
            try {
                _prim.execute(stream.get_onednn_stream(), _args[net_id]);
            } catch (dnnl::error& err) {
                auto err_code = err.status == dnnl_status_t::dnnl_out_of_memory ? CL_OUT_OF_RESOURCES : CL_INVALID_OPERATION;
                ocl::rethrow_or_exit(err.what(), err_code, _engine->get_device_info());
            }

            if (_enable_profiling) {
                // Call wait() function here instead of finish() to prevent cache flushing,
                // this synchronization point is needed for correct OneDNN's profiling process
                stream.wait();

                std::vector<uint64_t> duration = dnnl::get_profiling_data(stream.get_onednn_stream(), dnnl::profiling_data_kind::time);
                OPENVINO_ASSERT(duration.size() == 1, "[GPU] oneDNN profiling data is expected to have info only for single primitive ",
                                                      "actual number is ", duration.size());

                event = std::make_shared<ocl::ocl_event>(duration[0]);
            } else {
                // If oneDNN primitive is the output primitive or it's user is CPU implementation, then enqueue marker
                // with empty events wait list (which will trigger wait for all previously enqueued tasks) and
                // return it as oneDNN primitive's event as it is a single option for proper synchronization
                if (instance.needs_completion_event())
                    event = stream.enqueue_marker({});
            }
        }

        return event;
    }

    std::vector<layout> get_internal_buffer_layouts_impl(const kernel_impl_params& /*params*/) const override {
        if (_scratchpad_md.get_size() == 0)
            return {};
        return {{{1, 1, 1, (tensor::value_type)(_scratchpad_md.get_size())}, cldnn::data_types::u8, format::bfyx}};
    }
};

}  // namespace onednn
}  // namespace cldnn
