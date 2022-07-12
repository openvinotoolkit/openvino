// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "to_string_utils.h"
#include "register.hpp"
#include "utils.hpp"

#include "quantize_inst.h"
#include "reorder_inst.h"

#include "reorder/reorder_weights_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"

#include <fstream>
#include <vector>
#include <list>
#include <utility>

#include <oneapi/dnnl/dnnl.hpp>

#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined __INTEL_COMPILER || defined _MSC_VER
#            define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) || defined(__clang__)
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif

#ifndef _WIN32
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        include <codecvt>
#        include <locale>
#    endif
#else
#    include <Windows.h>
#endif

namespace cldnn {
namespace onednn {

static std::mutex cacheAccessMutex;

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
static std::wstring multiByteCharToWString(const char* str) {
#ifdef _WIN32
    int strSize = static_cast<int>(std::strlen(str));
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str, strSize, NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, strSize, &wstrTo[0], size_needed);
    return wstrTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
#endif  // _WIN32
}
#endif  // defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)


static std::vector<uint8_t> load_cached_binary(std::string path) {
    std::lock_guard<std::mutex> lock(cacheAccessMutex);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = multiByteCharToWString(path.c_str());
    const wchar_t* filename = widefilename.c_str();
    FILE *fp = _wfopen(filename, L"rb");
#else
    const char* filename = path.c_str();
    FILE *fp = fopen(filename, "rb");
#endif

    if (fp) {
        fseek(fp, 0, SEEK_END);
        auto sz = ftell(fp);
        if (sz < 0) {
            fclose(fp);
            return {};
        }
        auto nsize = static_cast<size_t>(sz);

        fseek(fp, 0, SEEK_SET);

        std::vector<uint8_t> ret(nsize);

        auto res = fread(ret.data(), sizeof(uint8_t), nsize, fp);
        (void)res;
        fclose(fp);
        return ret;
    }

    return {};
}

static void store_cache_blob_on_disk(std::string path, std::vector<uint8_t> cache_blob) {
    std::lock_guard<std::mutex> lock(cacheAccessMutex);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = multiByteCharToWString(path.c_str());
    const wchar_t* filename = widefilename.c_str();
#else
    const char* filename = path.c_str();
#endif
    std::ofstream out_file(filename, std::ios::out | std::ios::binary);
    if (out_file.is_open()) {
        out_file.write(reinterpret_cast<const char*>(&cache_blob[0]), cache_blob.size());
    } else {
        throw std::runtime_error("Could not store cl_cache to " + path);
    }
}

template <class PType, class DescType, class PrimDescType = dnnl::primitive_desc, class PrimType = dnnl::primitive>
struct typed_primitive_onednn_impl : public typed_primitive_impl<PType> {
    const typed_program_node<PType>& _outer;
    std::shared_ptr<DescType> _desc;
    std::shared_ptr<dnnl::primitive_attr> _attrs;
    PrimDescType _pd;
    PrimType _prim;
    std::unordered_map<uint32_t, std::unordered_map<int, dnnl::memory>> _args;

    typed_primitive_onednn_impl(const typed_program_node<PType>& arg,
                                std::shared_ptr<DescType> desc,
                                std::shared_ptr<dnnl::primitive_attr> attrs,
                                const PrimDescType& pd,
                                kernel_selector::WeightsReorderParams weights_reorder = {})
        : typed_primitive_impl<PType>(weights_reorder, pd.impl_info_str()),
          _outer(arg),
          _desc(desc),
          _attrs(attrs),
          _pd(pd) {
            build_primitive();
        }

    bool is_cpu() const override { return false; }

private:
    std::string get_cache_path() const {
        auto path = _outer.get_program().get_engine().configuration().kernels_cache_path;
        if (path.empty()) {
            return {};
        }

        if (path.back() != '/' && path.back() != '\\') {
            path += "/";
        }
        return path;
    }

    std::string get_cache_filepath(std::vector<uint8_t> key) const {
        auto path = get_cache_path();
        if (path.empty()) {
            return {};
        }

        std::string key_str(key.begin(), key.end());
        size_t hash = std::hash<std::string>()(key_str);
        return path + std::to_string(hash) + ".onednn.cl_cache";
    }

    void build_primitive() {
        auto cache_outpath = get_cache_path();
        if (cache_outpath.empty()) {
            _prim = PrimType(_pd);
        } else {
            std::vector<uint8_t> key = _pd.get_cache_blob_id();
            assert(!key.empty());

            std::vector<uint8_t> cache = load_cached_binary(get_cache_filepath(key));
            if (cache.empty()) {
                _prim = PrimType(_pd);
                cache = _prim.get_cache_blob();
                store_cache_blob_on_disk(get_cache_filepath(key), cache);
            } else {
                _prim = PrimType(_pd, cache);
            }
        }
    }

protected:
    virtual bool optimized_out(typed_primitive_inst<PType>&) const { return false; }

    static bool has_output_scales(const std::shared_ptr<dnnl::primitive_attr>& attr) {
        int mask;
        std::vector<float> scales;
        attr->get_output_scales(mask, scales);
        const auto drfv = reinterpret_cast<const int32_t&>(DNNL_RUNTIME_F32_VAL);
        return !scales.empty() && (reinterpret_cast<const int32_t&>(scales[0]) == drfv);
    }

    static bool has_zero_points(int arg, const std::shared_ptr<dnnl::primitive_attr>& attr) {
        int mask;
        std::vector<int32_t> zp;
        attr->get_zero_points(arg, mask, zp);
        const auto drsv = reinterpret_cast<const int32_t&>(DNNL_RUNTIME_S32_VAL);
        return !zp.empty() && (reinterpret_cast<const int32_t&>(zp[0]) == drsv);
    }

    void configure_post_ops_arguments(typed_primitive_inst<PType>& instance, std::unordered_map<int, dnnl::memory>& args) const {
        auto& node = instance.get_node();
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        // Get current post-ops info
        auto onednn_attrs = node.get_onednn_primitive_attributes();
        dnnl::post_ops post_ops = onednn_attrs->get_post_ops();

        // Create onednn memory buffers for post-ops
        auto& cur_post_ops = node.get_fused_primitives_onednn();
        auto post_ops_size = cur_post_ops.size();
        for (size_t post_op_idx = 0, num_of_optimized_post_ops = 0; post_op_idx < post_ops_size; post_op_idx++) {
            auto post_op_type = cur_post_ops[post_op_idx].op_type;
            auto memory_offset = cur_post_ops[post_op_idx].mem_offset;
            auto onednn_post_op_idx = has_output_scales(onednn_attrs) && post_op_idx > 0 ? post_op_idx - 1 : post_op_idx;
            onednn_post_op_idx -= num_of_optimized_post_ops;

            switch (post_op_type) {
                case onednn_post_op_type::eltwise_act:
                case onednn_post_op_type::eltwise_clip:
                case onednn_post_op_type::eltwise_linear:
                case onednn_post_op_type::eltwise_round:
                {
                    // onednn elwise doesn't need any data from memory buffers
                    break;
                }

                case onednn_post_op_type::binary_add:
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

                case onednn_post_op_type::scale:
                {
                    auto scale_op_mem = instance.fused_memory(memory_offset);
                    dnnl::memory::desc desc = onednn::layout_to_memory_desc(scale_op_mem->get_layout(), dnnl::memory::format_tag::a, true);
                    args.insert({DNNL_ARG_ATTR_OUTPUT_SCALES, scale_op_mem->get_onednn_memory(desc)});
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
            auto offset = onednn::get_f_offset(instance.node.input().get_output_layout(), _pd.dnnl::primitive_desc_base::src_desc(0));
            args.insert({DNNL_ARG_SRC, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(0), offset)});
        }

        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_f_offset(instance.node.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }

        configure_post_ops_arguments(instance, args);

        return args;
    }

    void init_kernels() override { }

    event::ptr aggregate_events(const std::vector<event::ptr>& events, stream& stream, bool group = false, bool is_output = false) const {
        if (events.size() == 1 && !is_output)
            return events[0];

        if (group && !is_output)
            return stream.group_events(events);

        return stream.enqueue_marker(events, is_output);
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        uint32_t net_id = instance.get_network().get_id();
        _args[net_id] = get_arguments(instance);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<PType>& instance) override {
        auto& network = instance.get_network();
        auto& engine = network.get_engine();
        auto& stream = network.get_stream();
        auto profiling = engine.configuration().enable_profiling;
        auto net_id = network.get_id();
        event::ptr event;

        if (profiling) {
            stream.finish();
            event = stream.create_user_event(false);
        }

        if (!instance.can_be_optimized()) {
            _prim.execute(stream.get_onednn_stream(), _args[net_id]);
        }

        if (profiling) {
            stream.finish();
            event->set();
        }

        return event;
    }
};

}  // namespace onednn
}  // namespace cldnn
