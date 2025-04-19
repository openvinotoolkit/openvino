// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/graph_util.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "ov_ops/moe_expert.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/moe_expert.hpp"
#include <oneapi/dnnl/dnnl.hpp>
#include <tuple>

namespace ov::intel_gpu {

struct reorder_helper {
    reorder_helper(cldnn::engine& eng, cldnn::stream_ptr stream) : _eng(eng), _stream(stream) {
        _dnnl_eng = eng.get_onednn_engine();
    }
    dnnl::reorder& get_reorder(int K, int N, cldnn::data_types out_dtype) {
        if (_reorders.count({K, N, out_dtype}) == 0) {
            auto src_md = dnnl::memory::desc({K, N}, convert_data_type(out_dtype), dnnl::memory::format_tag::ba);
            auto dst_md = dnnl::memory::desc({K, N}, convert_data_type(out_dtype), dnnl::memory::format_tag::ab);
            auto src_mem = dnnl::memory(src_md, _dnnl_eng);
            auto dst_mem = dnnl::memory(dst_md, _dnnl_eng);
            auto reorder_pd = dnnl::reorder::primitive_desc(_dnnl_eng, src_md, _dnnl_eng, dst_md);
            _reorders[{K, N, out_dtype}] = dnnl::reorder(reorder_pd);
        }
        return _reorders[{K, N, out_dtype}];
    }
    static dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
        switch (dt) {
        case cldnn::data_types::f32:
            return dnnl::memory::data_type::f32;
        case cldnn::data_types::f16:
            return dnnl::memory::data_type::f16;
        case cldnn::data_types::i8:
            return dnnl::memory::data_type::s8;
        case cldnn::data_types::u8:
            return dnnl::memory::data_type::u8;
        case cldnn::data_types::i32:
            return dnnl::memory::data_type::s32;
        case cldnn::data_types::i4:
            return dnnl::memory::data_type::s4;
        case cldnn::data_types::u4:
            return dnnl::memory::data_type::u4;
        default:
            throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to onednn type");
        }
    }
    dnnl::memory convert2dnnl(const cldnn::memory::ptr& ptr, const std::vector<int64_t>& dim, dnnl::memory::format_tag tag, int offset = 0) {
        return ptr->get_onednn_memory(dnnl::memory::desc(dnnl::memory::dims(dim), convert_data_type(ptr->get_layout().data_type), tag), offset);
    }
    cldnn::memory::ptr convert(const std::shared_ptr<ov::Node>& node, const cldnn::layout& new_layout, bool transpose = true) {
        auto op = ov::as_type_ptr<ov::op::v0::Constant>(node);
        ov::Shape const_shape = op->get_shape();
        OPENVINO_ASSERT(const_shape.size() == 3, "convert expects rank is 3, current is ", const_shape.size());
        auto constFormat = cldnn::format::get_default_format(const_shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
        auto layout = cldnn::layout(const_shape, out_dtype, constFormat);
        auto data = op->get_data_ptr<uint8_t>();
        cldnn::memory::ptr mem;
        if (transpose) {
            auto raw_ptr = get_tmp_buf(layout);
            raw_ptr->copy_from(*_stream, data, 0, 0, layout.bytes_count(), true);
            auto N = static_cast<int>(const_shape[0]);
            auto K = static_cast<int>(const_shape[1] * const_shape[2]);
            auto& reorder = get_reorder(K, N, out_dtype);
            mem = _eng.allocate_memory(new_layout, cldnn::allocation_type::usm_device, false);
            std::unordered_map<int, dnnl::memory> reorder_args;
            reorder_args.insert({DNNL_ARG_SRC, convert2dnnl(raw_ptr, {K, N}, dnnl::memory::format_tag::ba)});
            reorder_args.insert({DNNL_ARG_DST, convert2dnnl(mem, {K, N}, dnnl::memory::format_tag::ab)});
            reorder.execute(_stream->get_onednn_stream(), reorder_args);
            _stream->get_onednn_stream().wait();
        } else {
            mem = _eng.allocate_memory(layout, cldnn::allocation_type::usm_device, false);
            mem->copy_from(*_stream, data, 0, 0, layout.bytes_count(), true);
        }
        return mem;
    }
    cldnn::memory::ptr get_tmp_buf(const cldnn::layout& layout) {
        if (layout.bytes_count() > _buf_size) {
            _buf = _eng.allocate_memory(layout, cldnn::allocation_type::usm_device, false);
            _buf_size = _buf->size();
        }
        return _eng.reinterpret_buffer(*_buf, layout);
    }
    std::map<std::tuple<int, int, cldnn::data_types>, dnnl::reorder> _reorders;
    cldnn::memory::ptr _buf;
    cldnn::engine& _eng;
    dnnl::engine _dnnl_eng;
    cldnn::stream_ptr _stream;
    size_t _buf_size = 0;
};

static void prepare_weights(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert2>& op, std::vector<cldnn::moe_expert::mlp_params>& params) {
    const auto& bodys = op->get_body();
    auto& engine = p.get_engine();
    ExecutionConfig config;
    // Onednn engine currently does NOT support out_of_order
    config.set_property(ov::intel_gpu::queue_type(ov::intel_gpu::QueueTypes::in_order));
    auto stream = engine.create_stream(config);
    engine.create_onednn_engine(config);
    reorder_helper helper(p.get_engine(), stream);
    auto alloc = [&] (const std::shared_ptr<ov::Node>& node, bool is_weight = false) {
        auto op = ov::as_type_ptr<ov::op::v0::Constant>(node);
        ov::Shape const_shape = op->get_shape();
        auto constFormat = cldnn::format::get_default_format(const_shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
        cldnn::memory::ptr dst;
        if (is_weight) {
            dst = helper.convert(op, {}, false);
        } else {
            auto new_shape = ov::Shape{const_shape[1], const_shape[0], 1};
            auto layout = cldnn::layout(new_shape, out_dtype, constFormat);
            dst = helper.convert(op, layout);
        }
        return dst;
    };
    OPENVINO_ASSERT(op->get_config().expert_num == bodys.size());
    params.resize(bodys.size());
    for (size_t i = 0; i < bodys.size(); i++) {
    //ov::parallel_for(bodys.size(), [&](size_t i) {
        auto internal_body = bodys[i];
        for (auto& node : internal_body->get_ordered_ops()) {
            auto& rt = node->get_rt_info();
            if (rt.count("__weight_const__")) {
                auto idx = rt["__weight_const__"].as<int>();
                OPENVINO_ASSERT(idx >= 0 && idx < 3);
                params[i].param[idx].weight = alloc(node, true);
            }
            if (rt.count("__scale_const__")) {
                auto idx = rt["__scale_const__"].as<int>();
                OPENVINO_ASSERT(idx >= 0 && idx < 3);
                params[i].param[idx].scale = alloc(node);
            }
            if (rt.count("__zp_const__")) {
                auto idx = rt["__zp_const__"].as<int>();
                OPENVINO_ASSERT(idx >= 0 && idx < 3);
                params[i].param[idx].zp = alloc(node);
            }
        }
    }
}

static void CreateMOEExpert2Op(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOEExpert2>& op) {
    auto inputs = p.GetInputInfo(op);
    OPENVINO_ASSERT(inputs.size() == 4, "Inputs count should be 4");

    const std::string layerName = layer_type_name_ID(op);
    std::vector<cldnn::moe_expert::mlp_params> params;
    prepare_weights(p, op, params);

    const cldnn::moe_expert moe(layerName,
                                inputs,
                                op->get_config(),
                                params
                                );

    p.add_primitive(*op, moe);
}

REGISTER_FACTORY_IMPL(internal, MOEExpert2);

}  // namespace ov::intel_gpu
