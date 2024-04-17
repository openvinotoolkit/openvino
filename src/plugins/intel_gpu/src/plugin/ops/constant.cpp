// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"

#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {

static void CreateConstantOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Constant>& op) {
    auto const_shape = op->get_output_shape(0);
    auto constFormat = cldnn::format::get_default_format(const_shape.size());

    cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    cldnn::layout const_layout = cldnn::layout(const_shape, out_dtype, constFormat);

    cldnn::primitive_id initialconstPrimID = layer_type_name_ID(op);
    cldnn::primitive_id constPrimID;
    auto data = op->get_data_ptr<char>();

    const auto cache_key = std::make_tuple(data, const_shape, op->get_output_element_type(0));

    auto bufIter = p.blobMemCache.find(cache_key);

    if (bufIter != p.blobMemCache.end()) {
        constPrimID = bufIter->second;
        p.primitive_ids[initialconstPrimID] = constPrimID;
        p.profiling_ids.push_back(initialconstPrimID);
    } else {
        cldnn::memory::ptr mem = nullptr;
        if (const_layout.bytes_count() > 0) {
            mem = p.get_engine().allocate_memory(const_layout, false);
        } else {
            // In the case of empty const data with {0} shape, it has zero byte.
            // To avoid zero byte memory allocation issue, reinterpret one dimension memory to zero dimension memory.
            auto one_dim_layout = cldnn::layout(ov::PartialShape({1}), const_layout.data_type, const_layout.format);
            auto one_dim_mem = p.get_engine().allocate_memory(one_dim_layout, false);
            mem = p.get_engine().reinterpret_buffer(*one_dim_mem, const_layout);
        }

        GPU_DEBUG_LOG << "[" << initialconstPrimID << ": constant] layout: "
                        << const_layout.to_short_string() << ", mem_ptr(" << mem << ", " << mem->size() << " bytes)"<< std::endl;
        auto& stream = p.get_engine().get_service_stream();
        cldnn::mem_lock<char> lock{mem, stream};
        auto buf = lock.data();
        auto bufSize = const_layout.bytes_count();

        std::memcpy(&buf[0], &data[0], bufSize);
        p.add_primitive(*op, cldnn::data(initialconstPrimID, mem));
        p.blobMemCache[cache_key] = initialconstPrimID;
        constPrimID = initialconstPrimID;
    }
}

REGISTER_FACTORY_IMPL(v0, Constant);

}  // namespace intel_gpu
}  // namespace ov
