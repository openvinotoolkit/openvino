// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/rank_constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/roi_align.hpp"
#include "openvino/op/roi_align_rotated.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/tensor_iterator.hpp"

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace op {
namespace internal {
using RankConstant = ov::intel_gpu::op::RankConstant;
}  // namespace internal
}  // namespace op
}  // namespace ov
namespace ov {
namespace intel_gpu {

static cldnn::tensor getConstTensor(const ov::Shape constDims) {
    std::vector<cldnn::tensor::value_type> shuffled_dims(constDims.size());

    // cldnn tensor c-tor expects constants be in a reversed order (x, y, z, w, u, v)
    for (size_t i = 0; i < constDims.size(); i++) {
        shuffled_dims[i] = TensorValue(constDims[i < 2 ? i : (constDims.size() - 1 - i)]);
    }
    cldnn::tensor constTensor;
    switch (constDims.size()) {
    case 8:
    case 7:
        constTensor = cldnn::tensor(shuffled_dims);
        break;
    case 6: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]),
                                        TensorValue(constDims[5]), TensorValue(constDims[4]),
                                        TensorValue(constDims[3]), TensorValue(constDims[2]));
        break;
    case 5: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]),
                                        TensorValue(constDims[4]), TensorValue(constDims[3]), TensorValue(constDims[2]));
        break;
    case 4: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]),
                                        TensorValue(constDims[3]), TensorValue(constDims[2]));
        break;
    case 3: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]),
                                        1, TensorValue(constDims[2]));
        break;
    case 2: constTensor = cldnn::tensor(TensorValue(constDims[0]), TensorValue(constDims[1]), 1, 1);
        break;
    case 1: constTensor = cldnn::tensor(1, TensorValue(constDims[0]), 1, 1);
        break;
    case 0: constTensor = cldnn::tensor(1, 1, 1, 1);
        break;
    default: OPENVINO_THROW("Invalid constant blob dimensions");
    }
    return constTensor;
}

static void CreateRankConstantOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::RankConstant>& op) {
    ov::Shape const_shape = op->get_shape();
    cldnn::tensor constTensor = getConstTensor(const_shape);
    auto constFormat = cldnn::format::get_default_format(const_shape.size());

    // If const_shape has a dimension = 0, then create tensor with single value
    // TODO: check if dim=0 is a valid case
    if (std::accumulate(const_shape.begin(), const_shape.end(), size_t(1), std::multiplies<size_t>()) == 0)
        constTensor = cldnn::tensor{1};

    cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    cldnn::layout constLayout = p.use_new_shape_infer() ? cldnn::layout(const_shape, out_dtype, constFormat) :
                                                          cldnn::layout(out_dtype, constFormat, constTensor);

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
        if (constLayout.bytes_count() > 0) {
            mem = p.get_engine().allocate_memory(constLayout, false);
        } else {
            // In the case of empty const data with {0} shape, it has zero byte.
            // To avoid zero byte memory allocation issue, reinterpret one dimension memory to zero dimension memory.
            auto one_dim_layout = cldnn::layout(ov::PartialShape({1}), constLayout.data_type, constLayout.format);
            auto one_dim_mem = p.get_engine().allocate_memory(one_dim_layout, false);
            mem = p.get_engine().reinterpret_buffer(*one_dim_mem, constLayout);
        }

        GPU_DEBUG_LOG << "[" << initialconstPrimID << ": constant] layout: "
                        << constLayout.to_short_string() << ", mem_ptr(" << mem << ", " << mem->size() << " bytes)"<< std::endl;
        auto& stream = p.get_engine().get_service_stream();
        cldnn::mem_lock<char> lock{mem, stream};
        auto buf = lock.data();
        auto bufSize = constLayout.bytes_count();
        int rank = op->get_rank();
        int offset = rank * bufSize;
        std::memcpy(&buf[0], &data[0] + offset, bufSize);
        p.add_primitive(*op, cldnn::data(initialconstPrimID, mem));
        p.blobMemCache[cache_key] = initialconstPrimID;
        constPrimID = initialconstPrimID;
    }
}

REGISTER_FACTORY_IMPL(internal, RankConstant);

}  // namespace intel_gpu
}  // namespace ov
