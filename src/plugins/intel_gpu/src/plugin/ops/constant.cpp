// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/op/convolution.hpp"

#include "openvino/op/constant.hpp"
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
#include "openvino/op/bucketize.hpp"
#include "openvino/op/util/binary_elementwise_bitwise.hpp"

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

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

struct ConstProperties {
    bool needsBatchInterpretation;
};

static void create_data(ProgramBuilder& p, const ov::Shape& const_shape, const std::shared_ptr<ov::op::v0::Constant>& op, const ConstProperties& props) {
    cldnn::tensor constTensor = getConstTensor(const_shape);
    auto constFormat = cldnn::format::get_default_format(const_shape.size());

    if (props.needsBatchInterpretation) {
        constTensor.batch[0] = static_cast<cldnn::tensor::value_type>(constTensor.count());
        constTensor.feature[0] = 1;
    }

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

        std::memcpy(&buf[0], &data[0], bufSize);
        p.add_primitive(*op, cldnn::data(initialconstPrimID, mem));
        p.blobMemCache[cache_key] = initialconstPrimID;
        constPrimID = initialconstPrimID;
    }
}

static bool is_btiwise(Node* node) {
    return dynamic_cast<const ov::op::util::BinaryElementwiseBitwise*>(node) != nullptr;
}

static void CreateConstantOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Constant>& op) {
    ov::Shape constDims = op->get_shape();
    auto constUsers = op->get_output_target_inputs(0);
    std::unordered_map<std::shared_ptr<ov::op::v0::Constant>, ConstProperties> consts = {
        {op, {false}}
    };

    auto is_binary_eltwise = [&] (ov::Node* op) -> bool {
        if (ov::op::util::is_binary_elementwise_arithmetic(op) ||
            ov::op::util::is_binary_elementwise_logical(op) ||
            ov::op::util::is_binary_elementwise_comparison(op) ||
            is_btiwise(op)) {
            return true;
        } else {
            return false;
        }
    };

    auto is_all_inputs_1d = [&] (ov::Node* op) -> bool {
        for (size_t i = 0; i < op->get_input_size(); i++) {
            auto& in_shape = op->get_input_partial_shape(i);
            if (in_shape.size() > 1)
                return false;
        }
        return true;
    };

    auto is_convert_into_binary_eltwise = [&] (ov::Node* op) -> bool {
        if (ov::is_type<ov::op::v0::Convert>(op)) {
            for (size_t i = 0; i < op->get_output_size(); ++i) {
                auto convertUsers = op->get_output_target_inputs(i);
                for (auto user : convertUsers) {
                    if (is_binary_eltwise(user.get_node()) &&
                        is_all_inputs_1d(user.get_node())) {
                        return true;
                    }
                }
            }
        }
        return false;
    };

    auto is_grouped_conv = [](ov::Node* op) -> bool {
        if (ov::is_type<ov::op::v1::GroupConvolution>(op))
            return true;

        if (ov::is_type<op::Convolution>(op)) {
            return ov::as_type<op::Convolution>(op)->get_groups() > 0;
        }

        return false;
    };
    // WA to inconsistency between input and const 1d tensors
    // For Concat along batch we go with batch interpretation
    // For Gather input we go with batch interpretation
    // Also check if constant users is a backprop convolution - in that case O and I need to be swapped.
    for (auto& node : constUsers) {
        auto outOp = node.get_node();
        if (auto castedOp = dynamic_cast<ov::op::v0::Concat*>(outOp)) {
            if (castedOp->get_axis() == 0) {
                consts[op].needsBatchInterpretation = constDims.size() == 1;
            }
        } else if (((is_binary_eltwise(outOp) || ov::is_type<ov::op::v0::SquaredDifference>(outOp)) && is_all_inputs_1d(outOp)) ||
                     is_convert_into_binary_eltwise(outOp)) {
            consts[op].needsBatchInterpretation = constDims.size() == 1;
        } else if (ov::is_type<ov::op::v1::Gather>(outOp) ||
                   ov::is_type<ov::op::v7::Gather>(outOp) ||
                   ov::is_type<ov::op::v8::Gather>(outOp) ||
                   ov::is_type<ov::op::v1::Split>(outOp) ||
                   ov::is_type<ov::op::v1::VariadicSplit>(outOp)) {
            consts[op].needsBatchInterpretation = constDims.size() == 1;
        } else if (ov::is_type<ov::op::v0::PRelu>(outOp) && node.get_index() == 1) {
            // PReLU slope tensor reshape policy
            //
            // 1. 1-dim slope is handled by 'getConstTensor' (if slope dimension is equal to the feature dimension of input).
            //   ex) [1] --> [1, 1, 1, 1]
            //       [N] --> [1, N, 1, 1]
            //
            // 2. Multi-dims slope tensor is handled by the numpy broadcasting rule that is defined at
            //    'https://docs.openvino.ai/2023.0/openvino_docs_ops_broadcast_rules.html'.
            //   ex) [N, 1, 1] --> [1, N, 1, 1]
            //       [N, M, 1] --> [1, N, M, 1]
            auto input_shape = outOp->get_input_partial_shape(0);
            if ((constDims.size() != 1 && constDims.size() < input_shape.size()) ||
                (constDims.size() == 1 && input_shape.is_static() && static_cast<int64_t>(constDims[0]) != input_shape[1].get_length())) {
                // Reshape 'constDims' according to the numpy broadcasting rule.
                ov::Shape slope_shape(input_shape.size(), 1);
                for (size_t j = 1; j <= constDims.size(); j++)
                    slope_shape[slope_shape.size() - j] = constDims[constDims.size() - j];
                constDims = slope_shape;
            }
        } else if (is_grouped_conv(outOp) && node.get_index() == 1 && !p.use_new_shape_infer()) {
            auto input_shape = outOp->get_input_partial_shape(0);
            if (constDims.size() == 4 && input_shape.size() == 3) { // In case of weight dim 4 and input dim 3,
                constDims.push_back(1);                             // The weight cldnn tensor adds 1d to the end as the input cldnn tensor does
            }
        } else if (ov::is_type<ov::op::v3::ROIAlign>(outOp) || ov::is_type<ov::op::v9::ROIAlign>(outOp) ||
                   ov::is_type<ov::op::v15::ROIAlignRotated>(outOp)) { //< Hacks...
            consts[op].needsBatchInterpretation = constDims.size() == 1;
        } else if ((ov::is_type<ov::op::v5::Loop>(outOp) || ov::is_type<ov::op::v0::TensorIterator>(outOp))) {
            // when inner network has 1d parameter which is connected to outer loop's constant 1d data,
            // outer constant 1d data and inner 1d parameter has same bytes_count but layout is different
            // (outer constant is [1, N, 1, 1] but inner parameter is [N, 1, 1, 1]).
            // To pass check_memory_to_set in input_layout::set_data for this case, Set constDims to [N, 1, 1, 1]
            // when constDims is one dim and user op is Loop or TensorIterator.
            consts[op].needsBatchInterpretation = constDims.size() == 1;
        } else if (ov::is_type<ov::op::v0::Result>(outOp) && !p.use_new_shape_infer() && p.is_inner_program()) {
            // When IF-operation generates branch-true and branch-false,
            // simple nodes for both can be created such as Parameter->Result, Constant->Result
            // And each layout will be like Parameter->Result [N, 1, 1, 1], Constant->Result [1, N, 1, 1], that produces layout mismatch error.
            // For that case, Constant->Result needs to be [N, 1, 1, 1]
            consts[op].needsBatchInterpretation = constDims.size() == 1;
        }
    }

    for (auto& it : consts) {
        create_data(p, constDims, it.first, it.second);
    }
}

REGISTER_FACTORY_IMPL(v0, Constant);

}  // namespace intel_gpu
}  // namespace ov
