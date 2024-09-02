// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dcoff.hpp"

#include "../../logging.hpp"
#include "../../util.hpp"
#include "../partitioning.hpp"  // Subgraph
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {

namespace pattern_utils {

std::shared_ptr<ov::op::v0::MatMul> get_root_matmul(const ov::Output<Node>& last_node_output) {
    auto consumer = last_node_output.get_target_inputs().begin()->get_node()->shared_from_this();
    // Attempt to cast the consumer node to a MatMul node
    auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(consumer);
    NPUW_ASSERT(matmul_node);
    return matmul_node;
}

bool transpose_required(const std::shared_ptr<ov::op::v0::MatMul>& matmul_node) {
    const auto& input_shape = matmul_node->input_value(1).get_shape();
    const auto& output_shape = matmul_node->output(0).get_shape();

    if (output_shape.back() != input_shape.back()) {
        return true; // Transpose is required
    }

    return false;
}

void transpose_param_shape(std::shared_ptr<ov::op::v0::Parameter>& param) {
    auto partial_shape = param->get_partial_shape();
    NPUW_ASSERT(partial_shape.is_static());
    auto shape = partial_shape.to_shape();

    // Check if the shape is 2D or 3D
    if (shape.size() == 2) {
        // For 2D shapes, swap the dimensions
        std::swap(shape[0], shape[1]);
    } else if (shape.size() == 3) {
        // For 3D shapes, bring the last dimension to the front
        std::rotate(shape.rbegin(), shape.rbegin() + 1, shape.rend());
    }

    // Set the new shape to the parameter
    param->set_partial_shape(ov::PartialShape(shape));
    LOG_DEBUG("Modifying the shape of: " << param << " to " << param->get_partial_shape());
}

std::vector<uint8_t> unpack_u4_to_u8(const uint8_t* packed_data, size_t total_u4_elements) {
    std::vector<uint8_t> unpacked_data(total_u4_elements);
    for (size_t i = 0; i < total_u4_elements; ++i) {
        if (i % 2 == 0) {
            // Even index: take the lower 4 bits directly
            unpacked_data[i] = packed_data[i / 2] & 0x0F;
        } else {
            // Odd index: take the higher 4 bits and shift them to the lower 4 bits
            unpacked_data[i] = (packed_data[i / 2] >> 4) & 0x0F;
        }
    }
    return unpacked_data;
}

std::vector<uint8_t> pack_u8_to_u4(const uint8_t* unpacked_data, size_t total_u4_elements) {
    std::vector<uint8_t> packed_data((total_u4_elements + 1) / 2);
    for (size_t i = 0; i < total_u4_elements; i += 2) {
        // Combine two u4 elements into one uint8_t, with one in the lower 4 bits and the other in the higher 4 bits
        packed_data[i / 2] = (unpacked_data[i] & 0x0F) | ((unpacked_data[i + 1] & 0x0F) << 4);
    }
    // Handle the case where the total number of u4 elements is odd
    if (total_u4_elements % 2 != 0) {
        packed_data[total_u4_elements / 2] = unpacked_data[total_u4_elements - 1] & 0x0F;
    }
    return packed_data;
}

ov::Tensor transpose_u4(const ov::Tensor& tensor) {
    const auto& shape = tensor.get_shape();
    size_t total_u4_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    const uint8_t* packed_data = tensor.data<uint8_t>();
    std::vector<uint8_t> unpacked_data = unpack_u4_to_u8(packed_data, total_u4_elements);

    if (shape.size() == 2) {
        // For a 2D tensor with shape KxM, transpose to MxK
        size_t K = shape[0];
        size_t M = shape[1];

        std::vector<uint8_t> transposed_unpacked_data(K * M);

        for (size_t k = 0; k < K; ++k) {
            for (size_t m = 0; m < M; ++m) {
                transposed_unpacked_data[m * K + k] = unpacked_data[k * M + m];
            }
        }

        std::vector<uint8_t> transposed_packed_data = pack_u8_to_u4(transposed_unpacked_data.data(), total_u4_elements);
        ov::Tensor transposed_tensor(ov::element::u4, {M, K});
        std::memcpy(transposed_tensor.data(), transposed_packed_data.data(), transposed_packed_data.size());
        return transposed_tensor;
    }

    if (shape.size() == 3) {
        // For a 3D tensor with shape KxMxN, transpose to NxKxM
        size_t K = shape[0];
        size_t M = shape[1];
        size_t N = shape[2];

        std::vector<uint8_t> transposed_unpacked_data(K * M * N);

        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t m = 0; m < M; ++m) {
                    transposed_unpacked_data[n * (K * M) + k * M + m] = unpacked_data[k * (M * N) + m * N + n];
                }
            }
        }
        std::vector<uint8_t> transposed_packed_data = pack_u8_to_u4(transposed_unpacked_data.data(), total_u4_elements);
        ov::Tensor transposed_tensor(ov::element::u4, {N, K, M});
        std::memcpy(transposed_tensor.data(), transposed_packed_data.data(), transposed_packed_data.size());
        return transposed_tensor;
    }

    // Invalid case
    NPUW_ASSERT(false);
}

std::vector<int8_t> unpack_i4_to_i8(const int8_t* packed_data, size_t total_i4_elements) {
    std::vector<int8_t> unpacked_data(total_i4_elements);
    for (size_t i = 0; i < total_i4_elements; ++i) {
        if (i % 2 == 0) {
            // Even index: take the lower 4 bits directly
            unpacked_data[i] = packed_data[i / 2] & 0x0F;
        } else {
            // Odd index: take the higher 4 bits and shift them to the lower 4 bits
            unpacked_data[i] = (packed_data[i / 2] >> 4) & 0x0F;
        }
    }
    return unpacked_data;
}

std::vector<int8_t> pack_i8_to_i4(const int8_t* unpacked_data, size_t total_i4_elements) {
    std::vector<int8_t> packed_data((total_i4_elements + 1) / 2, 0);
    for (size_t i = 0; i < total_i4_elements; i += 2) {
        // Combine two i4 elements into one uint8_t, with one in the lower 4 bits and the other in the higher 4 bits
        packed_data[i / 2] = (unpacked_data[i] & 0x0F) | ((unpacked_data[i + 1] & 0x0F) << 4);
    }
    // Handle the case where the total number of i4 elements is odd
    if (total_i4_elements % 2 != 0) {
        packed_data[total_i4_elements / 2] |= unpacked_data[total_i4_elements - 1] & 0x0F;
    }
    return packed_data;
}

ov::Tensor transpose_i4(const ov::Tensor& tensor) {
    const auto& shape = tensor.get_shape();
    size_t total_i4_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    const int8_t* packed_data = tensor.data<int8_t>();
    std::vector<int8_t> unpacked_data = unpack_i4_to_i8(packed_data, total_i4_elements);

    if (shape.size() == 2) {
        // For a 2D tensor with shape KxM, transpose to MxK
        size_t K = shape[0];
        size_t M = shape[1];

        std::vector<int8_t> transposed_unpacked_data(K * M);

        for (size_t k = 0; k < K; ++k) {
            for (size_t m = 0; m < M; ++m) {
                transposed_unpacked_data[m * K + k] = unpacked_data[k * M + m];
            }
        }

        std::vector<int8_t> transposed_packed_data = pack_i8_to_i4(transposed_unpacked_data.data(), total_i4_elements);
        ov::Tensor transposed_tensor(ov::element::i4, {M, K});
        std::memcpy(transposed_tensor.data(), transposed_packed_data.data(), transposed_packed_data.size());
        return transposed_tensor;
    }

    if (shape.size() == 3) {
        // For a 3D tensor with shape KxMxN, transpose to NxKxM
        size_t K = shape[0];
        size_t M = shape[1];
        size_t N = shape[2];

        std::vector<int8_t> transposed_unpacked_data(K * M * N);

        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t m = 0; m < M; ++m) {
                    transposed_unpacked_data[n * (K * M) + k * M + m] = unpacked_data[k * (M * N) + m * N + n];
                }
            }
        }
        std::vector<int8_t> transposed_packed_data = pack_i8_to_i4(transposed_unpacked_data.data(), total_i4_elements);
        ov::Tensor transposed_tensor(ov::element::i4, {N, K, M});
        std::memcpy(transposed_tensor.data(), transposed_packed_data.data(), transposed_packed_data.size());
        return transposed_tensor;
    }

    // Invalid case
    NPUW_ASSERT(false);
}

ov::Tensor transpose_f16(const ov::Tensor& tensor) {
    const auto& shape = tensor.get_shape();
    const int16_t* data = reinterpret_cast<const int16_t*>(tensor.data());

    if (shape.size() == 2) {
        // For a 2D tensor with shape KxM, transpose to MxK
        size_t K = shape[0];
        size_t M = shape[1];

        ov::Tensor transposed_tensor(ov::element::f16, {M, K});
        int16_t* transposed_data = reinterpret_cast<int16_t*>(transposed_tensor.data());

        for (size_t k = 0; k < K; ++k) {
            for (size_t m = 0; m < M; ++m) {
                transposed_data[m * K + k] = data[k * M + m];
            }
        }
        return transposed_tensor;
    } else if (shape.size() == 3) {
        // For a 3D tensor with shape KxMxN, transpose to NxKxM
        size_t K = shape[0];
        size_t M = shape[1];
        size_t N = shape[2];

        ov::Tensor transposed_tensor(ov::element::f16, {N, K, M});
        int16_t* transposed_data = reinterpret_cast<int16_t*>(transposed_tensor.data());

        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t m = 0; m < M; ++m) {
                    transposed_data[n * (K * M) + k * M + m] = data[k * (M * N) + m * N + n];
                }
            }
        }
        return transposed_tensor;
    }

    // Invalid case
    NPUW_ASSERT(false);
}

ov::Tensor transpose_f32(const ov::Tensor& tensor) {
    const auto& shape = tensor.get_shape();
    const float* data = tensor.data<float>();

    if (shape.size() == 2) {
        // For a 2D tensor with shape KxM, transpose to MxK
        size_t K = shape[0];
        size_t M = shape[1];
        
        ov::Tensor transposed_tensor(ov::element::f32, {M, K});
        float* transposed_data = transposed_tensor.data<float>();

        for (size_t k = 0; k < K; ++k) {
            for (size_t m = 0; m < M; ++m) {
                transposed_data[m * K + k] = data[k * M + m];
            }
        }
        return transposed_tensor;
    } else if (shape.size() == 3) {
        // For a 3D tensor with shape KxMxN, transpose to NxKxM
        size_t K = shape[0];
        size_t M = shape[1];
        size_t N = shape[2];

        ov::Tensor transposed_tensor(ov::element::f32, {N, K, M});
        float* transposed_data = transposed_tensor.data<float>();

        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t m = 0; m < M; ++m) {
                    transposed_data[n * (K * M) + k * M + m] = data[k * (M * N) + m * N + n];
                }
            }
        }
        return transposed_tensor;
    }

    // Invalid case
    NPUW_ASSERT(false);
}

ov::Tensor transpose_tensor(const ov::Tensor& tensor) {
    switch (tensor.get_element_type()) {
        case ov::element::u4:
            return transpose_u4(tensor);
        case ov::element::i4:
            return transpose_i4(tensor);
        case ov::element::f16:
            return transpose_f16(tensor);
        case ov::element::f32:
            return transpose_f32(tensor);
        default:
            NPUW_ASSERT(false);
    }
}

}  // namespace matmul_utils

namespace patterns {

namespace opp = ov::pass::pattern;

// The update procedure is tricky: The closure vector needs to be
// freed of the scale coefficient tensors which are no longer the
// arguments to the function.  At the same time, these scale
// coefficients need to be recorded elsewhere.

// How does the procedure look like:
// - Take the function body;
// - Walk through its parameters, starting with base offset (the base
//   offset indicates where the closure parameters start)
// - If a Parameter is found in the params_to_scale map, it is
//   a Scaling factor Parameter for a compressed Weight:
//   1. This parameter [i] is removed from the _parameters list
//   2. Scale remap [k++] is set to [i-base] -> meaning the k'th
//      Scale tensor will be taken from [i-base]'th closure
//      - also remember which Closure tensor this Scale tensor stands
//        for
// - If a Parameter is NOT found, this tensor doesn't need
//   scaling/decompression:
//   1. The Const remap [n++] is set to [i-base] -> meaning the n'th
//      Const tensor will be taken from [i-base]'th closure in the
//      updated closure tensor.

ClosureRemap build_remap(const Function& fbody, const DCOFFParams& params_to) {
    LOG_DEBUG("Creating a closure remap for " << fbody._model->get_friendly_name());
    LOG_BLOCK();

    const auto& body_params = fbody._model->get_parameters();
    LOG_DEBUG("There is " << body_params.size() << " parameters for this function");

    ClosureRemap m;

    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    std::unordered_set<PPtr> ban_list;

    for (const auto& scale_pair : params_to.scales) {
        ban_list.insert(scale_pair.first);
    }

    for (const auto& zerop_pair : params_to.zerops_asymm) {
        ban_list.insert(zerop_pair.second);
    }

    // FIXME: use indexed() here instead
    for (std::size_t i = fbody._param_offset; i < body_params.size(); i++) {
        const auto& param = body_params[i];
        LOG_DEBUG("Checking the function parameter " << param);
        LOG_BLOCK();

        if(params_to.transpose_required.find(param) != params_to.transpose_required.end()) {
            m.transpose_indices.push_back(i - fbody._param_offset);
        }

        // First find among scale factors...
        auto pscale_iter = params_to.scales.find(param);
        auto pzerop_iter = params_to.zerops_asymm.find(param);
        if (pscale_iter != params_to.scales.end()) {
            LOG_DEBUG("This is a Scale factor parameter, will be removed");
            auto& pscale_weight_param = pscale_iter->second;
            auto pscale_weight_pindex = fbody._model->get_parameter_index(pscale_weight_param);
            auto pscale_weight_cindex = pscale_weight_pindex - fbody._param_offset;
            m.scale_remap[pscale_weight_cindex] = i - fbody._param_offset;
            m.params_to_remove.push_back(param);
        } else if (pzerop_iter != params_to.zerops_asymm.end()) {
            LOG_DEBUG("There is an Asymmetric zero point corresponding to this parameter, it will be removed");
            auto zerop_pindex = fbody._model->get_parameter_index(pzerop_iter->second);
            auto zerop_cindex = zerop_pindex - fbody._param_offset;
            m.zerop_remap[i - fbody._param_offset] = zerop_cindex;
            m.params_to_remove.push_back(pzerop_iter->second);
            m.closure_remap.push_back(i - fbody._param_offset);
        } else if (ban_list.find(param) == ban_list.end()) {
            // If it's not in the ban list, it's an OK parameter and should be kept
            LOG_DEBUG("This is an OK parameter, will be kept");
            m.closure_remap.push_back(i - fbody._param_offset);
        }

        // Process zero points for parameters
        auto zerop_iter = params_to.zerops.find(param);
        if (zerop_iter != params_to.zerops.end()) {
            LOG_DEBUG("This parameter requires zero point: " << zerop_iter->second);
            m.zero_points.push_back(ov::npuw::util::tensor_from_const(zerop_iter->second));
        } else {
            m.zero_points.push_back(ov::Tensor());
        }
    }
    NPUW_ASSERT((body_params.size() - fbody._param_offset) ==
                (m.scale_remap.size() + m.closure_remap.size() + m.zerop_remap.size()));
    NPUW_ASSERT((body_params.size() - fbody._param_offset) == m.zero_points.size());

    LOG_DEBUG("DONE");
    return m;
}

void apply_remap(Subgraph& fcall, const ClosureRemap& m) {
    std::vector<ov::Tensor> new_closure;
    std::vector<ov::Tensor> new_scales;
    std::vector<ov::Tensor> new_zerops;

    // For a new_closure vector by rearranging the old one.  Also
    // reserve a new_scales vector to have the same size, filled with
    // empty tensors by default.
    for (auto&& i : m.closure_remap) {
        new_closure.push_back(fcall._closure[i]);
        
        auto scale_iter = m.scale_remap.find(i);
        new_scales.push_back(scale_iter != m.scale_remap.end() ? fcall._closure[scale_iter->second] : ov::Tensor());
        // Check for asymmetric zero points and add them to new_zerops
        auto zerop_iter = m.zerop_remap.find(i);
        const auto& zerop = zerop_iter != m.zerop_remap.end() ? fcall._closure[zerop_iter->second] : m.zero_points[i];
        new_zerops.push_back(zerop);
    }

    for (auto&& i : m.transpose_indices) {
        if (std::find(m.closure_remap.begin(), m.closure_remap.end(), i) != m.closure_remap.end()) {
            new_scales[i] = pattern_utils::transpose_tensor(new_scales[i]);
        } else {
            auto it_scale = std::find_if(m.scale_remap.begin(), m.scale_remap.end(),
                                     [i](const std::pair<std::size_t, std::size_t>& pair) {
                                         return pair.second == i;
                                     });
            if (it_scale != m.scale_remap.end()) {
                new_scales[i] = pattern_utils::transpose_tensor(new_scales[i]);
                continue;
            }
            auto it_zerop = std::find_if(m.zerop_remap.begin(), m.zerop_remap.end(),
                                     [i](const std::pair<std::size_t, std::size_t>& pair) {
                                         return pair.second == i;
                                     });
            if (it_zerop != m.zerop_remap.end()) {
                new_zerops[i] = pattern_utils::transpose_tensor(new_zerops[i]);
            }
        }
    }

    fcall._closure = std::move(new_closure);
    fcall._scales = std::move(new_scales);
    fcall._zerops = std::move(new_zerops);
}

void finalize_remap(Function& fbody, const ClosureRemap& m) {
    LOG_DEBUG("Removing retired parameters...");
    LOG_BLOCK();
    for (auto&& p : m.params_to_remove) {
        LOG_DEBUG("Removing parameter " << p);
        LOG_BLOCK();
        fbody._model->remove_parameter(p);
    }
    fbody._model->validate_nodes_and_infer_types();
    LOG_DEBUG("DONE");
}

////////////////////////////////////////////////////////////////////////////////
//
// Decompression Cut-off patterns
//
////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
// Pattern: 4/8SymW16A
//
// In the diagram below, pattern on the left is identified and
// is modified to pattern in the middle:
//
//   Parameter:A  Parameter:B      Parameter:A  Parameter:B      Parameter:A
//   int4|int8    fp32|fp16           fp16      fp32|fp16             fp16
//         :      :            ->        :      :            ->        :
//         V      :                      V      :                      V
//        Convert :                     Convert :                     Convert
//          fp32  :            ->         fp32  :            ->         fp32
//           :    :                        :    :                        :
//           V    V                        V    V                        :
//    (...) Multiply           ->   (...) Multiply           ->   (...)  :
//       :    fp32                     :    fp32                     :   :
//       :     :                       :     :                       :   :
//       V     V               ->      V     V               ->      V   V
//  [MatMul|Gather]                 [MatMul|Gather]               [MatMul|Gather]
//        fp32                          fp32                          fp32
//         :                   ->        :                   ->        :
//         V                             V                             V
//
// An easy change, but how does it work? The key here is that the
// original closure tensor (Parameter:A) stays the same in the
// closure, we just make the function body think it is no longer
// int4/int8 but fp16 now. The int4/int8-to-fp16 conversion will
// happen on host (e.g., CPU) as part of the function prologue.
//
// What else we need to do here? Just store that this Parameter:A
// requires such manual conversion so we don't forget about this
// in the function prologue.
//
// If OPENVINO_NPUW_DCOFF_SCALE is YES, then the Multiply in the
// above graph also removed from the function body (as well as
// its Parameter B).
namespace SymmNoZP {

DCOFFPassBase::DCOFFPassBase(DCOffMode dcoff_mode,
                             ov::element::Type dcoff_type,
                             bool enable_transpose,
                             DCOFFParamRef pref)
    : m_dcoff_mode(dcoff_mode),
      m_dcoff_type(dcoff_type),
      m_params_to(pref),
      m_enable_transpose(enable_transpose) {}

void DCOFFPassBase::build() {
    paramA = opp::wrap_type<ov::op::v0::Parameter>();
    paramB = opp::wrap_type<ov::op::v0::Parameter>();
    toFP32 = opp::wrap_type<ov::op::v0::Convert>({paramA});
    mulply = opp::wrap_type<ov::op::v1::Multiply>({toFP32, paramB});
}

bool DCOFFPassBase::matcher_callback(ov::pass::pattern::Matcher& m) {
    auto& node_to_output = m.get_pattern_value_map();
    auto matched_nodeA = node_to_output.at(paramA).get_node_shared_ptr();
    NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeA));

    auto matched_paramA = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeA);
    auto element_type = matched_paramA->get_element_type();
    if (element_type == ov::element::i4 || element_type == ov::element::i8) {
        LOG_DEBUG("Matched: " << matched_paramA << ", set element type to " << m_dcoff_type);
        matched_paramA->set_element_type(m_dcoff_type);

        if (m_dcoff_mode == DCOffMode::CAST_SCALE) {
            LOG_DEBUG("Removing Multiply as part of DCOFF...");
            LOG_BLOCK();

            NPUW_ASSERT(m_dcoff_type == ov::element::f16);
            // Off-graph scaling works only for f16 target data type.
            // Extra transformation here: remove multiply, mark paramB for removal
            // MatMul/Gather will be reconnected to Convert directly

            auto matched_nodeB = node_to_output.at(paramB).get_node_shared_ptr();
            NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeB));

            auto matched_paramB = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeB);
            LOG_DEBUG("Matched: " << matched_paramB << " - parameter to remove...");

            // Record mapping from the Scale coeff paramter to the Real weight parameter
            m_params_to.get().scales[matched_paramB] = matched_paramA;

            // Disconnect Multiply and Convert from their outputs
            auto matched_mulply = node_to_output.at(mulply).get_node_shared_ptr();
            auto matched_convrt = node_to_output.at(toFP32).get_node_shared_ptr();
            auto drop_outputs = [](std::shared_ptr<ov::Node> node) {
                for (auto&& node_outputs : node->outputs()) {
                    for (auto&& node_reader_port : node_outputs.get_target_inputs()) {
                        node_outputs.remove_target_input(node_reader_port);
                    }
                }
            };
            LOG_DEBUG("Dropping the connections...");
            drop_outputs(matched_mulply);
            drop_outputs(matched_convrt);

            LOG_DEBUG("Reconnecting the root...");
            reconnect_root_to_convert(m);

            LOG_DEBUG("Done");
        }
    }
    return false;  // root node hasn't changed
}

void DCOFFPassMatMul::build() {
    DCOFFPassBase::build();
    auto _mmin1 = opp::any_input();
    matmul = opp::wrap_type<ov::op::v0::MatMul>({_mmin1, mulply});
    register_matcher(std::make_shared<opp::Matcher>(matmul, "TagDCOFFMatMul"),
                     std::bind(&DCOFFPassMatMul::matcher_callback, this, std::placeholders::_1));
}

void DCOFFPassMatMul::reconnect_root_to_convert(ov::pass::pattern::Matcher& m) {
    // In this pattern, Convert goes to the MatMul's (root) 1st (0-based) input
    auto& node_to_output = m.get_pattern_value_map();
    auto matched_convrt = node_to_output.at(toFP32).get_node_shared_ptr();
    auto matched_matmul = node_to_output.at(matmul).get_node_shared_ptr();
    matched_matmul->input(1).replace_source_output(matched_convrt);
}

void DCOFFPassGather::build() {
    DCOFFPassBase::build();
    auto _gin2 = opp::any_input();
    auto _gin3 = opp::any_input();
    gather = opp::wrap_type<ov::op::v8::Gather>({mulply, _gin2, _gin3});
    register_matcher(std::make_shared<opp::Matcher>(gather, "TagDCOFFGather"),
                     std::bind(&DCOFFPassGather::matcher_callback, this, std::placeholders::_1));
}

void DCOFFPassGather::reconnect_root_to_convert(ov::pass::pattern::Matcher& m) {
    // In this pattern, Convert goes to the Gathers's (root) 0's input
    auto& node_to_output = m.get_pattern_value_map();
    auto matched_convrt = node_to_output.at(toFP32).get_node_shared_ptr();
    auto matched_gather = node_to_output.at(gather).get_node_shared_ptr();
    matched_gather->input(0).replace_source_output(matched_convrt);
}

}  // namespace SymmNoZP

//------------------------------------------------------------------------------
// Pattern: SymmZP, and in fact is used for GPTQ.
//
namespace SymmZP {

// As seen in ChatGLM3 and a newer LLaMa-v2:
// Since it is Symm, all zero points for all blocks must have the same
// value so NPUW will detect it and fuse to function body (so it is
// not Parameter but Const).
//
// In the diagram below, pattern on the left is identified and
// is modified to pattern in the right if type is promoted to f16
//
//   "tensor"     "zero point" "scale"
//   Parameter:A  Parameter:B  Parameter:C  >    Parameter:A
//                Const:B
//         u4      u4|f32       f16|f32     >       f16
//         :         :          :           >        :
//         V         :         :            >        V
//        Convert  Convert    :             >       Convert
//        f16|f32   f16      :              >          f32  <Const>
//            :      :      :               >           :     :
//            V      V     :                >           V     V
//            Subtract    :                 >        Reshape|Convert
//              f16|f32  :                  >             :
//               :      :                   >             V
//               V      V                   >
//               Multiply                   >
//               fp16|f32 <Const>           >
//                  :     :                 >
//                  V     V                 >
//              Reshape|Convert             >
//                    :                     >
//                    V                     >
//

DCOFFPassBase::DCOFFPassBase(DCOffMode dcoff_mode,
                             ov::element::Type dcoff_type,
                             bool enable_transpose,
                             DCOFFParamRef pref)
    : m_dcoff_mode(dcoff_mode),
      m_dcoff_type(dcoff_type),
      m_enable_transpose(enable_transpose),
      m_params_to(pref) {}

void DCOFFPassBase::build() {
    paramA = opp::wrap_type<ov::op::v0::Parameter>();
    constB = opp::wrap_type<ov::op::v0::Constant>();
    paramC = opp::wrap_type<ov::op::v0::Parameter>();
    cvtA = opp::wrap_type<ov::op::v0::Convert>({paramA});
    cvtB = opp::wrap_type<ov::op::v0::Convert>({constB});
    subtr = opp::wrap_type<ov::op::v1::Subtract>({cvtA, cvtB});
    mulply = opp::wrap_type<ov::op::v1::Multiply>({subtr, paramC});
}


bool DCOFFPassBase::matcher_callback(ov::pass::pattern::Matcher& m) {
    auto& node_to_output = m.get_pattern_value_map();
    auto matched_nodeA = node_to_output.at(paramA).get_node_shared_ptr();
    auto matched_nodeB = node_to_output.at(constB).get_node_shared_ptr();
    auto matched_nodeC = node_to_output.at(paramC).get_node_shared_ptr();

    NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeA));
    NPUW_ASSERT(ov::op::util::is_constant(matched_nodeB));
    NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeC));

    auto matched_paramA = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeA);
    auto matched_valueB = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeB);
    auto matched_paramC = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeC);

    if (ov::element::u4 == matched_paramA->get_element_type() &&
        ov::element::u4 == matched_valueB->get_element_type() &&
        ov::element::f16 == matched_paramC->get_element_type()) {
        LOG_DEBUG("Matched: " << matched_paramA << ", set element type to " << m_dcoff_type);
        matched_paramA->set_element_type(m_dcoff_type);

        ov::Output<Node> last_node_output = get_last_node_output(m);
        auto matched_MM = pattern_utils::get_root_matmul(last_node_output);
        const bool need_transpose = pattern_utils::transpose_required(matched_MM);
        if (m_enable_transpose && need_transpose) {
            m_params_to.get().transpose_required.insert(matched_paramA);
            m_params_to.get().transpose_required.insert(matched_paramC);
            pattern_utils::transpose_param_shape(matched_paramA);
            pattern_utils::transpose_param_shape(matched_paramC);
            matched_MM->set_transpose_b(true);
        }

        if (m_dcoff_mode == DCOffMode::CAST_SCALE) {
            NPUW_ASSERT(m_dcoff_type == ov::element::f16);

            LOG_DEBUG("Matched: " << matched_valueB << " - value to remove...");
            LOG_DEBUG("Matched: " << matched_paramC << " - parameter to remove...");
            LOG_BLOCK();

            // Extra transformation here:
            // - remove Subtract + Multiply,
            // - mark paramB and paramC for removal.
            // Rshape will be reconnected to Convert directly (TODO:
            // it can be probably eliminated as well)

            // Record mapping from the Scale coeff paramter to the Real weight parameter
            m_params_to.get().zerops[matched_paramA] = matched_valueB;
            m_params_to.get().scales[matched_paramC] = matched_paramA;

            // Disconnect Multiply and Convert from their outputs
            auto matched_mulply = node_to_output.at(mulply).get_node_shared_ptr();
            auto matched_convrt = node_to_output.at(cvtA).get_node_shared_ptr();
            auto drop_outputs = [](std::shared_ptr<ov::Node> node) {
                for (auto&& node_outputs : node->outputs()) {
                    for (auto&& node_reader_port : node_outputs.get_target_inputs()) {
                        node_outputs.remove_target_input(node_reader_port);
                    }
                }
            };
            LOG_DEBUG("Dropping the connections...");
            drop_outputs(matched_mulply);
            drop_outputs(matched_convrt);

            LOG_DEBUG("Reconnecting the root...");
            reconnect_root(m);
        }
        LOG_DEBUG("Done");
    }
    return false;  // root node hasn't changed
}

void DCOFFPassReshape1::build() {
    DCOFFPassBase::build();
    auto scalar = opp::wrap_type<ov::op::v0::Constant>();
    reshpe = opp::wrap_type<ov::op::v1::Reshape>({mulply, scalar});
    register_matcher(std::make_shared<opp::Matcher>(reshpe, "TagDCOFFReshape1"),
                     std::bind(&DCOFFPassReshape1::matcher_callback, this, std::placeholders::_1));
}

void DCOFFPassReshape1::reconnect_root(ov::pass::pattern::Matcher& m) {
    auto& node_to_output = m.get_pattern_value_map();
    auto matched_convrt = node_to_output.at(cvtA).get_node_shared_ptr();
    auto matched_reshpe = node_to_output.at(reshpe).get_node_shared_ptr();
    matched_reshpe->input(0).replace_source_output(matched_convrt);
}

ov::Output<Node> DCOFFPassReshape1::get_last_node_output(ov::pass::pattern::Matcher& m) const {
        auto& node_to_output = m.get_pattern_value_map();
        return node_to_output.at(reshpe).get_node_shared_ptr()->output(0);
}

void DCOFFPassConvert1::build() {
    DCOFFPassBase::build();
    cvtEnd = opp::wrap_type<ov::op::v0::Convert>({mulply});
    register_matcher(std::make_shared<opp::Matcher>(cvtEnd, "TagDCOFFConvert1"),
                     std::bind(&DCOFFPassConvert1::matcher_callback, this, std::placeholders::_1));
}

void DCOFFPassConvert1::reconnect_root(ov::pass::pattern::Matcher& m) {
    // FIXME: Two converts can be further squashed into one!
    auto& node_to_output = m.get_pattern_value_map();
    auto matched_convrt = node_to_output.at(cvtA).get_node_shared_ptr();
    auto matched_cvtEnd = node_to_output.at(cvtEnd).get_node_shared_ptr();
    matched_cvtEnd->input(0).replace_source_output(matched_convrt);
}

ov::Output<Node> DCOFFPassConvert1::get_last_node_output(ov::pass::pattern::Matcher& m) const {
        auto& node_to_output = m.get_pattern_value_map();
        return node_to_output.at(cvtEnd).get_node_shared_ptr()->output(0);
}

//------------------------------------------------------------------------------
// Pattern: LlamaGPTQ
//
// As seen in llama2_7B_chat_GPTQ:
// Since it is Symm, all zero points for all blocks must have the same
// value so NPUW will detect it and fuse to function body (so it is
// not Parameter but Const).
//
// In the diagram below, pattern on the left is identified and
// is modified to pattern in the right if type is promoted to f16
//
//   "tensor"     "zero point" "scale"
//   Parameter:A  Const:B      Parameter:C  >    Parameter:A
//         u4       f32         f32         >       f16
//         :         :          :           >        :
//         V         :         :            >        V
//        Convert    :        :             >       Convert
//         f32       :       :              >          f32
//            :      :      :               >           :
//            V      V     :                >           V
//            Subtract    :                 >        Reshape
//              f32      :                  >             :
//               :      :                   >             V
//               V      V                   >
//               Multiply                   >
//                 f32  <Const>             >
//                  :      :                >
//                  V      V                >
//                   Reshape                >
//                      :                   >
//                      V                   >
//

DCOFFPassReshape2::DCOFFPassReshape2(DCOffMode dcoff_mode, ov::element::Type dcoff_type,  bool enable_transpose, DCOFFParamRef pref) {
    auto paramA = opp::wrap_type<ov::op::v0::Parameter>();
    auto constB = opp::wrap_type<ov::op::v0::Constant>();
    auto paramC = opp::wrap_type<ov::op::v0::Parameter>();
    auto cvtA = opp::wrap_type<ov::op::v0::Convert>({paramA});
    auto subtr = opp::wrap_type<ov::op::v1::Subtract>({cvtA, constB});
    auto mulply = opp::wrap_type<ov::op::v1::Multiply>({subtr, paramC});

    auto scalar = opp::wrap_type<ov::op::v0::Constant>();
    auto reshpe = opp::wrap_type<ov::op::v1::Reshape>({mulply, scalar});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_nodeA = node_to_output.at(paramA).get_node_shared_ptr();
        auto matched_nodeB = node_to_output.at(constB).get_node_shared_ptr();
        auto matched_nodeC = node_to_output.at(paramC).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeA));
        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeB));
        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeC));

        auto matched_paramA = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeA);
        auto matched_valueB = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeB);
        auto matched_paramC = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeC);

        if (ov::element::u4 == matched_paramA->get_element_type() &&
            ov::element::f32 == matched_valueB->get_element_type() &&
            ov::element::f32 == matched_paramC->get_element_type()) {
            LOG_DEBUG("Matched: " << matched_paramA << ", set element type to " << dcoff_type);
            matched_paramA->set_element_type(dcoff_type);

            if (dcoff_mode == DCOffMode::CAST_SCALE) {
                NPUW_ASSERT(dcoff_type == ov::element::f16);

                LOG_DEBUG("Matched: " << matched_valueB << " - value to remove...");
                LOG_DEBUG("Matched: " << matched_paramC << " - parameter to remove...");
                LOG_BLOCK();

                // Extra transformation here:
                // - remove Subtract + Multiply,
                // - mark paramC for removal.
                // Reshape will be reconnected to Convert directly

                // Record mapping from the Scale coeff parameter to the Real weight parameter
                pref.get().zerops[matched_paramA] = matched_valueB;
                pref.get().scales[matched_paramC] = matched_paramA;

                // Disconnect Multiply and Convert from their outputs
                auto matched_mulply = node_to_output.at(mulply).get_node_shared_ptr();
                auto matched_convrt = node_to_output.at(cvtA).get_node_shared_ptr();
                auto drop_outputs = [](std::shared_ptr<ov::Node> node) {
                    for (auto&& node_outputs : node->outputs()) {
                        for (auto&& node_reader_port : node_outputs.get_target_inputs()) {
                            node_outputs.remove_target_input(node_reader_port);
                        }
                    }
                };
                LOG_DEBUG("Dropping the connections...");
                drop_outputs(matched_mulply);
                drop_outputs(matched_convrt);

                LOG_DEBUG("Reconnecting the Root...");
                auto matched_reshpe = node_to_output.at(reshpe).get_node_shared_ptr();
                matched_reshpe->input(0).replace_source_output(matched_convrt);
            }
            LOG_DEBUG("Done");
        }
        return false;  // root node hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(reshpe, "TagDCOFFReshape2"), std::move(callback));
}

// Pattern: Phi-3 4SymW16A/GPTQ
//
//
//   "tensor"       "scale"           >            "tensor"
//    Param:A       Param:C           >             Param:A
//      i4          f16|f32           >              f16
//       :           :                >               :
//       V          :                 >               V
//     Convert     :                  >              Convert
//     f16|f32    :                   >                f32
//        :      :                    >
//        V      V                    >
//        Multiply                    >
//         f16|f32                    >
//            :                       >
//            :                       >
//            V                       >
//         Convert

DCOFFPassReshape3::DCOFFPassReshape3(DCOffMode dcoff_mode, ov::element::Type dcoff_type, bool enable_transpose, DCOFFParamRef pref) {
    auto paramA = opp::wrap_type<ov::op::v0::Parameter>();
    auto paramC = opp::wrap_type<ov::op::v0::Parameter>();
    auto cvtA = opp::wrap_type<ov::op::v0::Convert>({paramA});
    auto mulply = opp::wrap_type<ov::op::v1::Multiply>({cvtA, paramC});
    auto cvt = opp::wrap_type<ov::op::v0::Convert>({mulply});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_nodeA = node_to_output.at(paramA).get_node_shared_ptr();
        auto matched_nodeC = node_to_output.at(paramC).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeA));
        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeC));

        auto matched_paramA = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeA);
        auto matched_paramC = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeC);

        if (ov::element::i4 == matched_paramA->get_element_type() &&
            (ov::element::f16 == matched_paramC->get_element_type() ||
             ov::element::f32 == matched_paramC->get_element_type())) {
            LOG_DEBUG("Matched: " << matched_paramA << ", set element type to " << dcoff_type);
            matched_paramA->set_element_type(dcoff_type);

            if (dcoff_mode == DCOffMode::CAST_SCALE) {
                NPUW_ASSERT(dcoff_type == ov::element::f16);

                LOG_DEBUG("Matched: " << matched_paramC << " - parameter to remove...");
                LOG_BLOCK();

                // Extra transformation here:
                // - remove Multiply + Intermediate Convert
                // - mark paramC for removal.
                // Convert will be reconnected to paramA directly.

                // Record mapping from the Scale coeff parameter to the Real weight parameter
                pref.get().scales[matched_paramC] = matched_paramA;

                // Disconnect Multiply and Convert from their outputs
                auto matched_mulply = node_to_output.at(mulply).get_node_shared_ptr();
                auto matched_convrt = node_to_output.at(cvtA).get_node_shared_ptr();
                auto drop_outputs = [](std::shared_ptr<ov::Node> node) {
                    for (auto&& node_outputs : node->outputs()) {
                        for (auto&& node_reader_port : node_outputs.get_target_inputs()) {
                            node_outputs.remove_target_input(node_reader_port);
                        }
                    }
                };
                LOG_DEBUG("Dropping the connections...");
                drop_outputs(matched_mulply);
                drop_outputs(matched_convrt);

                LOG_DEBUG("Reconnecting the Root...");
                auto matched_cvt = node_to_output.at(cvt).get_node_shared_ptr();
                matched_cvt->input(0).replace_source_output(matched_paramA);
            }
            LOG_DEBUG("Done");
        }
        return false;  // root node hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(cvt, "TagDCOFFPassCWAI3"), std::move(callback));
}

//------------------------------------------------------------------------------
// Pattern: 4SymW16A for CWAI
//
// Note: it is the same pattern as in above, but it is called in the different
// function processing pipeline and at a different stage. The purpose is different
// too - preserve Scale tensors in the function bodies when folding is not done.
// So it doesn't really transform anything, just collecting the information
//
// FIXME: Think how it can be unified with the above
//
//   "tensor"   "zero point"  "scale"
//    Const:A      Const:B    Const:C
//         u4      u4|f32    f16|f32
//         :         :          :
//         V         :         :
//        Convert  Convert    :
//        f16|f32   f16      :
//            :      :      :
//            V      V     :
//            Subtract    :
//              f16|f32  :
//               :      :
//               V      V
//               Multiply
//               fp16|f32

CWAI1::CWAI1(CWAI1::Results scales) {
    auto constA = opp::wrap_type<ov::op::v0::Constant>();
    auto constB = opp::wrap_type<ov::op::v0::Constant>();
    auto constC = opp::wrap_type<ov::op::v0::Constant>();
    auto cvtA = opp::wrap_type<ov::op::v0::Convert>({constA});
    auto cvtB = opp::wrap_type<ov::op::v0::Convert>({constB});
    auto subtr = opp::wrap_type<ov::op::v1::Subtract>({cvtA, cvtB});
    auto mulply = opp::wrap_type<ov::op::v1::Multiply>({subtr, constC});

    auto matcher_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_nodeA = node_to_output.at(constA).get_node_shared_ptr();
        auto matched_nodeB = node_to_output.at(constB).get_node_shared_ptr();
        auto matched_nodeC = node_to_output.at(constC).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeA));
        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeB));
        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeC));

        auto matched_valueA = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeA);
        auto matched_valueB = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeB);
        auto matched_valueC = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeC);

        if (ov::element::u4 == matched_valueA->get_element_type() &&
            (ov::element::u4 == matched_valueB->get_element_type() ||
             ov::element::f32 == matched_valueB->get_element_type()) &&
            (ov::element::f16 == matched_valueC->get_element_type() ||
             ov::element::f32 == matched_valueC->get_element_type())) {
            LOG_DEBUG("Matched: " << matched_valueC);
            scales.get().push_back(matched_valueC);
        }
        return true;
    };  // matcher_callback

    register_matcher(std::make_shared<opp::Matcher>(mulply, "TagCWAI1"), std::move(matcher_callback));
}

// FIXME: Think how it can be unified with the above. THIS is the GPTQ verision
//
//   "tensor"   "zero point"  "scale"
//    Const:A      Const:B    Const:C
//         u4       f32       f16|f32
//         :         :          :
//         V         :         :
//        Convert    :        :
//           f32     :       :
//            :      :      :
//            V      V     :
//            Subtract    :
//              f16|f32  :
//               :      :
//               V      V
//               Multiply
//               fp16|f32

CWAI2::CWAI2(CWAI2::Results scales) {
    auto constA = opp::wrap_type<ov::op::v0::Constant>();
    auto constB = opp::wrap_type<ov::op::v0::Constant>();
    auto constC = opp::wrap_type<ov::op::v0::Constant>();
    auto cvtA = opp::wrap_type<ov::op::v0::Convert>({constA});
    auto subtr = opp::wrap_type<ov::op::v1::Subtract>({cvtA, constB});
    auto mulply = opp::wrap_type<ov::op::v1::Multiply>({subtr, constC});

    auto matcher_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_nodeA = node_to_output.at(constA).get_node_shared_ptr();
        auto matched_nodeB = node_to_output.at(constB).get_node_shared_ptr();
        auto matched_nodeC = node_to_output.at(constC).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeA));
        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeB));
        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeC));

        auto matched_valueA = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeA);
        auto matched_valueB = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeB);
        auto matched_valueC = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeC);

        if (ov::element::u4 == matched_valueA->get_element_type() &&
            ov::element::f32 == matched_valueB->get_element_type() &&
            (ov::element::f16 == matched_valueC->get_element_type() ||
             ov::element::f32 == matched_valueC->get_element_type())) {
            LOG_DEBUG("Matched: " << matched_valueC);
            scales.get().push_back(matched_valueC);
        }
        return true;
    };  // matcher_callback

    register_matcher(std::make_shared<opp::Matcher>(mulply, "TagCWAI2"), std::move(matcher_callback));
}

// Pattern: Phi-3 4SymW16A/GPTQ for CWAI
//
// FIXME: Think how it can be unified with the above
//
//   "tensor"       "scale"
//    Const:A       Const:C
//      i4          f16|f32
//       :           :
//       V          :
//     Convert     :
//     f16|f32    :
//        :      :
//        V      V
//        Multiply
//         f16|f32

CWAI3::CWAI3(CWAI3::Results scales) {
    auto constA = opp::wrap_type<ov::op::v0::Constant>();
    auto constC = opp::wrap_type<ov::op::v0::Constant>();
    auto cvtA = opp::wrap_type<ov::op::v0::Convert>({constA});
    auto mulply = opp::wrap_type<ov::op::v1::Multiply>({cvtA, constC});

    auto matcher_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_nodeA = node_to_output.at(constA).get_node_shared_ptr();
        auto matched_nodeC = node_to_output.at(constC).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeA));
        NPUW_ASSERT(ov::op::util::is_constant(matched_nodeC));

        auto matched_valueA = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeA);
        auto matched_valueC = std::static_pointer_cast<ov::op::v0::Constant>(matched_nodeC);

        if (ov::element::i4 == matched_valueA->get_element_type() &&
            (ov::element::f16 == matched_valueC->get_element_type() ||
             ov::element::f32 == matched_valueC->get_element_type())) {
            LOG_DEBUG("Matched: " << matched_valueC);
            scales.get().push_back(matched_valueC);
        }
        return true;
    };  // matcher_callback

    register_matcher(std::make_shared<opp::Matcher>(mulply, "TagCWAI3"), std::move(matcher_callback));
}

// As seen in LLaMa-v2-7b:
// Since it is Symm, all zero points for all blocks must have the same
// value so NPUW will detect it and fuse to function body (so it is
// not Parameter but Const).

// In the diagram below, pattern on the left is identified and
// is modified to pattern in the right:
//
//   "tensor"     "zero point" "scale"
//   Parameter:A  Parameter:B  Parameter:C  >    Parameter:A
//                Const:B                   >
//         u4       f32         f32         >       f16
//         :         :          :           >        :
//         V         :         :            >        V
//        Convert    :        :             >       Convert
//        f16|f32    :       :              >          f32  <Const>
//            :      :      :               >           :     :
//            V      V     :                >           V     V
//            Subtract    :                 >           Reshape
//              f16|f32  :                  >             :
//               :      :                   >             V
//               V      V                   >
//               Multiply                   >
//               fp16|f32 <Const>           >
//                  :     :                 >
//                  V     V                 >
//                  Reshape                 >
//                    :                     >
//                    V                     >
//

// Implementation TBD

}  // namespace SymmZP

//------------------------------------------------------------------------------
// Pattern: ASymmZP, weights with asymmetric quantization
//
namespace AsymmZP {
// As seen in asymmetric TinyLlama:
// Since it is ASymm, all zero points for all blocks have different
// values so they will be Parameters but not Constants.
//
// In the diagram below, pattern on the left is identified and
// is modified to pattern in the right if type is promoted to f16
//
//   "tensor"     "zero point" "scale"
//   Parameter:A  Parameter:B  Parameter:C  >    Parameter:A
//         u4        u4        f16          >       f16     <Const>
//         :         :          :           >        :         :
//         V         :         :            >        V         V
//        Convert  Convert    :             >       Reshape|Convert
//           f16    f16      :              >
//            :      :      :               >
//            V      V     :                >
//            Subtract    :                 >
//              f16      :                  >
//               :      :                   >
//               V      V                   >
//               Multiply                   >
//               fp16  <Const>              >
//                  :     :                 >
//                  V     V                 >
//              Reshape|Convert             >
//                    :                     >
//                    V                     >
//
DCOFFPassReshape::DCOFFPassReshape(DCOffMode dcoff_mode, ov::element::Type dcoff_type, bool enable_transpose, DCOFFParamRef pref) {
    auto paramA = opp::wrap_type<ov::op::v0::Parameter>();
    auto paramB = opp::wrap_type<ov::op::v0::Parameter>();
    auto paramC = opp::wrap_type<ov::op::v0::Parameter>();
    auto cvtA = opp::wrap_type<ov::op::v0::Convert>({paramA});
    auto cvtB = opp::wrap_type<ov::op::v0::Convert>({paramB});
    auto subtr = opp::wrap_type<ov::op::v1::Subtract>({cvtA, cvtB});
    auto mulply = opp::wrap_type<ov::op::v1::Multiply>({subtr, paramC});

    auto scalar = opp::wrap_type<ov::op::v0::Constant>();
    auto reshpe = opp::wrap_type<ov::op::v1::Reshape>({mulply, scalar});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_nodeA = node_to_output.at(paramA).get_node_shared_ptr();
        auto matched_nodeB = node_to_output.at(paramB).get_node_shared_ptr();
        auto matched_nodeC = node_to_output.at(paramC).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeA));
        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeB));
        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeC));

        auto matched_paramA = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeA);
        auto matched_paramB = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeB);
        auto matched_paramC = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeC);

        if (ov::element::u4 == matched_paramA->get_element_type() &&
            ov::element::u4 == matched_paramB->get_element_type() &&
            ov::element::f16 == matched_paramC->get_element_type()) {
            LOG_DEBUG("Matched: " << matched_paramA << ", set element type to " << dcoff_type);
            matched_paramA->set_element_type(dcoff_type);

            if (dcoff_mode == DCOffMode::CAST_SCALE) {
                NPUW_ASSERT(dcoff_type == ov::element::f16);

                LOG_DEBUG("Matched: " << matched_paramB << " - value to remove...");
                LOG_DEBUG("Matched: " << matched_paramC << " - parameter to remove...");
                LOG_BLOCK();

                // Extra transformation here:
                // - remove Subtract + Multiply,
                // - mark paramC for removal.
                // Reshape will be reconnected to ParamA directly

                // Record mapping from the Scale coeff parameter to the Real weight parameter
                pref.get().zerops_asymm[matched_paramA] = matched_paramB;
                pref.get().scales[matched_paramC] = matched_paramA;

                // Disconnect Multiply and Convert from their outputs
                auto matched_mulply = node_to_output.at(mulply).get_node_shared_ptr();
                auto matched_convrt = node_to_output.at(cvtA).get_node_shared_ptr();
                auto drop_outputs = [](std::shared_ptr<ov::Node> node) {
                    for (auto&& node_outputs : node->outputs()) {
                        for (auto&& node_reader_port : node_outputs.get_target_inputs()) {
                            node_outputs.remove_target_input(node_reader_port);
                        }
                    }
                };
                LOG_DEBUG("Dropping the connections...");
                drop_outputs(matched_mulply);
                drop_outputs(matched_convrt);

                LOG_DEBUG("Reconnecting the Root...");
                auto matched_reshpe = node_to_output.at(reshpe).get_node_shared_ptr();
                matched_reshpe->input(0).replace_source_output(matched_paramA);
            }
            LOG_DEBUG("Done");
        }
        return false;  // root node hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(reshpe, "TagDCOFFReshape"), std::move(callback));
}
}  // namespace AsymmZP
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
