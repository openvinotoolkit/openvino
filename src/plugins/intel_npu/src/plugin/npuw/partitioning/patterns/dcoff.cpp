// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dcoff.hpp"

#include "../../lazy_tensor.hpp"
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
            m.weights_to_unpack.insert(i - fbody._param_offset);
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
    std::vector<ov::npuw::weights::LazyTensor> new_lazy_closure;
    std::vector<ov::Tensor> new_scales;
    std::vector<ov::Tensor> new_zerops;

    // For a new_lazy_closure vector by rearranging the old one.  Also
    // reserve a new_scales vector to have the same size, filled with
    // empty tensors by default.
    for (auto&& i : m.closure_remap) {
        new_lazy_closure.push_back(fcall._lazy_closure[i]);
        if (fcall._closure[i]) {
            new_closure.push_back(fcall._closure[i]);
        } else {
            new_closure.push_back(m.weights_to_unpack.count(i) ? fcall._lazy_closure[i].eval() : fcall._closure[i]);
        }

        auto scale_iter = m.scale_remap.find(i);
        auto zerop_iter = m.zerop_remap.find(i);

        new_scales.push_back(scale_iter != m.scale_remap.end() ? fcall._lazy_closure[scale_iter->second].eval()
                                                               : ov::Tensor());
        // Check for asymmetric zero points and add them to new_zerops
        new_zerops.push_back(zerop_iter != m.zerop_remap.end() ? fcall._lazy_closure[zerop_iter->second].eval()
                                                               : m.zero_points[i]);
    }
    fcall._scales = std::move(new_scales);
    fcall._zerops = std::move(new_zerops);
    fcall._closure = std::move(new_closure);
    fcall._lazy_closure = std::move(new_lazy_closure);
}

void finalize_remap(Function& fbody, Subgraph& fsg, const ClosureRemap& m) {
    LOG_DEBUG("Removing retired parameters...");
    LOG_BLOCK();

    // Unfortunate truth - this function has to be aware of the
    // Host Gather existence to properly update indices after
    // Remap.
    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    struct GatherParams {
        PPtr pidx;  // Parameter @ function body - input_ids
        PPtr psrc;  // Parameter @ function body - vocab tensor
        PPtr pdst;  // Parameter @ function body - gathered ids
    };
    GatherParams gather_params;
    const auto& params = fbody._model->get_parameters();
    if (fsg._host_gather.dst_idx != -1) {
        gather_params = GatherParams{params[fsg._host_gather.idx_idx],
                                     params[fsg._host_gather.src_idx],
                                     params[fsg._host_gather.dst_idx]};
    }

    for (auto&& p : m.params_to_remove) {
        LOG_DEBUG("Removing parameter " << p);
        LOG_BLOCK();
        fbody._model->remove_parameter(p);
    }

    // Update indices for gather
    if (fsg._host_gather.dst_idx != -1) {
        fsg._host_gather.idx_idx = fbody._model->get_parameter_index(gather_params.pidx);
        fsg._host_gather.src_idx = fbody._model->get_parameter_index(gather_params.psrc);
        fsg._host_gather.dst_idx = fbody._model->get_parameter_index(gather_params.pdst);
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

DCOFFPassBase::DCOFFPassBase(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref)
    : m_dcoff_mode(dcoff_mode),
      m_dcoff_type(dcoff_type),
      m_params_to(pref) {}

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

DCOFFPassBase::DCOFFPassBase(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref)
    : m_dcoff_mode(dcoff_mode),
      m_dcoff_type(dcoff_type),
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

DCOFFPassReshape2::DCOFFPassReshape2(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref) {
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

// Pattern: Phi-3 4SymW16A
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

DCOFFPassReshape3::DCOFFPassReshape3(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref) {
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

    register_matcher(std::make_shared<opp::Matcher>(cvt, "TagDCOFFPassReshape3"), std::move(callback));
}

// Pattern: i4 group-quant
//
//
//   "tensor"       "scale"           >            "tensor"
//    Param:A       Param:C           >             Param:A
//      i4          f16|f32           >              f16
//       :           :                >               :
//       V          :                 >               V
//     Convert     :                  >            [ Convert]
//     f16|f32    :                   >            [   f32  ]
//        :      :                    >                :
//        V      V                    >                V
//        Multiply                    >              Reshape
//         f16|f32                    >              f16|f32
//            :                       >
//            :                       >
//         Reshape                    >
//         f16|f32                    >

DCOFFPassReshape4::DCOFFPassReshape4(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref) {
    auto paramA = opp::wrap_type<ov::op::v0::Parameter>();
    auto paramC = opp::wrap_type<ov::op::v0::Parameter>();
    auto cvtA = opp::wrap_type<ov::op::v0::Convert>({paramA});
    auto mulply = opp::wrap_type<ov::op::v1::Multiply>({cvtA, paramC});
    auto scalar = opp::wrap_type<ov::op::v0::Constant>();
    auto reshape = opp::wrap_type<ov::op::v1::Reshape>({mulply, scalar});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_nodeA = node_to_output.at(paramA).get_node_shared_ptr();
        auto matched_nodeC = node_to_output.at(paramC).get_node_shared_ptr();

        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeA));
        NPUW_ASSERT(ov::op::util::is_parameter(matched_nodeC));

        auto matched_paramA = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeA);
        auto matched_paramC = std::static_pointer_cast<ov::op::v0::Parameter>(matched_nodeC);

        auto matched_out_mulply = node_to_output.at(mulply);

        if (ov::element::i4 == matched_paramA->get_element_type() &&
            (ov::element::f16 == matched_paramC->get_element_type() ||
             ov::element::f32 == matched_paramC->get_element_type())) {
            LOG_DEBUG("Matched: " << matched_paramA << ", set element type to " << dcoff_type);
            matched_paramA->set_element_type(dcoff_type);

            if (dcoff_mode == DCOffMode::CAST_SCALE) {
                NPUW_ASSERT(dcoff_type == ov::element::f16);

                LOG_DEBUG("Matched: " << matched_paramC << " - parameter to remove...");
                LOG_BLOCK();

                // Record mapping from the Scale coeff parameter to the Real weight parameter
                pref.get().scales[matched_paramC] = matched_paramA;

                std::shared_ptr<ov::Node> new_rshp_in = matched_paramA;
                if (matched_out_mulply.get_element_type() == ov::element::f32) {
                    new_rshp_in = std::make_shared<ov::op::v0::Convert>(matched_paramA, ov::element::f32);
                }

                LOG_DEBUG("Reconnecting the Root...");
                auto matched_reshape = node_to_output.at(reshape).get_node_shared_ptr();
                matched_reshape->input(0).replace_source_output(new_rshp_in);
            }
            LOG_DEBUG("Done");
        }
        return false;  // root node hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(reshape, "TagDCOFFPassReshape4"), std::move(callback));
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
DCOFFPassReshape::DCOFFPassReshape(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref) {
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
