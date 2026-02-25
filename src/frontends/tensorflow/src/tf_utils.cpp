// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tf_utils.hpp"

#include <stdint.h>

#include <vector>

#include "helper_ops/merge.hpp"
#include "helper_ops/switch.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::element;
using namespace ov::frontend::tensorflow;
using namespace std;

namespace {
void copy_conditional_flow_marker_with_branches(const CfMarkerType& copy_from,
                                                unordered_map<uint32_t, SetOfBranchIndices>& copy_to_braches) {
    for (const auto& marker : copy_from.existing_markers_with_branches) {
        const auto& switch_marker = marker.first;
        const auto& branch_indices = marker.second;
        copy_to_braches[switch_marker].insert(branch_indices.begin(), branch_indices.end());
    }
}

void copy_conditional_flow_marker_with_switches(const CfMarkerType& copy_from,
                                                unordered_map<uint32_t, SetOfSwitchNodes>& copy_to_switches) {
    for (const auto& marker : copy_from.existing_markers_with_switches) {
        const auto& switch_marker = marker.first;
        const auto& switch_nodes = marker.second;
        copy_to_switches[switch_marker].insert(switch_nodes.begin(), switch_nodes.end());
    }
}

void copy_conditional_flow_markers_for_producer(
    unordered_map<uint32_t, SetOfBranchIndices>& combined_markers_with_braches,
    unordered_map<uint32_t, SetOfSwitchNodes>& combined_markers_with_switches,
    const Output<Node>& producer_output) {
    // walk through all data producer and collect conditional flow markers
    const shared_ptr<const Node>& producer_node = producer_output.get_node_shared_ptr();
    uint32_t branch_index = static_cast<uint32_t>(producer_output.get_index());
    if (!cf_marker_exists(producer_node)) {
        return;
    }
    auto producer_markers = get_cf_marker(producer_node);
    copy_conditional_flow_marker_with_branches(producer_markers, combined_markers_with_braches);
    copy_conditional_flow_marker_with_switches(producer_markers, combined_markers_with_switches);
    // if data goes from Switch node, it needs to create a new marker and branch marker
    for (const auto& new_marker : producer_markers.new_markers) {
        auto switch_nodes = new_marker.second;
        combined_markers_with_braches[new_marker.first].insert(branch_index);
        combined_markers_with_switches[new_marker.first].insert(switch_nodes.begin(), switch_nodes.end());
    }
}

template <typename T>
void extract_tensor_content(const std::string& tensor_content, Tensor* values) {
    const auto tensor_content_size = tensor_content.size();
    FRONT_END_GENERAL_CHECK(tensor_content_size % sizeof(T) == 0,
                            "Size of tensor_content (",
                            tensor_content_size,
                            ") is not a multiple of ",
                            sizeof(T));

    const T* tensor_values = reinterpret_cast<const T*>(tensor_content.data());
    FRONT_END_GENERAL_CHECK(values->get_size() == tensor_content_size / sizeof(T),
                            "Size of tensor is not equal to tensor_content size.");
    copy(tensor_values, tensor_values + tensor_content_size / sizeof(T), values->data<T>());
}

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4244)  // possible loss of data
#    pragma warning(disable : 4267)  // possible loss of data
#endif
template <typename SRC_T, typename DST_T = SRC_T>
void extract_compressed_tensor_content(const ::tensorflow::TensorProto& tensor_proto,
                                       int64_t val_size,
                                       Tensor* values) {
    auto val_lastsaved = static_cast<SRC_T>(0);
    auto values_data = values->data<DST_T>();
    for (size_t i = 0; i < values->get_size(); i++) {
        if (val_size == 0) {
            values_data[i] = static_cast<DST_T>(0);
        } else if (static_cast<int64_t>(i) < val_size) {
            auto val_i = static_cast<SRC_T>(0);
            switch (values->get_element_type()) {
            // TODO: there are more element types to support here
            case boolean:
                val_i = tensor_proto.bool_val()[i];
                break;
            case i32:
                val_i = tensor_proto.int_val()[i];
                break;
            case i64:
                val_i = tensor_proto.int64_val()[i];
                break;
            case f16:
                val_i = float16::from_bits(tensor_proto.half_val()[i]);
                break;
            case f32:
                val_i = tensor_proto.float_val()[i];
                break;
            case f64:
                val_i = tensor_proto.double_val()[i];
                break;
            case u8:
                val_i = tensor_proto.int_val()[i];
                break;
            case u16:
                val_i = tensor_proto.int_val()[i];
                break;
            case u64:
                val_i = tensor_proto.uint64_val()[i];
                break;
            case i8:
                val_i = tensor_proto.int_val()[i];
                break;
            case u32:
                val_i = tensor_proto.uint32_val()[i];
                break;
            case i16:
                val_i = tensor_proto.int_val()[i];
                break;
            default:
                FRONT_END_THROW("Encountered unknown element type " + values->get_element_type().get_type_name());
            }
            values_data[i] = static_cast<DST_T>(val_i);
            val_lastsaved = val_i;
        } else {
            values_data[i] = static_cast<DST_T>(val_lastsaved);
        }
    }
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace

namespace ov {
namespace frontend {
namespace tensorflow {

void copy_conditional_flow_marker(const CfMarkerType& copy_from, CfMarkerType& copy_to) {
    for (const auto& marker : copy_from.existing_markers_with_branches) {
        const auto& switch_marker = marker.first;
        const auto& branch_markers = marker.second;
        copy_to.existing_markers_with_branches[switch_marker].insert(branch_markers.begin(), branch_markers.end());
    }
    for (const auto& marker : copy_from.existing_markers_with_switches) {
        const auto& switch_marker = marker.first;
        const auto& branch_markers = marker.second;
        copy_to.existing_markers_with_switches[switch_marker].insert(branch_markers.begin(), branch_markers.end());
    }
}

bool CfMarkerType::is_copyable() const {
    return false;
}

Type get_ov_type(const ::tensorflow::DataType& type) {
    using ::tensorflow::DataType;

    static map<DataType, Type> type_map{{DataType::DT_FLOAT, f32},
                                        {DataType::DT_DOUBLE, f64},
                                        {DataType::DT_INT32, i32},
                                        {DataType::DT_UINT8, u8},
                                        {DataType::DT_INT16, i16},
                                        {DataType::DT_INT8, i8},
                                        {DataType::DT_INT64, i64},
                                        {DataType::DT_BOOL, boolean},
                                        {DataType::DT_BFLOAT16, bf16},
                                        {DataType::DT_UINT16, u16},
                                        {DataType::DT_HALF, f16},
                                        {DataType::DT_UINT32, u32},
                                        {DataType::DT_UINT64, u64},
                                        {DataType::DT_FLOAT_REF, f32},
                                        {DataType::DT_DOUBLE_REF, f64},
                                        {DataType::DT_INT32_REF, i32},
                                        {DataType::DT_UINT8_REF, u8},
                                        {DataType::DT_INT16_REF, i16},
                                        {DataType::DT_INT8_REF, i8},
                                        {DataType::DT_INT64_REF, i64},
                                        {DataType::DT_BOOL_REF, boolean},
                                        {DataType::DT_BFLOAT16_REF, bf16},
                                        {DataType::DT_UINT16_REF, u16},
                                        {DataType::DT_HALF_REF, f16},
                                        {DataType::DT_UINT32_REF, u32},
                                        {DataType::DT_UINT64_REF, u64},
                                        {DataType::DT_STRING, element::string},
                                        {DataType::DT_STRING_REF, element::string}};

    auto it = type_map.find(type);
    // for all unsupported types return dynamic type
    return it == type_map.end() ? dynamic : it->second;
}

Any unpack_tensor_proto(const ::tensorflow::TensorProto& tensor_proto) {
    return unpack_tensor_proto(tensor_proto, tensor_proto.tensor_shape(), tensor_proto.dtype());
}

Any unpack_tensor_proto(const ::tensorflow::TensorProto& tensor_proto,
                        const ::tensorflow::TensorShapeProto& tensor_shape,
                        const ::tensorflow::DataType& tensor_type) {
    PartialShape pshape;
    for (int i = 0; i < tensor_shape.dim_size(); i++) {
        pshape.push_back(tensor_shape.dim(i).size());
    }
    FRONT_END_GENERAL_CHECK(pshape.is_static(), "Dynamic shapes are not supported for Tensor attribute.");
    Type ov_type = get_ov_type(tensor_type);

    FRONT_END_GENERAL_CHECK(
        ov_type.is_static(),
        "Encountered unknown element type " + DataType_Name(tensor_type) + " on an empty tensor_proto");

    Tensor res(ov_type, pshape.get_shape());
    auto tensor_content = tensor_proto.tensor_content();
    if (!tensor_content.empty() && tensor_proto.has_tensor_shape()) {
        switch (ov_type) {
        case f32:
            extract_tensor_content<float>(tensor_content, &res);
            break;
        case u8:
            extract_tensor_content<uint8_t>(tensor_content, &res);
            break;
        case i64:
            extract_tensor_content<int64_t>(tensor_content, &res);
            break;
        case u16:
            extract_tensor_content<uint16_t>(tensor_content, &res);
            break;
        case u64:
            extract_tensor_content<uint64_t>(tensor_content, &res);
            break;
        case i32:
            extract_tensor_content<int32_t>(tensor_content, &res);
            break;
        case i8:
            extract_tensor_content<int8_t>(tensor_content, &res);
            break;
        case bf16:
            extract_tensor_content<bfloat16>(tensor_content, &res);
            break;
        case u32:
            extract_tensor_content<uint32_t>(tensor_content, &res);
            break;
        case f64:
            extract_tensor_content<double>(tensor_content, &res);
            break;
        case i16:
            extract_tensor_content<int16_t>(tensor_content, &res);
            break;
        case boolean:
            extract_tensor_content<bool>(tensor_content, &res);
            break;
        case f16:
            extract_tensor_content<float16>(tensor_content, &res);
            break;
        case element::string: {
            auto string_val_size = static_cast<size_t>(tensor_proto.string_val_size());
            FRONT_END_GENERAL_CHECK(
                res.get_size() == string_val_size,
                "Internal error: OpenVINO and TensorFlow string tensors contains different number of elements");
            auto string_src = tensor_proto.string_val();
            auto string_dst = res.data<std::string>();
            for (size_t ind = 0; ind < string_val_size; ++ind) {
                string_dst[ind] = string_src[static_cast<int>(ind)];
            }
            break;
        }
        default:
            FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
        }
    } else {
        int64_t val_size = 0;
        switch (ov_type) {
        case f32:
            val_size = tensor_proto.float_val_size();
            extract_compressed_tensor_content<float>(tensor_proto, val_size, &res);
            break;
        case u8:
            val_size = tensor_proto.int_val_size();
            extract_compressed_tensor_content<int32_t, uint8_t>(tensor_proto, val_size, &res);
            break;
        case i64:
            val_size = tensor_proto.int64_val_size();
            extract_compressed_tensor_content<int64_t>(tensor_proto, val_size, &res);
            break;
        case u16:
            val_size = tensor_proto.int_val_size();
            extract_compressed_tensor_content<uint16_t, uint16_t>(tensor_proto, val_size, &res);
            break;
        case u64:
            val_size = tensor_proto.uint64_val_size();
            extract_compressed_tensor_content<uint64_t>(tensor_proto, val_size, &res);
            break;
        case i32:
            val_size = tensor_proto.int_val_size();
            extract_compressed_tensor_content<int32_t>(tensor_proto, val_size, &res);
            break;
        case i8:
            val_size = tensor_proto.int_val_size();
            extract_compressed_tensor_content<int32_t, int8_t>(tensor_proto, val_size, &res);
            break;
        case u32:
            val_size = tensor_proto.uint32_val_size();
            extract_compressed_tensor_content<uint32_t>(tensor_proto, val_size, &res);
            break;
        case f64:
            val_size = tensor_proto.double_val_size();
            extract_compressed_tensor_content<double>(tensor_proto, val_size, &res);
            break;
        case i16:
            val_size = tensor_proto.int_val_size();
            extract_compressed_tensor_content<int32_t, int16_t>(tensor_proto, val_size, &res);
            break;
        case boolean:
            val_size = tensor_proto.bool_val_size();
            extract_compressed_tensor_content<bool>(tensor_proto, val_size, &res);
            break;
        case f16:
            val_size = tensor_proto.half_val_size();
            extract_compressed_tensor_content<float16>(tensor_proto, val_size, &res);
            break;
        case element::string: {
            auto string_val_size = static_cast<size_t>(tensor_proto.string_val_size());
            FRONT_END_GENERAL_CHECK(
                res.get_size() == string_val_size,
                "Internal error: OpenVINO and TensorFlow string tensors contains different number of elements");
            auto string_src = tensor_proto.string_val();
            auto string_dst = res.data<std::string>();
            for (size_t ind = 0; ind < string_val_size; ++ind) {
                string_dst[ind] = string_src[static_cast<int>(ind)];
            }
            break;
        }
        default:
            FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
        }
    }
    return res;
}

CfMarkerType get_cf_marker(const shared_ptr<const Node>& node) {
    auto rt_info = node->get_rt_info();
    FRONT_END_GENERAL_CHECK(rt_info.count(CF_MARKER_TAG) > 0,
                            "[TensorFlow Frontend] internal error: node does not contain conditional flow marker");
    auto& ov_any_cf_marker = rt_info[CF_MARKER_TAG];
    FRONT_END_GENERAL_CHECK(ov_any_cf_marker.is<CfMarkerType>(),
                            "[TensorFlow Frontend] internal error: incorrect type of conditional flow marker");
    return ov_any_cf_marker.as<CfMarkerType>();
}

uint32_t generate_cf_marker() {
    static uint32_t marker = 0;
    return marker++;
}

bool propagate_conditional_flow(const OutputVector& ov_inputs,
                                const frontend::NamedOutputVector& ov_outputs,
                                const set<Output<Node>>& input_control_deps,
                                set<Output<Node>>& output_control_deps) {
    // returns if there is conditional flow to propagate
    // it checks all producer-nodes connected via data edges and control dependencies
    // compute combined markers
    // it is a map from conditional flow marker to a set of branch markers
    unordered_map<uint32_t, SetOfBranchIndices> combined_markers_with_branches;
    unordered_map<uint32_t, SetOfSwitchNodes> combined_markers_with_switches;
    for (const auto& ov_input : ov_inputs) {
        // walk through all input producers and propagate CF marker
        copy_conditional_flow_markers_for_producer(combined_markers_with_branches,
                                                   combined_markers_with_switches,
                                                   ov_input);
    }
    // walk through all control dependencies and collect conditional flow markers
    for (const auto& input_control_dep : input_control_deps) {
        // walk through all control dependencies and collect conditional flow markers
        copy_conditional_flow_markers_for_producer(combined_markers_with_branches,
                                                   combined_markers_with_switches,
                                                   input_control_dep);
    }

    // walk through all nodes and mark them if needed
    bool to_propagate = false;
    for (const auto& ov_output : ov_outputs) {
        const auto& node = ov_output.port.get_node_shared_ptr();

        // skip already marked node
        // it can be a case of Identity node that the conversion rule skips
        if (cf_marker_exists(node)) {
            to_propagate = true;
            continue;
        }

        // if this is Merge node, it needs to eliminate markers with multiple branch markers
        // and put such markers into eliminated list
        CfMarkerType resulted_cf_marker;
        if (as_type_ptr<Merge>(node)) {
            for (const auto& marker : combined_markers_with_branches) {
                auto switch_marker = marker.first;
                auto branch_markers = marker.second;
                if (branch_markers.size() > 1) {
                    if (combined_markers_with_switches.count(switch_marker) > 0) {
                        resulted_cf_marker.merge_eliminated_markers[switch_marker] =
                            combined_markers_with_switches[switch_marker];
                    }
                } else {
                    resulted_cf_marker.existing_markers_with_branches.insert(marker);
                    if (combined_markers_with_switches.count(switch_marker) > 0) {
                        resulted_cf_marker.existing_markers_with_switches[switch_marker] =
                            combined_markers_with_switches[switch_marker];
                    }
                }
            }
        } else if (const auto& switch_node = as_type_ptr<Switch>(node)) {
            // update conditional flow marker with new marker for the current Switch node
            auto switch_marker = switch_node->get_switch_marker();
            resulted_cf_marker.new_markers[switch_marker] = {switch_node};
            resulted_cf_marker.existing_markers_with_branches = combined_markers_with_branches;
            resulted_cf_marker.existing_markers_with_switches = combined_markers_with_switches;
        } else {
            // non-Merge nodes can contain both branch markers of the same conditional flow
            // it can happen if some conditional edge is going directly from Switch node to this non-Merge node
            // it means that the output value is external for If represented with Switch-Merge nodes
            // and must be executed before If
            for (const auto& marker : combined_markers_with_branches) {
                resulted_cf_marker.existing_markers_with_branches.insert(marker);
            }
            resulted_cf_marker.existing_markers_with_switches.insert(combined_markers_with_switches.begin(),
                                                                     combined_markers_with_switches.end());
        }

        // set conditional flow marker only if one of the fields is not empty
        // check if any input or input control dependency contain control flow
        // if yes, it makes sense to continue
        if (resulted_cf_marker.new_markers.size() > 0 || resulted_cf_marker.existing_markers_with_branches.size() > 0 ||
            resulted_cf_marker.existing_markers_with_switches.size() > 0 ||
            resulted_cf_marker.merge_eliminated_markers.size() > 0) {
            set_cf_marker(resulted_cf_marker, node);
            to_propagate = true;
        }
    }

    // compute output control dependencies
    // logically, the next nodes will dependend on outputs and input control dependencies
    output_control_deps.clear();
    if (to_propagate) {
        output_control_deps = input_control_deps;
        for (const auto& ov_output : ov_outputs) {
            output_control_deps.insert(ov_output.port);
        }
    }

    return to_propagate;
}

// create Loop operation corresponding to TensorFlow While operation
shared_ptr<v5::Loop> create_loop_for_tf_while(const std::string& while_node_name,
                                              const shared_ptr<Model>& body_model,
                                              const shared_ptr<Model>& cond_model,
                                              const OutputVector& ov_inputs,
                                              const shared_ptr<Model>& prior_cond_model) {
    size_t input_size = ov_inputs.size();
    // inject condition body graph prior to Loop node
    // to check condition before to start iterations
    auto cond_params = cond_model->get_parameters();
    FRONT_END_GENERAL_CHECK(input_size == cond_params.size(),
                            "[TensorFlow Frontend] internal error: mismatch number of inputs to While and a number of "
                            "inputs in a conditional graph");
    // type setting for body graph parameters is needed for TensorList support since DT_VARIANT type is present
    // also for more accurate execution_condition variable shape deducing we need shape inference for condition graph
    for (size_t input_ind = 0; input_ind < input_size; ++input_ind) {
        cond_params[input_ind]->set_element_type(ov_inputs[input_ind].get_element_type());
        cond_params[input_ind]->set_partial_shape(ov_inputs[input_ind].get_partial_shape());
    }
    cond_model->validate_nodes_and_infer_types();

    if (prior_cond_model) {
        auto prior_cond_params = prior_cond_model->get_parameters();
        FRONT_END_GENERAL_CHECK(
            input_size == prior_cond_params.size(),
            "[TensorFlow Frontend] internal error: mismatch number of inputs to While and a number of "
            "inputs in a conditional graph");
        for (size_t input_ind = 0; input_ind < input_size; ++input_ind) {
            prior_cond_params[input_ind]->set_element_type(ov_inputs[input_ind].get_element_type());
            prior_cond_params[input_ind]->set_partial_shape(ov_inputs[input_ind].get_partial_shape());
        }
        prior_cond_model->validate_nodes_and_infer_types();
    }
    auto cond_prior = prior_cond_model ? prior_cond_model : cond_model->clone();

    ov::OutputVector ov_outputs;
    inject_body_model(cond_prior, while_node_name + "/cond", ov_inputs, ov_outputs);
    FRONT_END_GENERAL_CHECK(
        ov_outputs.size() == 1,
        "[TensorFlow Frontend] Internal error or inconsistent model: condition body must contain one Result node.");
    auto exec_cond = ov_outputs[0];
    auto trip_count = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto loop = make_shared<v5::Loop>(trip_count, exec_cond);

    // prepare body model to be set for the Loop node
    // note that condition should be computed on the updated input
    // because this is while(cond) {} construction,
    // that is why condition graph is stitched to the body results
    auto body_params = body_model->get_parameters();
    auto body_results = body_model->get_results();
    auto cond_results = cond_model->get_results();
    FRONT_END_GENERAL_CHECK(body_params.size() == input_size,
                            "[TensorFlow Frontend] Internal error or inconsistent model: body graph "
                            " must have the same number of Parameter nodes as a number of inputs to While.");
    FRONT_END_GENERAL_CHECK(cond_params.size() == input_size,
                            "[TensorFlow Frontend] Internal error or inconsistent model: condition graph "
                            " must have the same number of Parameter nodes as a number of inputs to While.");
    for (size_t param_ind = 0; param_ind < body_results.size(); ++param_ind) {
        cond_params[param_ind]->output(0).replace(body_results[param_ind]->input_value(0));
    }
    auto body_condition_output_idx = body_results.size();
    // body_results may contain less nodes than body_params that means back edge exists not for all body_params
    for (size_t param_ind = body_condition_output_idx; param_ind < input_size; ++param_ind) {
        cond_params[param_ind]->output(0).replace(body_params[param_ind]->output(0));
    }

    // update body model with the new result that corresponds to execution condition
    FRONT_END_GENERAL_CHECK(
        cond_results.size() == 1 && cond_results[0],
        "[TensorFlow Frontend] Internal error or inconsistent model: condition body must contain one Result node.");
    body_model->add_results(cond_results);
    // type setting for body graph parameters is needed for TensorList support since DT_VARIANT type is present
    for (size_t input_ind = 0; input_ind < input_size; ++input_ind) {
        body_params[input_ind]->set_element_type(ov_inputs[input_ind].get_element_type());
    }

    // set data for the Loop node
    loop->set_function(body_model);

    // body_results may contain less nodes than body_params that means back edge exists not for all body_params
    for (size_t input_ind = 0; input_ind < body_condition_output_idx; ++input_ind) {
        loop->set_merged_input(body_params[input_ind], ov_inputs[input_ind], body_results[input_ind]->input_value(0));
    }
    loop->set_special_body_ports({-1, static_cast<int64_t>(body_condition_output_idx)});
    // set invariant inputs for the loop
    for (size_t input_ind = body_condition_output_idx; input_ind < input_size; ++input_ind) {
        loop->set_invariant_input(body_params[input_ind], ov_inputs[input_ind]);
    }

    // set external outputs for Loop node
    // do not get execution condition outside of the Loop node
    for (size_t output_ind = 0; output_ind < body_condition_output_idx; ++output_ind) {
        loop->get_iter_value(body_results[output_ind]);
    }
    loop->validate_and_infer_types();
    return loop;
}

void inject_body_model(std::shared_ptr<ov::Model> ov_model_to_inject,
                       const std::string& operation_type,
                       const ov::OutputVector& ov_inputs,
                       ov::OutputVector& ov_outputs,
                       const std::vector<std::string>& ov_input_names) {
    ov_outputs.clear();
    auto body_parameters = ov_model_to_inject->get_parameters();
    // some external inputs can be skipped if some body graph inputs turn to be Constant nodes
    FRONT_END_GENERAL_CHECK(body_parameters.size() <= ov_inputs.size(),
                            "[TensorFlow Error] Internal error or incorrect input models: number of "
                            "inputs and arguments to the function " +
                                operation_type + " do not match.");
    for (size_t param_ind = 0; param_ind < body_parameters.size(); ++param_ind) {
        auto param_name = body_parameters[param_ind]->get_friendly_name();
        // find suitable index of external input
        size_t ext_found_ind = param_ind;
        if (ov_input_names.size() > 0) {
            // only used for PartitionedCall translator
            for (size_t ext_input_ind = 0; ext_input_ind < ov_input_names.size(); ++ext_input_ind) {
                if (ov_input_names[ext_input_ind] == param_name) {
                    ext_found_ind = ext_input_ind;
                    break;
                }
            }
        }

        auto orig_type = body_parameters[param_ind]->get_element_type();
        // avoid not needed tensor names from body graph Parameter node after replacing
        body_parameters[param_ind]->output(0).set_names({});
        body_parameters[param_ind]->output(0).replace(ov_inputs[ext_found_ind]);
        if (auto ext_parameter = as_type_ptr<v0::Parameter>(ov_inputs[ext_found_ind].get_node_shared_ptr())) {
            // save type of a Parameter as converted in the body
            // this is important if the external conversion extension is applied to body graph node
            // with setting its own type
            if (orig_type != element::dynamic) {
                ext_parameter->set_element_type(orig_type);
            }
        }
    }
    for (const auto& result_node : ov_model_to_inject->get_results()) {
        ov_outputs.push_back(result_node->input_value(0));
    }
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
