// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tf_utils.hpp"

#include <stdint.h>

#include <vector>

#include "helper_ops/merge.hpp"
#include "helper_ops/switch.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov;
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
void extract_tensor_content(const string& tensor_content, Tensor* values) {
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
template <typename T>
void extract_compressed_tensor_content(const ::tensorflow::TensorProto& tensor_proto,
                                       int64_t val_size,
                                       Tensor* values) {
    auto val_lastsaved = static_cast<T>(0);
    auto values_data = values->data<T>();
    for (size_t i = 0; i < values->get_size(); i++) {
        if (val_size == 0) {
            values_data[i] = static_cast<T>(0);
        } else if (static_cast<int64_t>(i) < val_size) {
            auto val_i = static_cast<T>(0);
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
            default:
                FRONT_END_THROW("Encountered unknown element type " + values->get_element_type().get_type_name());
            }
            values_data[i] = val_i;
            val_lastsaved = val_i;
        } else {
            values_data[i] = val_lastsaved;
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
    static const map<::tensorflow::DataType, Type> type_map{{::tensorflow::DataType::DT_BOOL, boolean},
                                                            {::tensorflow::DataType::DT_INT16, i16},
                                                            {::tensorflow::DataType::DT_INT32, i32},
                                                            {::tensorflow::DataType::DT_INT64, i64},
                                                            {::tensorflow::DataType::DT_HALF, f16},
                                                            {::tensorflow::DataType::DT_FLOAT, f32},
                                                            {::tensorflow::DataType::DT_DOUBLE, f64},
                                                            {::tensorflow::DataType::DT_UINT8, u8},
                                                            {::tensorflow::DataType::DT_INT8, i8},
                                                            {::tensorflow::DataType::DT_BFLOAT16, bf16}};

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

    if (tensor_type != ::tensorflow::DataType::DT_STRING) {
        FRONT_END_GENERAL_CHECK(
            ov_type.is_static(),
            "Encountered unknown element type " + DataType_Name(tensor_type) + " on an empty tensor_proto");
    } else {
        auto data = vector<string>();
        for (const auto& item : tensor_proto.string_val()) {
            data.push_back(item);
        }
        return data;
    }
    Tensor res(ov_type, pshape.get_shape());
    auto tensor_content = tensor_proto.tensor_content();
    if (!tensor_content.empty() && tensor_proto.has_tensor_shape()) {
        switch (ov_type) {
        case u8:
            extract_tensor_content<uint8_t>(tensor_content, &res);
            break;
        case i8:
            extract_tensor_content<int8_t>(tensor_content, &res);
            break;
        case i16:
            extract_tensor_content<int16_t>(tensor_content, &res);
            break;
        case i32:
            extract_tensor_content<int32_t>(tensor_content, &res);
            break;
        case i64:
            extract_tensor_content<int64_t>(tensor_content, &res);
            break;
        case f16:
            extract_tensor_content<float16>(tensor_content, &res);
            break;
        case f32:
            extract_tensor_content<float>(tensor_content, &res);
            break;
        case f64:
            extract_tensor_content<double>(tensor_content, &res);
            break;
        case bf16:
            extract_tensor_content<bfloat16>(tensor_content, &res);
            break;
        default:
            FRONT_END_THROW("Encountered unknown element type " + ov_type.get_type_name());
        }
    } else {
        int64_t val_size = 0;
        switch (ov_type) {
        case boolean:
            val_size = tensor_proto.bool_val_size();
            extract_compressed_tensor_content<bool>(tensor_proto, val_size, &res);
            break;
        case i32:
            val_size = tensor_proto.int_val_size();
            extract_compressed_tensor_content<int32_t>(tensor_proto, val_size, &res);
            break;
        case i64:
            val_size = tensor_proto.int64_val_size();
            extract_compressed_tensor_content<int64_t>(tensor_proto, val_size, &res);
            break;
        case f16:
            val_size = tensor_proto.half_val_size();
            extract_compressed_tensor_content<float16>(tensor_proto, val_size, &res);
            break;
        case f32:
            val_size = tensor_proto.float_val_size();
            extract_compressed_tensor_content<float>(tensor_proto, val_size, &res);
            break;
        case f64:
            val_size = tensor_proto.double_val_size();
            extract_compressed_tensor_content<double>(tensor_proto, val_size, &res);
            break;
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

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
