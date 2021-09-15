/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <fstream>
#include <ngraph/pass/manager.hpp>
#include <numeric>

#include "ngraph/op/util/logical_reduction.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/slice_plan.hpp"
//#include <ngraph/pass/transpose_sinking.h>
#include <frontend_manager/frontend_exceptions.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <tensorflow_frontend/exceptions.hpp>
#include <tensorflow_frontend/place.hpp>
#include <tensorflow_frontend/utility.hpp>

#include "default_opset.h"
#include "graph.hpp"
#include "ngraph_builder.h"
#include "ngraph_conversions.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

static bool VecStrCmp(const std::vector<string>& a, const std::vector<string>& b) {
    return a == b;
}

static Status ValidateInputCount(const NodeContext& op, size_t count) {
    if (op.get_ng_input_size() != count) {
        std::ostringstream buf;
        buf << "\"" << op.get_names()[0] << "\" requires " << count << " input(s), got " << op.get_ng_input_size()
            << " instead";
        return errors::InvalidArgument(buf.str());
    }
    return Status::OK();
}

static void ValidateInputCountMin(const NodeContext& node, size_t count) {
    if (node.get_ng_input_size() < count) {
        std::ostringstream buf;
        buf << "\"" << node.get_name() << "\" requires at least " << count << " input(s), got "
            << node.get_ng_input_size() << " instead";
        throw errors::InvalidArgument(buf.str());
    }
}

// Check to make sure the axis dimension for reduction are in within range.
// Returns error if axis is out of range. Otherwise returns Status::OK().
static Status CheckAxisDimInRange(std::vector<int64_t> axes, size_t rank) {
    for (auto i : axes) {
        if (i < -(int)rank || i >= (int)rank) {
            std::ostringstream buf;
            buf << "Axis Dimension is out of range. Got " << i << ", should be in range [-" << rank << ", " << rank
                << ")";
            return errors::InvalidArgument(buf.str());
        }
    }
    return Status::OK();
}

//
// Helper for storing ops in ng_op_map.
// For most of the cases, op would have one output so
// vector ng_op_map[op_name] would contain one element.
//
// If storing more than one output_nodes, make sure it's in
// the same order as tensorflow would do that.
//
// Parameters:
//    Builder::OpMap& ng_op_map        - The TF-to-nGraph op map.
//    std::string op_name              - Name of the op.
//
//    ng::Output<ng::Node> output_node - ng::Node to store
//

static void SaveNgOp(Builder::OpMap& ng_op_map, const std::string& op_name, ng::Output<ng::Node> output_node) {
    // no need to try-catch, map[key] will create vector object
    // if not exists
    ng_op_map[op_name].push_back(output_node);
}

void Builder::SetTracingInfo(const std::string& op_name, const ng::Output<ng::Node> ng_node) {
    auto node = ng_node.get_node_shared_ptr();
    node->set_friendly_name(op_name);
    node->add_provenance_tag(op_name);
}

template <class TOpType, class... TArg>
ng::Output<ng::Node> ConstructNgNode(const std::string& op_name, TArg&&... Args) {
    auto ng_node = std::make_shared<TOpType>(std::forward<TArg>(Args)...);
    Builder::SetTracingInfo(op_name, ng_node);
    return ng_node;
}

// Helper for fetching correct input node from ng_op_map.
// Handles edge checking to make sure correct input node is
// fetched.
//
// Reduces some boilerplate code (incorrect from now) like this:
//
//      TFNodeDecoder* tf_input;
//      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
//
//      ng::Output<ng::Node> ng_input;
//      try {
//        ng_input = ng_op_map.at(tf_input->name());
//      } catch (const std::out_of_range&) {
//        return errors::NotFound(tf_input->name(),
//                                    " is not found in the ng_op_map");
//      }
//
// Into 2 lines:
//
//      ng::Output<ng::node> ng_input;
//      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input))
//
//
//
// Parameters:
//    Builder::OpMap& ng_op_map     - The TF-to-nGraph op map.
//    TFNodeDecoder* op                  - TF op being translated.
//    input_idx                     - index of input
//
//    ng::Output<ng::Node> *result  - ng::Node pointer where result
//                                    will be written
//
//

static Status GetInputNode(const NodeContext op, size_t input_idx, ng::Output<ng::Node>& result) {
// Stub
#if 0
  // input op may have resulted in more than one ng::Node (eg. Split)
  // we need to look at Edge to check index of the input op
  std::vector<const Edge*> edges;
  TF_RETURN_IF_ERROR(op->input_edges(&edges));
  size_t src_output_idx;
  try {
    src_output_idx = edges.at(input_idx)->src_output();
  } catch (const out_of_range&) {
    return Status(error::NOT_FOUND, "Edge not found");
  }

#endif

    result = op.get_ng_input(input_idx);
    return Status::OK();
}

namespace detail {
static Status GetInputNodes(const NodeContext&, size_t) {
    return Status::OK();
}

template <typename... Arguments>
static Status GetInputNodes(const NodeContext node,
                            size_t index,
                            ng::Output<ng::Node>& result,
                            Arguments&... remaining) {
    TF_RETURN_IF_ERROR(GetInputNode(node, index, result));
    return GetInputNodes(node, index + 1, remaining...);
}
}  // namespace detail

template <typename... Arguments>
static Status GetInputNodes(const NodeContext& node, Arguments&... remaining) {
    constexpr size_t args_len = sizeof...(Arguments);
    TF_RETURN_IF_ERROR(ValidateInputCount(node, args_len));
    return detail::GetInputNodes(node, 0, remaining...);
}

static Status GetStaticNodeTensor(
    const TFNodeDecoder* node,
    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map,
    ngraph::frontend::tensorflow::detail::TensorWrapper* result) {
    if (node->IsArg()) {
        int arg_index;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &arg_index));
        const ngraph::frontend::tensorflow::detail::TensorWrapper* source_tensor = static_input_map[arg_index];
        if (source_tensor == nullptr) {
            return errors::Internal("GetStaticNodeTensor called on _Arg but input tensor is missing from "
                                    "static input map");
        }
        *result = *source_tensor;
        return Status::OK();
    } else if (node->type_string() == "Const") {
        if (GetNodeAttr(node->attrs(), "value", &result).status != 0) {
            return errors::Internal("GetStaticNodeTensor: Const tensor proto parsing failed");
        }
        return Status::OK();
    } else {
        return errors::Internal("GetStaticNodeTensor called on node with type " + node->type_string() +
                                "; _Arg or Const expected");
    }
}

template <typename Ttensor, typename Tvector>
static void ConvertTensorDataToVector(const ngraph::frontend::tensorflow::detail::TensorWrapper& tensor,
                                      std::vector<Tvector>* vector) {
    const Ttensor* data = tensor.flat<Ttensor>().data();
    vector->resize(tensor.NumElements());
    for (int64_t i = 0; i < tensor.NumElements(); i++) {
        (*vector)[i] = Tvector(data[i]);
    }
}

template <typename T>
static Status TensorDataToVector(const ngraph::frontend::tensorflow::detail::TensorWrapper& tensor,
                                 std::vector<T>* vector) {
    // stub
#if 0
  DataType dt = tensor.dtype();

  // If dt and T match, we can just copy.
  if (dt == DataTypeToEnum<T>::value) {
    *vector = std::vector<T>(tensor.flat<T>().data(),
                             tensor.flat<T>().data() + tensor.NumElements());
  }
  // Else we have to convert.
  else {
    switch (dt) {
      case DT_FLOAT:
        ConvertTensorDataToVector<float, T>(tensor, vector);
        break;
      case DT_DOUBLE:
        ConvertTensorDataToVector<double, T>(tensor, vector);
        break;
      case DT_INT8:
        ConvertTensorDataToVector<int8_t, T>(tensor, vector);
        break;
      case DT_INT16:
        ConvertTensorDataToVector<int16_t, T>(tensor, vector);
        break;
      case DT_INT32:
        ConvertTensorDataToVector<int32_t, T>(tensor, vector);
        break;
      case DT_INT64:
        ConvertTensorDataToVector<int64_t, T>(tensor, vector);
        break;
      case DT_UINT8:
        ConvertTensorDataToVector<uint8_t, T>(tensor, vector);
        break;
      case DT_UINT16:
        ConvertTensorDataToVector<uint16_t, T>(tensor, vector);
        break;
      case DT_UINT32:
        ConvertTensorDataToVector<uint32, T>(tensor, vector);
        break;
      case DT_UINT64:
        ConvertTensorDataToVector<uint64, T>(tensor, vector);
        break;
      case DT_BOOL:
        ConvertTensorDataToVector<bool, T>(tensor, vector);
        break;
      default:
        return errors::Internal("TensorDataToVector: tensor has element type ",
                                DataType_Name(dt), ", vector has type ",
                                DataType_Name(DataTypeToEnum<T>::value),
                                "; don't know how to convert");
    }
  }
  return Status::OK();
#endif

    NGRAPH_TF_FE_NOT_IMPLEMENTED
}

template <typename T>
static void GetStaticInputVector(const NodeContext& node, int64_t input_index, std::vector<T>* vector) {
    ng::Output<ng::Node> ng_input = node.get_ng_input(input_index);
    if (auto constant = std::dynamic_pointer_cast<ngraph::opset5::Constant>(ng_input.get_node_shared_ptr())) {
        *vector = constant->cast_vector<T>();
        return;
    }

    NGRAPH_TF_FE_NOT_IMPLEMENTED;
    /*
        TFNodeDecoder* input_node;
        TF_RETURN_IF_ERROR(op->input_node(input_index, &input_node));
        ngraph::frontend::tensorflow::detail::TensorWrapper* input_tensor;
        TF_RETURN_IF_ERROR(
                GetStaticNodeTensor(input_node, static_input_map, &input_tensor));
        TF_RETURN_IF_ERROR(TensorDataToVector(input_tensor, vector));
        return Status::OK();*/
}

#if 0
template <typename T>
static Status GetStaticInputVector(
    const TFNodeDecoder* op, int64_t input_index,
    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map,
    std::vector<T>* vector) {
  TFNodeDecoder* input_node;
  TF_RETURN_IF_ERROR(op->input_node(input_index, &input_node));
  ngraph::frontend::tensorflow::detail::TensorWrapper* input_tensor;
  TF_RETURN_IF_ERROR(
      GetStaticNodeTensor(input_node, static_input_map, &input_tensor));
  TF_RETURN_IF_ERROR(TensorDataToVector(input_tensor, vector));
  return Status::OK();
}

static Status GetStaticInputNode(
    const TFNodeDecoder* op, int64_t input_index,
    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map, DataType dt,
    ng::Output<ng::Node>& node_) {
  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dt, &type));
  switch (dt) {
    case DataType::DT_FLOAT: {
      std::vector<float> vec_float;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_float));
      node_ = ConstructNgNode<opset::Constant>(node.get_name(), type, ng::Shape{},
                                               vec_float[0]);
    } break;
    case DataType::DT_DOUBLE: {
      std::vector<double> vec_double;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_double));
      node_ = ConstructNgNode<opset::Constant>(node.get_name(), type, ng::Shape{},
                                               vec_double[0]);
    } break;
    case DataType::DT_INT32: {
      std::vector<int32_t> vec_i32;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_i32));
      node_ = ConstructNgNode<opset::Constant>(node.get_name(), type, ng::Shape{},
                                               vec_i32[0]);
    } break;
    case DataType::DT_INT64: {
      std::vector<int64_t> vec_i64;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_i64));
      node_ = ConstructNgNode<opset::Constant>(node.get_name(), type, ng::Shape{},
                                               vec_i64[0]);
    } break;
    default:
      return errors::Internal("GetStaticInputNode: TF data type " +
                              DataType_Name(dt) + " not supported.");
      break;
  }
  return Status::OK();
}
#endif

// Taken from: tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc
// Extract values from a Const op to `values`. Returns true if succeeds.
//
// Modified with an extra `VecT` parameter to handle the case where the type
// in the vector does not match TensorFlow's notion of what the C++ type
// should be (e.g. when T is `bool`, we actually need a vector of `char` for
// compatibility with nGraph).
template <typename T, typename VecT = T>
static Status ValuesFromConstNode(const TFNodeDecoder* node,
                                  ngraph::Shape* const_tensor_shape,
                                  std::vector<VecT>* values) {
#if 1

    if (node->op() != "Const") {
        return errors::InvalidArgument("TFNodeDecoder not a Const");
    }
    DataType dt;
    node->getAttrValue2("dtype", &dt);

    /*
    if (dt != DataTypeToEnum<T>::value) {
      std::stringstream ss;
      ss << "Invalid data type defined for Const. Defined: "
         << node.attr().at("dtype").type();
      return errors::InvalidArgument(ss.str());
    }
     */

    // ngraph::frontend::tensorflow::detail::TensorWrapper represents the content of the tensor in either <type>_val or
    // tensor_content.
    ngraph::frontend::tensorflow::detail::TensorWrapper* tensor;
    node->getAttrValue2("value", &tensor);
    // typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
    //    checkpoint::MutableTensorProtoData<T>(const_cast<ngraph::frontend::tensorflow::detail::TensorWrapper*>(&tensor));

    const TensorShapeProto& shape = tensor->tensor_def->tensor_shape();
    ngraph::PartialShape pshape;
    TFTensorShapeToNGraphShape(shape, &pshape);
    *const_tensor_shape = pshape.get_shape();
    if (pshape.is_dynamic())
        NGRAPH_TF_FE_NOT_IMPLEMENTED;
    auto tensor_content = tensor->tensor_def->tensor_content();
    std::vector<char> tensor_values_plain(tensor_content.begin(), tensor_content.end());
    const T* tensor_values = reinterpret_cast<const T*>(tensor_values_plain.data());

    if (!tensor_values_plain.empty() && tensor->tensor_def->has_tensor_shape()) {
        // When tensor_shape is set, theoretically the representation of the data
        // could be compressed. So, before copying values to the returned vector,
        // make sure no compression happens.
        // if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values_plain.size()/sizeof(T)) {
        values->insert(values->end(), tensor_values, tensor_values + tensor_values_plain.size() / sizeof(T));
        return Status::OK();
        //}
    }

    const auto tensor_content_size = tensor->tensor_def->tensor_content().size();
    if (tensor_content_size % sizeof(VecT)) {
        std::cerr << "[ ERROR ] tensor_content_size (" << tensor_content_size << ") is not a multiple of "
                  << sizeof(VecT);
    }

    // If tensor_content_size is zero, we'll have to take the values from
    // int_val, float_val, etc.
    if (tensor_content_size == 0) {
        int64_t n_elements = 1;
        for (auto i = 0; i < shape.dim_size(); i++) {
            if (shape.dim(i).size() < 0) {
                return errors::InvalidArgument("Const node has empty tensor and an unknown dimension size");
            }
            n_elements *= shape.dim(i).size();
        }
        values->resize(n_elements);

        auto val_lastsaved = (T)0;  // cast

        for (auto i = 0; i < n_elements; i++) {
            auto& tensor_proto = *tensor->tensor_def;
            int64_t val_size = 0;
            auto val_i = (T)0;  // cast
            switch (dt) {
            // TODO(amprocte/NGRAPH-2502): there are more element types to support
            // here
            case DT_INT32:
                val_size = tensor_proto.int_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.int_val()[i];
                break;
            case DT_INT64:
                val_size = tensor_proto.int64_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.int64_val()[i];
                break;
            case DT_FLOAT:
                val_size = tensor_proto.float_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.float_val()[i];
                break;
            case DT_BOOL:
                val_size = tensor_proto.bool_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.bool_val()[i];
                break;
            case DT_DOUBLE:
                val_size = tensor_proto.double_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.double_val()[i];
                break;
            default:
                NGRAPH_VLOG(0) << "Const node has empty tensor_proto and we don't know how to "
                                  "handle this element type";
                NGRAPH_VLOG(0) << node->DebugString();
                NGRAPH_VLOG(0) << shape.DebugString();
                return errors::Unimplemented("Encountered unknown element type " + DataType_Name(dt) +
                                             " on an empty tensor_proto");
            }
            if (val_size == 0) {
                return errors::InvalidArgument("Empty values vector");
            } else if (i < val_size) {
                (*values)[i] = val_i;
                val_lastsaved = val_i;
            } else {
                (*values)[i] = val_lastsaved;
            }
        }
    } else {
        return Status::OK();
        // values->resize(tensor_content_size / sizeof(VecT));
        // port::CopyToArray(tensor.tensor_content(),
        //                  reinterpret_cast<char*>(values->data()));
    }

    return Status::OK();
#endif
}

// Helper for Builder::TranslateGraph ("Const" op)
template <typename T, typename VecT = T>
static Status MakeConstOp(const NodeContext& node, ng::element::Type et, ng::Output<ng::Node>& ng_node) {
    vector<VecT> const_values;
    ngraph::Shape ng_shape;

    TF_RETURN_IF_ERROR((ValuesFromConstNode<T, VecT>(node._get_decoder(), &ng_shape, &const_values)));

    ng_node = ConstructNgNode<opset::Constant>(node.get_name(), et, ng_shape, const_values);
    return Status::OK();
}

const Builder::ConstMap& Builder::TF_NGRAPH_CONST_MAP() {
    static const Builder::ConstMap the_map = {
        {ng::element::f32, make_pair(MakeConstOp<float>, ng::element::f32)},
        {ng::element::f64, make_pair(MakeConstOp<double>, ng::element::f64)},
        {ng::element::i8, make_pair(MakeConstOp<int8_t>, ng::element::i8)},
        {ng::element::i16, make_pair(MakeConstOp<int16_t>, ng::element::i16)},
#if 0
      {DataType::DT_QINT8, make_pair(MakeConstOp<qint8>, ng::element::i8)},
      {DataType::DT_QUINT8, make_pair(MakeConstOp<quint8>, ng::element::u8)},
      {DataType::DT_QUINT16, make_pair(MakeConstOp<quint16>, ng::element::u16)},
#endif
        {ng::element::i32, make_pair(MakeConstOp<int32_t>, ng::element::i32)},
        {ng::element::i64, make_pair(MakeConstOp<int64_t>, ng::element::i64)},
        {ng::element::u8, make_pair(MakeConstOp<uint8_t>, ng::element::u8)},
        {ng::element::u16, make_pair(MakeConstOp<uint16_t>, ng::element::u16)},
        {ng::element::boolean, make_pair(MakeConstOp<bool, char>, ng::element::boolean)}
    };
    return the_map;
}

// Helper function to translate a unary op.
//
// Parameters:
//
//    TFNodeDecoder* op                   - TF op being translated. Must have one input.
//    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map
//                               - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-nGraph op map.
//
//    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>>
//      create_unary_op           - Function to construct the graph implementing
//                                 the unary op, given the input to the unop
//                                 as an argument.
//
// Example usage:
//
//  if (n->type_string == "Square") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp(n, static_input_map, ng_op_map,
//                       [] (ng::Output<ng::Node> n) {
//                           return
//                           (ng::Output<opset::Multiply>(n,n));
//                       });
//  }
static ngraph::OutputVector TranslateUnaryOp(
    const NodeContext& op,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>)> create_unary_op) {
    ng::Output<ng::Node> ng_input = op.get_ng_input(0);
    auto ng_node = create_unary_op(ng_input);
    if (ng_node != ng_input) {
        Builder::SetTracingInfo(op.get_name(), ng_node);
    }
    // SaveNgOp(ng_op_map, node.get_name(), ng_node);
    // return Status::OK();
    return {ng_node};
}

// Helper function to translate a unary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Abs") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp<ng::op::Abs>(n, static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static OutputVector TranslateUnaryOp(const NodeContext& node) {
    return TranslateUnaryOp(node, [&node](ng::Output<ng::Node> n) {
        return ConstructNgNode<T>(node.get_name(), n);
    });
}

// Helper function to translate a binary op
// Parameters:
//
//    TFNodeDecoder* op               - TF op being translated. Must have only two
//    inputs.
//    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map - the static input
//    map Builder::OpMap& ng_op_map  - The TF-to-nGraph op map. std::function<ng::Output<ng::Node>(ng::Output<ng::Node>,
//    ng::Output<ng::Node>)>
//    create_binary_op           - Function to construct the graph implementing
//                                 the binary op, given the 2 ng_inputs to the
//                                 binaryop
// Example Usage:
//
// if (op->type_string() == "SquaredDifference") {
//      TF_RETURN_IF_ERROR(TranslateBinaryOp(op, ng_op_map,
//         [](ng::Output<ng::Node> ng_input1, ng::Output<ng::Node>
//         ng_input2) {
//           auto ng_diff = ng::Output<opset::Subtract>(input1,
//           input2);
//           return ng::Output<opset::Multiply>(ng_diff,ng_diff);
//         }));
//    }
//

static OutputVector TranslateBinaryOp(
    const NodeContext& node,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>&, ng::Output<ng::Node>&)> create_binary_op) {
    ng::Output<ng::Node> ng_lhs = node.get_ng_input(0), ng_rhs = node.get_ng_input(1);
    auto ng_node = create_binary_op(ng_lhs, ng_rhs);
    if (ng_node != ng_lhs && ng_node != ng_rhs) {
        Builder::SetTracingInfo(node.get_name(), ng_node);
    }
    return {ng_node};
}

// Helper function to translate a binary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Add") {
//    TF_RETURN_IF_ERROR(TranslateBinaryOp<opset::Add>(op,
//    static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static OutputVector TranslateBinaryOp(const NodeContext& node) {
    return TranslateBinaryOp(node, [&node](ng::Output<ng::Node>& ng_lhs, ng::Output<ng::Node>& ng_rhs) {
        return ConstructNgNode<T>(node.get_name(), ng_lhs, ng_rhs);
    });
}

static OutputVector TranslateAddNOp(const NodeContext& node) {
    OutputVector ng_arg_vec = node.get_ng_inputs();

    auto ng_addn = std::accumulate(std::next(ng_arg_vec.begin()),
                                   ng_arg_vec.end(),
                                   ng_arg_vec.at(0),
                                   [&node](ng::Output<ng::Node> a, ng::Output<ng::Node> b) {
                                       return ConstructNgNode<opset::Add>(node.get_name(), a, b);
                                   });  // accumulation: start with
                                        // first element. default op is
                                        // addition
    return {ng_addn};
}
static OutputVector TranslateArgMinMax(const NodeContext& node, std::string mode) {
    ng::Output<ng::Node> ng_input = node.get_ng_input(0);

    std::vector<int64_t> tf_dim;
    GetStaticInputVector(node, 1, &tf_dim);

    ng::Shape input_shape = ng_input.get_shape();
    size_t input_rank = input_shape.size();

    if (tf_dim.size() != 1) {
        throw errors::InvalidArgument("ArgMax Op: dimension must be scalar, operates on a single axis");
    }

    // If input dimension is negative, make it positive
    if (tf_dim[0] < 0) {
        NGRAPH_VLOG(3) << "Input dimension is negative, make it positive " << tf_dim[0];
        tf_dim[0] = (int64_t)input_rank + tf_dim[0];
    }
    NGRAPH_VLOG(3) << "Axis along which to compute " << tf_dim[0];
    size_t k_axis = tf_dim[0];

    auto ng_et = node.get_attribute<ng::element::Type>("output_type");

    auto ng_k =
        ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64, ng::Shape{}, std::vector<int64_t>({1}));

    std::string sort = "none";
    auto ng_topk = std::make_shared<opset::TopK>(ng_input, ng_k, k_axis, mode, sort, ng_et);
    auto ng_indices = ng_topk->output(1);
    int axis = ng_topk->get_axis();
    auto axis_to_remove =
        ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64, ng::Shape{1}, std::vector<int64_t>({axis}));
    auto reshaped_indices = ConstructNgNode<opset::Squeeze>(node.get_name(), ng_indices, axis_to_remove);
    Builder::SetTracingInfo(node.get_name(), reshaped_indices);
    return {reshaped_indices};
}

static OutputVector TranslateArgMaxOp(const NodeContext& node) {
    return (TranslateArgMinMax(node, "max"));
}

static OutputVector TranslateArgMinOp(const NodeContext& node) {
    return (TranslateArgMinMax(node, "min"));
}

static OutputVector TranslateAvgPoolOp(const NodeContext& node) {
    ng::Output<ng::Node> ng_input = node.get_ng_input(0);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_ksize = node.get_attribute<std::vector<int32_t>>("ksize");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");
    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("AvgPool data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    NGRAPH_VLOG(3) << ng::join(tf_strides);
    NGRAPH_VLOG(3) << ng::join(tf_ksize);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    ng::Strides ng_strides(2);
    ng::Shape ng_image_shape(2);
    ng::Shape ng_kernel_shape(2);
    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);
    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff padding_below;
    ng::CoordinateDiff padding_above;
    ng::Shape ng_dilations{1, 1};
    Builder::MakePadding(tf_padding_type,
                         ng_image_shape,
                         ng_kernel_shape,
                         ng_strides,
                         ng_dilations,
                         padding_below,
                         padding_above);

    // TODO: remove this once nGraph supports negative padding
    // (CoordinateDiff) for AvgPool
    ng::Shape ng_padding_below(padding_below.begin(), padding_below.end());
    ng::Shape ng_padding_above(padding_above.begin(), padding_above.end());

    ng::Output<ng::Node> ng_avgpool = ConstructNgNode<opset::AvgPool>(node.get_name(),
                                                                      ng_input,
                                                                      ng_strides,
                                                                      ng_padding_below,
                                                                      ng_padding_above,
                                                                      ng_kernel_shape,
                                                                      true,
                                                                      ng::op::RoundingType::FLOOR);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_avgpool);
    NGRAPH_VLOG(3) << "avgpool outshape: {" << ng::join(ng_avgpool.get_shape()) << "}";

    return {ng_avgpool};
}

static OutputVector TranslateBiasAddOp(const NodeContext& node) {
    ng::Output<ng::Node> ng_input = node.get_ng_input(0), ng_bias = node.get_ng_input(1);

    std::string tf_data_format = node.get_attribute<std::string>("data_format", "NHWC");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("BiasAdd data format is neither NHWC nor NCHW");
    }

    auto ng_input_shape = ng_input.get_shape();
    auto ng_bias_shape = ng_bias.get_shape();
    if (ng_bias_shape.size() != 1) {
        throw errors::InvalidArgument("Bias argument to BiasAdd does not have one dimension");
    }

    // We'll choose reshape over broadcast
    // Reshape the bias to (1, C, 1, ...) if input is channels-first.
    ng::Output<ng::Node> ng_bias_reshaped = ng_bias;
    if (tf_data_format == "NCHW") {
        auto channel_dim = ng_input_shape[1];
        std::vector<int64_t> target_shape(ng_input_shape.size());
        for (int64_t i = 0; i < ng_input_shape.size(); i++) {
            if (i == 1) {
                target_shape[i] = channel_dim;
            } else {
                target_shape[i] = 1;
            }
        }
        auto target_shape_node =
            make_shared<opset::Constant>(ng::element::i64, ng::Shape{ng_input_shape.size()}, target_shape);
        ng_bias_reshaped = ConstructNgNode<opset::Reshape>(node.get_name(), ng_bias, target_shape_node, false);
    }

    ng::Output<ng::Node> ng_add = ConstructNgNode<opset::Add>(node.get_name(), ng_input, ng_bias_reshaped);

    return {ng_add};
}

static OutputVector TranslateCastOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);

    auto ng_et = node.get_attribute<ng::element::Type>("DstT");
    return {ConstructNgNode<opset::Convert>(node.get_name(), ng_input, ng_et)};
}

static OutputVector TranslateConcatV2Op(const NodeContext& node) {
    ValidateInputCountMin(node, 2);

    std::vector<int64_t> tf_concat_axis_vec;
    GetStaticInputVector(node, node.get_ng_input_size() - 1, &tf_concat_axis_vec);

    int64_t concat_axis = tf_concat_axis_vec[0];

    if (concat_axis < 0) {
        auto ng_first_arg = node.get_ng_input(0);
        concat_axis += int64_t(ng_first_arg.get_shape().size());
    }

    ng::OutputVector ng_args;

    for (int i = 0; i < node.get_ng_input_size() - 1; i++) {
        ng::Output<ng::Node> ng_arg = node.get_ng_input(i);
        ng_args.push_back(ng_arg);
    }

    return {ConstructNgNode<opset::Concat>(node.get_name(), ng_args, size_t(concat_axis))};
}

static OutputVector TranslateConstOp(const NodeContext& node) {
    auto dt = node.get_attribute<ngraph::element::Type>("dtype");
    ng::Output<ng::Node> ng_node;

    // For some reason the following do not work (no specialization of
    // tensorflow::checkpoint::SavedTypeTraits...)
    // case DataType::DT_UINT32:
    //   TF_RETURN_IF_ERROR(MakeConstOp<uint32>(op, ng::element::u32,
    //   &ng_node));
    //   break;
    // case DataType::DT_UINT64:
    //   TF_RETURN_IF_ERROR(MakeConstOp<uint64>(op, ng::element::u64,
    //   &ng_node));
    //   break;
    try {
        const auto& func_param = Builder::TF_NGRAPH_CONST_MAP().at(dt);
        TF_RETURN_IF_ERROR(func_param.first(node, func_param.second, ng_node));
    } catch (const std::out_of_range&) {
        throw errors::Unimplemented("Failed to translate Constant with target ngraph type:" + dt.get_type_name());
    }

    return {ng_node};
}

static OutputVector TranslateConv2DOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("Conv2D data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    // TF Kernel Test Checks
    // Strides in the batch and depth dimension is not supported
    if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
        throw errors::InvalidArgument("Strides in batch and depth dimensions is not supported: " + node.get_op_type());
    }

    NGRAPH_VLOG(3) << ng::join(tf_strides);
    NGRAPH_VLOG(3) << ng::join(tf_dilations);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    ng::Strides ng_strides(2);
    ng::Strides ng_dilations(2);
    ng::Shape ng_image_shape(2);
    ng::Shape ng_kernel_shape(2);

    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    Transpose<3, 2, 0, 1>(ng_filter);
    Builder::SetTracingInfo(node.get_name(), ng_filter);

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff ng_padding_below;
    ng::CoordinateDiff ng_padding_above;
    Builder::MakePadding(tf_padding_type,
                         ng_image_shape,
                         ng_kernel_shape,
                         ng_strides,
                         ng_dilations,
                         ng_padding_below,
                         ng_padding_above);

    ng::Output<ng::Node> ng_conv = ConstructNgNode<opset::Convolution>(node.get_name(),
                                                                       ng_input,
                                                                       ng_filter,
                                                                       ng_strides,
                                                                       ng_padding_below,
                                                                       ng_padding_above,
                                                                       ng_dilations);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_conv);
    return {ng_conv};
}

static OutputVector TranslateConv2DBackpropInputOp(const NodeContext& node) {
    auto ng_filter = node.get_ng_input(1), ng_out_backprop = node.get_ng_input(2);

    // TODO: refactor me to be less redundant with other convolution ops
    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("Conv2DBackpropInput data format is neither NHWC nor NCHW: %s" + tf_data_format);
    }

    std::vector<int64_t> tf_input_sizes;
    GetStaticInputVector(node, 0, &tf_input_sizes);

    if (std::any_of(tf_input_sizes.begin(), tf_input_sizes.end(), [](int32_t size) {
            return size <= 0;
        })) {
        throw errors::InvalidArgument("Conv2DBackpropInput input sizes must be positive integers");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    NGRAPH_VLOG(3) << ng::join(tf_strides);
    NGRAPH_VLOG(3) << ng::join(tf_dilations);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    ng::Strides ng_strides(2);
    ng::Strides ng_dilations(2);
    ng::Shape ng_image_shape(2);
    ng::Shape ng_kernel_shape(2);
    ng::Shape ng_batch_shape(4);

    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoHW(is_nhwc, tf_input_sizes, ng_image_shape);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_out_backprop);
    if (is_nhwc) {
        ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                          static_cast<unsigned long>(tf_input_sizes[3]),
                          static_cast<unsigned long>(tf_input_sizes[1]),
                          static_cast<unsigned long>(tf_input_sizes[2])};
    } else {
        ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                          static_cast<unsigned long>(tf_input_sizes[1]),
                          static_cast<unsigned long>(tf_input_sizes[2]),
                          static_cast<unsigned long>(tf_input_sizes[3])};
    }

    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    Transpose<3, 2, 0, 1>(ng_filter);
    Builder::SetTracingInfo(node.get_name(), ng_filter);

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff ng_padding_below;
    ng::CoordinateDiff ng_padding_above;
    Builder::MakePadding(tf_padding_type,
                         ng_image_shape,
                         ng_kernel_shape,
                         ng_strides,
                         ng_dilations,
                         ng_padding_below,
                         ng_padding_above);

    auto ng_output_shape =
        ConstructNgNode<opset::Constant>(node.get_name(),
                                         ng::element::i64,
                                         ng::Shape{ng_batch_shape.size() - 2},
                                         vector<size_t>(ng_batch_shape.begin() + 2, ng_batch_shape.end()));

    auto ng_data = ConstructNgNode<opset::ConvolutionBackpropData>(node.get_name(),
                                                                   ng_out_backprop,
                                                                   ng_filter,
                                                                   ng_output_shape,
                                                                   ng_strides,
                                                                   ng_padding_below,
                                                                   ng_padding_above,
                                                                   ng_dilations);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_data);
    return {ng_data};
}

// Translate Conv3D Op
static OutputVector TranslateConv3DOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NDHWC" && tf_data_format != "NCDHW") {
        throw errors::InvalidArgument("Conv3D data format is neither NDHWC nor NCDHW");
    }

    bool is_ndhwc = (tf_data_format == "NDHWC");

    // TODO: in 3D
    // TF Kernel Test Checks
    // // Strides in the batch and depth dimension is not supported
    // if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
    //   return errors::InvalidArgument(
    //       "Strides in batch and depth dimensions is not supported: ",
    //       op->type_string());
    // }

    NGRAPH_VLOG(3) << ng::join(tf_strides);
    NGRAPH_VLOG(3) << ng::join(tf_dilations);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    ng::Strides ng_strides(3);
    ng::Strides ng_dilations(3);
    ng::Shape ng_image_shape(3);
    ng::Shape ng_kernel_shape(3);

    NHWCtoHW(is_ndhwc, tf_strides, ng_strides);
    NHWCtoHW(is_ndhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_ndhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(node.get_name(), is_ndhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    ng_kernel_shape[2] = ng_filter_shape[2];
    Transpose3D<4, 3, 0, 1, 2>(ng_filter);
    Builder::SetTracingInfo(node.get_name(), ng_filter);

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff ng_padding_below;
    ng::CoordinateDiff ng_padding_above;
    Builder::MakePadding(tf_padding_type,
                         ng_image_shape,
                         ng_kernel_shape,
                         ng_strides,
                         ng_dilations,
                         ng_padding_below,
                         ng_padding_above);

    ng::Output<ng::Node> ng_conv = ConstructNgNode<opset::Convolution>(node.get_name(),
                                                                       ng_input,
                                                                       ng_filter,
                                                                       ng_strides,
                                                                       ng_padding_below,
                                                                       ng_padding_above,
                                                                       ng_dilations);

    NCHWtoNHWC(node.get_name(), is_ndhwc, ng_conv);
    return {ng_conv};
}

static OutputVector TranslateCumsumOp(const NodeContext& node) {
    auto ng_x = node.get_ng_input(0), ng_axis = node.get_ng_input(1);
    auto exclusive = node.get_attribute<bool>("exclusive"), reverse = node.get_attribute<bool>("reverse");

    return {ConstructNgNode<opset::CumSum>(node.get_name(), ng_x, ng_axis, exclusive, reverse)};
}

// Translate DepthToSpace op
static OutputVector TranslateDepthToSpaceOp(const NodeContext& node) {
    ng::Output<ng::Node> ng_input = node.get_ng_input(0);

    // Get the attributes
    auto block_size = node.get_attribute<int64_t>("block_size");
    std::string tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("DepthToSpace data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);
    auto ng_mode = opset::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
    ng::Output<ng::Node> depth_to_space =
        ConstructNgNode<opset::DepthToSpace>(node.get_name(), ng_input, ng_mode, block_size);
    NCHWtoNHWC(node.get_name(), is_nhwc, depth_to_space);
    return {depth_to_space};
}

static OutputVector TranslateDepthwiseConv2dNativeOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("DepthwiseConv2D data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    NGRAPH_VLOG(3) << ng::join(tf_strides);
    NGRAPH_VLOG(3) << ng::join(tf_dilations);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    ng::Strides ng_strides(2);
    ng::Strides ng_dilations(2);
    ng::Shape ng_image_shape(2);
    ng::Shape ng_kernel_shape(2);

    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff ng_padding_below;
    ng::CoordinateDiff ng_padding_above;
    Builder::MakePadding(tf_padding_type,
                         ng_image_shape,
                         ng_kernel_shape,
                         ng_strides,
                         ng_dilations,
                         ng_padding_below,
                         ng_padding_above);

    // H W I M -> H W I 1 M
    auto filter_shape = ConstructNgNode<opset::Constant>(
        node.get_name(),
        ng::element::u64,
        ng::Shape{5},
        ngraph::Shape{ng_filter_shape[0], ng_filter_shape[1], ng_filter_shape[2], 1, ng_filter_shape[3]});
    auto reshaped_filter = ConstructNgNode<opset::Reshape>(node.get_name(), ng_filter, filter_shape, false);

    // H W I 1 M -> I M 1 H W
    auto order = ConstructNgNode<opset::Constant>(node.get_name(),
                                                  ng::element::i64,
                                                  ng::Shape{5},
                                                  vector<int64_t>{2, 4, 3, 0, 1});
    auto transposed_filter = ConstructNgNode<opset::Transpose>(node.get_name(), reshaped_filter, order);

    auto ng_conv = ConstructNgNode<opset::GroupConvolution>(node.get_name(),
                                                            ng_input,
                                                            transposed_filter,
                                                            ng_strides,
                                                            ng_padding_below,
                                                            ng_padding_above,
                                                            ng_dilations);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_conv);
    return {ng_conv};
}

static OutputVector TranslateExpandDimsOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    std::vector<int64_t> dims;
    GetStaticInputVector(node, 1, &dims);
    auto ng_dims =
        ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64, ngraph::Shape{dims.size()}, dims);
    return {ConstructNgNode<opset::Unsqueeze>(node.get_name(), ng_input, ng_dims)};
}

static OutputVector TranslateFillOp(const NodeContext& node) {
    auto ng_dims = node.get_ng_input(0), ng_value = node.get_ng_input(1);
    return {ConstructNgNode<opset::Broadcast>(node.get_name(), ng_value, ng_dims)};
}

static OutputVector TranslateFloorDivOp(const NodeContext& node) {
    auto floordiv_fn = [&node](ng::Output<ng::Node> x, ng::Output<ng::Node> y) {
        return ConstructNgNode<opset::Floor>(node.get_name(), ConstructNgNode<opset::Divide>(node.get_name(), x, y));
    };
    return TranslateBinaryOp(node, floordiv_fn);
}

static OutputVector TranslateFusedBatchNormOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_scale = node.get_ng_input(1), ng_offset = node.get_ng_input(2),
         ng_mean = node.get_ng_input(3), ng_variance = node.get_ng_input(4);
    bool is_v3 = node.get_op_type() == "FusedBatchNormV3";

    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        throw errors::InvalidArgument("Conv2D data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    NGRAPH_VLOG(3) << "data_format: " << tf_data_format;

    auto tf_epsilon = node.get_attribute<float>("epsilon", 0.0001);  // TODO: where does 0.0001 come from?

    NGRAPH_VLOG(3) << "epsilon: " << tf_epsilon;

    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    auto ng_batch_norm = ConstructNgNode<opset::BatchNormInference>(node.get_name(),
                                                                    ng_input,
                                                                    ng_scale,
                                                                    ng_offset,
                                                                    ng_mean,
                                                                    ng_variance,
                                                                    tf_epsilon);
    NCHWtoNHWC(node.get_name(), is_nhwc, ng_batch_norm);

    // TODO: Why are there so many? Is it correct?
    OutputVector result = {ng_batch_norm, ng_mean, ng_variance, ng_mean, ng_variance};
    if (is_v3) {
        // FusedBatchNormV3 has 6 outputs
        result.push_back(ng_mean);  // reserve_space_3
    }
    return result;
}

static OutputVector TranslateFusedMatMulOp(const NodeContext& node) {
    // auto num_args = node.get_attribute<int>("num_args"); // TODO: it is unused but why?
    auto fused_ops = node.get_attribute<std::vector<string>>("fused_ops");

    // Transpose arguments if requested.
    auto transpose_a = node.get_attribute<bool>("transpose_a", false);
    auto transpose_b = node.get_attribute<bool>("transpose_b", false);

    auto ng_lhs = node.get_ng_input(0), ng_rhs = node.get_ng_input(1), ng_bias = node.get_ng_input(2);

    ng::Output<ng::Node> ng_matmul =
        ConstructNgNode<opset::MatMul>(node.get_name(), ng_lhs, ng_rhs, transpose_a, transpose_b);

    auto ng_matmul_shape = ng_matmul.get_shape();
    auto ng_bias_shape = ng_bias.get_shape();

    if (ng_bias_shape.size() != 1) {
        throw errors::InvalidArgument("Bias argument to BiasAdd does not have one dimension");
    }

    auto ng_add = ConstructNgNode<opset::Add>(node.get_name(), ng_matmul, ng_bias);
    if (fused_ops.size() == 1) {  // Only fusing BiasAdd
        return {ng_add};
    } else if (fused_ops.size() == 2) {  // Also has activation
        if (fused_ops[1] == "Relu") {
            return {ConstructNgNode<opset::Relu>(node.get_name(), ng_add)};
        } else if (fused_ops[1] == "Relu6") {
            return {ConstructNgNode<opset::Clamp>(node.get_name(), ng_add, 0, 6)};
        } else {
            throw errors::Internal("Expected activation to be Relu or Relu6 but got " + fused_ops[1]);
        }
    } else {
        // Adding this here to catch future changes in _FusedMatMul
        throw errors::Internal("Unsupported combination");
    }
}

// See .../tensorflow/include/tensorflow/cc/ops/array_ops.h
// and .../openvino/ngraph/core/include/ngraph/op/gather.hpp
static OutputVector TranslateGatherOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_input_indices = node.get_ng_input(1);

    auto ng_axis = ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64, ng::Shape{}, 0);

    auto gather_op = ConstructNgNode<opset::Gather>(node.get_name(), ng_input, ng_input_indices, ng_axis);

    return {gather_op};
}

static OutputVector TranslateGatherV2Op(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_input_coords = node.get_ng_input(1);

    std::vector<int64_t> tf_axis;
    GetStaticInputVector(node, 2, &tf_axis);

    if (tf_axis.size() > 1) {
        std::ostringstream buf;
        buf << "Found axis in GatherV2 op (" << node.get_name() << ") translation to be non scalar, of size "
            << tf_axis.size();
        throw errors::Internal(buf.str());
    }

    // Negative axis is supported. Accounting for that
    auto ng_input_shape = ng_input.get_shape();
    size_t ng_input_rank = ng_input_shape.size();
    int axis;
    if (tf_axis[0] >= 0) {
        axis = tf_axis[0];
    } else {
        axis = tf_axis[0] + ng_input_rank;
    }
    if (axis < 0 || axis >= ng_input_rank) {
    std:
        ostringstream buf;
        buf << "Expected axis in the range [-" << ng_input_rank << ", " << ng_input_rank << "), but got " << tf_axis[0];
        throw errors::InvalidArgument(buf.str());
    }

    auto ng_axis =
        ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64, ng::Shape{tf_axis.size()}, tf_axis);

    auto gather_op = ConstructNgNode<opset::Gather>(node.get_name(), ng_input, ng_input_coords, ng_axis);

    return {gather_op};
}

static OutputVector TranslateFusedConv2DOp(const NodeContext& node) {
    auto num_args = node.get_attribute<int>("num_args");
    auto fused_ops = node.get_attribute<std::vector<string>>("fused_ops");

    auto tf_data_format = node.get_attribute<std::string>("data_format");
    bool is_nhwc = (tf_data_format == "NHWC");

    auto CreateNgConv = [&](ng::Output<ng::Node>& ng_input, ng::Output<ng::Node>& ng_filter) {
        auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
        auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
        auto tf_padding_type = node.get_attribute<std::string>("padding");

        if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
            throw errors::InvalidArgument("Conv2D data format is neither NHWC nor NCHW");
        }

        // TF Kernel Test Checks
        // Strides in the batch and depth dimension is not supported
        if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
            throw errors::InvalidArgument("Strides in batch and depth dimensions is not supported: " +
                                          node.get_op_type());
        }

        NGRAPH_VLOG(3) << ng::join(tf_strides);
        NGRAPH_VLOG(3) << ng::join(tf_dilations);
        NGRAPH_VLOG(3) << tf_padding_type;
        NGRAPH_VLOG(3) << tf_data_format;

        ng::Strides ng_strides(2);
        ng::Strides ng_dilations(2);
        ng::Shape ng_image_shape(2);
        ng::Shape ng_kernel_shape(2);

        NHWCtoHW(is_nhwc, tf_strides, ng_strides);
        NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
        NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
        NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

        NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
        NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
        NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

        auto& ng_filter_shape = ng_filter.get_shape();
        ng_kernel_shape[0] = ng_filter_shape[0];
        ng_kernel_shape[1] = ng_filter_shape[1];
        Transpose<3, 2, 0, 1>(ng_filter);
        Builder::SetTracingInfo(node.get_name(), ng_filter);

        NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

        ng::CoordinateDiff ng_padding_below;
        ng::CoordinateDiff ng_padding_above;
        Builder::MakePadding(tf_padding_type,
                             ng_image_shape,
                             ng_kernel_shape,
                             ng_strides,
                             ng_dilations,
                             ng_padding_below,
                             ng_padding_above);

        return ConstructNgNode<opset::Convolution>(node.get_name() + "_FusedConv2D_Conv",
                                                   ng_input,
                                                   ng_filter,
                                                   ng_strides,
                                                   ng_padding_below,
                                                   ng_padding_above,
                                                   ng_dilations);
    };

    if (VecStrCmp(fused_ops, {"BiasAdd"}) || VecStrCmp(fused_ops, {"BiasAdd", "Relu"}) ||
        VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
        if (num_args != 1) {
            throw errors::InvalidArgument("FusedConv2DBiasAdd has incompatible num_args");
        }

        auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1), ng_bias = node.get_ng_input(2),
             ng_conv = CreateNgConv(ng_input, ng_filter);

        auto ng_conv_shape = ng_conv.get_shape();
        auto ng_bias_shape = ng_bias.get_shape();
        if (ng_bias_shape.size() != 1) {
            throw errors::InvalidArgument("Bias argument to BiasAdd does not have one dimension");
        }

        std::vector<size_t> reshape_pattern_values(ng_conv_shape.size(), 1U);
        reshape_pattern_values[1] = ng_bias.get_shape().front();
        auto reshape_pattern = make_shared<opset::Constant>(ng::element::u64,
                                                            ng::Shape{reshape_pattern_values.size()},
                                                            reshape_pattern_values);
        auto ng_bias_reshaped = ConstructNgNode<opset::Reshape>(node.get_name(), ng_bias, reshape_pattern, false);

        auto ng_add = ConstructNgNode<opset::Add>(node.get_name() + "_FusedConv2D_BiasAdd", ng_conv, ng_bias_reshaped);

        if (VecStrCmp(fused_ops, {"BiasAdd", "Relu"})) {
            auto ng_relu = ConstructNgNode<opset::Relu>(node.get_name() + "_FusedConv2D_Relu", ng_add);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu);
            return {ng_relu};
        } else if (VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
            auto ng_relu6 = ConstructNgNode<opset::Clamp>(node.get_name() + "_FusedConv2D_Relu6", ng_add, 0, 6);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu6);
            return {ng_relu6};
        } else {
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_add);
            return {ng_add};
        }
    } else if (VecStrCmp(fused_ops, {"FusedBatchNorm"}) || VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
               VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
        if (num_args != 4) {
            throw errors::InvalidArgument("FusedConv2D with FusedBatchNorm has incompatible num_args");
        }

        auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1), ng_scale = node.get_ng_input(2),
             ng_offset = node.get_ng_input(3), ng_mean = node.get_ng_input(4), ng_variance = node.get_ng_input(5),
             ng_conv = CreateNgConv(ng_input, ng_filter);

        auto tf_epsilon = node.get_attribute<float>("epsilon");

        auto ng_batch_norm = ConstructNgNode<opset::BatchNormInference>(node.get_name() + "_FusedConv2D_BatchNorm",
                                                                        ng_conv,
                                                                        ng_scale,
                                                                        ng_offset,
                                                                        ng_mean,
                                                                        ng_variance,
                                                                        tf_epsilon);

        if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"})) {
            auto ng_relu = ConstructNgNode<opset::Relu>(node.get_name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu);
            return {ng_relu};
        } else if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
            auto ng_relu6 =
                ConstructNgNode<opset::Clamp>(node.get_name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm, 0, 6);
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_relu6);
            return {ng_relu6};
        } else {
            NCHWtoNHWC(node.get_name(), is_nhwc, ng_batch_norm);
            return {ng_batch_norm};
        }
    } else {
        throw errors::Unimplemented("Unsupported _FusedConv2D " + StrJoin(fused_ops, ","));
    }
}

static OutputVector TranslateIdentityOp(const NodeContext& node) {
    return {node.get_ng_input(0)};
}

#if 0

static OutputVector TranslateIsFiniteOp(
    const NodeContext& node) {
  // Implemented tf.is_finite by checking:
  // (in != inf) && (in != -inf) && (in == in)
  //                                 ^^^^^^^^ checks for NaN's
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto const_inf = ConstructNgNode<opset::Constant>(
      node.get_name(), ng_input.get_element_type(), ng::Shape{},
      std::vector<float>{std::numeric_limits<float>::infinity()});

  auto const_neg_inf = ConstructNgNode<opset::Constant>(
      node.get_name(), ng_input.get_element_type(), ng::Shape{},
      std::vector<float>{-std::numeric_limits<float>::infinity()});

  auto neq_inf =
      ConstructNgNode<opset::NotEqual>(node.get_name(), ng_input, const_inf);
  auto neq_neg_inf =
      ConstructNgNode<opset::NotEqual>(node.get_name(), ng_input, const_neg_inf);
  auto eq_nan = ConstructNgNode<opset::Equal>(node.get_name(), ng_input, ng_input);

  auto neq_inf_and_neq_neg_inf =
      ConstructNgNode<opset::LogicalAnd>(node.get_name(), neq_inf, neq_neg_inf);
  auto is_finite = ConstructNgNode<opset::LogicalAnd>(
      node.get_name(), neq_inf_and_neq_neg_inf, eq_nan);

  SaveNgOp(ng_op_map, node.get_name(), is_finite);
  return Status::OK();
}

static Status TranslateL2LossOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<float> val;
  val.push_back(2.0);
  auto const_2 = ConstructNgNode<opset::Constant>(
      node.get_name(), ng_input.get_element_type(), ng::Shape{}, val[0]);

  auto ng_pow =
      ConstructNgNode<opset::Multiply>(node.get_name(), ng_input, ng_input);

  size_t input_rank = ng_input.get_shape().size();
  std::vector<int64_t> axes;
  for (size_t i = 0; i < input_rank; ++i) {
    axes.push_back(i);
  }

  auto ng_reduction_axes = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{axes.size()}, axes);
  auto ng_sum =
      ConstructNgNode<opset::ReduceSum>(node.get_name(), ng_pow, ng_reduction_axes);
  auto ng_l2loss = ConstructNgNode<opset::Divide>(node.get_name(), ng_sum, const_2);
  SaveNgOp(ng_op_map, node.get_name(), ng_l2loss);
  return Status::OK();
}

static OutputVector TranslateLog1pOp(
    const NodeContext& node) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> val_1(ng::shape_size(shape), "1");
        auto ng_const1 =
            ConstructNgNode<opset::Constant>(node.get_name(), et, shape, val_1);
        auto ng_add = ConstructNgNode<opset::Add>(node.get_name(), ng_const1, n);
        return ConstructNgNode<opset::Log>(node.get_name(), ng_add);
      });
}

static Status TranslateLRNOp(const TFNodeDecoder* op,
                             const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp));

  float alpha;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "alpha", &alpha));
  float beta;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "beta", &beta));
  float bias;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "bias", &bias));
  int64_t depth_radius;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "depth_radius", &depth_radius));

  // OV: Each input value is divided by (bias+(alpha/size)*sum(xi^2 for every xi
  // in the local region))^beta
  // TF: sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d +
  // depth_radius + 1] ** 2)
  //     output = input / (bias + alpha * sqr_sum) ** beta
  int64_t size = depth_radius * 2 + 1;
  alpha = alpha * size;
  // nGraph expects the input to be in NCHW format
  NHWCtoNCHW(node.get_name(), true, ng_inp);
  auto ng_output = ConstructNgNode<opset::LRN>(node.get_name(), ng_inp, alpha, beta,
                                               bias, (size_t)size);
  NCHWtoNHWC(node.get_name(), true, ng_output);
  SaveNgOp(ng_op_map, node.get_name(), ng_output);
  return Status::OK();
}

#endif

static OutputVector TranslateLogSoftmaxOp(const NodeContext& node) {
    auto ng_inp = node.get_ng_input(0);
    auto inp_shape = ng_inp.get_shape();
    size_t rank = inp_shape.size();
    int64_t axes = rank - 1;

    return {ConstructNgNode<opset::LogSoftmax>(node.get_name(), ng_inp, axes)};
}

#if 0

static Status TranslateMatMulOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs));

  // Transpose arguments if requested.
  bool transpose_a = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_a", &transpose_a));

  bool transpose_b = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_b", &transpose_b));

  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<opset::MatMul>(node.get_name(), ng_lhs, ng_rhs,
                                          transpose_a, transpose_b));
  return Status::OK();
}

#endif

template <unsigned int N>
static OutputVector TranslateMaxPoolOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_ksize = node.get_attribute<std::vector<int32_t>>("ksize");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    bool is_nhwc = (tf_data_format == "NHWC") || (tf_data_format == "NDHWC");

    NGRAPH_VLOG(3) << ng::join(tf_strides);
    NGRAPH_VLOG(3) << ng::join(tf_ksize);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    ng::Strides ng_strides(N);
    ng::Shape ng_image_shape(N);
    ng::Shape ng_kernel_shape(N);
    ng::Shape ng_dilations(N, 1);

    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);
    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff padding_below;
    ng::CoordinateDiff padding_above;
    Builder::MakePadding(tf_padding_type,
                         ng_image_shape,
                         ng_kernel_shape,
                         ng_strides,
                         ng_dilations,
                         padding_below,
                         padding_above);

    // TODO: remove this once nGraph supports negative padding
    // (CoordinateDiff) for MaxPool
    ng::Shape ng_padding_below(padding_below.begin(), padding_below.end());
    ng::Shape ng_padding_above(padding_above.begin(), padding_above.end());

    auto ng_maxpool = ConstructNgNode<opset::MaxPool>(node.get_name(),
                                                      ng_input,
                                                      ng_strides,
                                                      ng_padding_below,
                                                      ng_padding_above,
                                                      ng_kernel_shape,
                                                      ng::op::RoundingType::FLOOR);

    NCHWtoNHWC(node.get_name(), is_nhwc, ng_maxpool);

    NGRAPH_VLOG(3) << "maxpool outshape: {" << ng::join(ng_maxpool.get_shape()) << "}";

    return {ng_maxpool};
}

#if 0

static OutputVector TranslateNonMaxSuppressionV2Op(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_boxes, ng_scores, ng_unused, ng_iou_threshold;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_boxes, ng_scores,
                                   ng_unused, ng_iou_threshold));

  auto ng_axis_boxes = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{1}, std::vector<int64_t>({0}));
  auto ng_boxes_unsqueezed =
      ConstructNgNode<opset::Unsqueeze>(node.get_name(), ng_boxes, ng_axis_boxes);

  auto ng_axis_scores = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{1}, std::vector<int64_t>({0}));
  auto ng_scores_unsqueezed1 =
      ConstructNgNode<opset::Unsqueeze>(node.get_name(), ng_scores, ng_axis_scores);
  auto ng_scores_unsqueezed2 = ConstructNgNode<opset::Unsqueeze>(
      node.get_name(), ng_scores_unsqueezed1, ng_axis_scores);

  std::vector<int> max_output_size;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 2, static_input_map, &max_output_size));

  // max_output_size must be scalar
  if (max_output_size.size() != 1) {
    return errors::InvalidArgument(
        "NonMaxSuppression Op: max_output_size of nms must be scalar " +
        to_string(max_output_size.size()));
  }

  auto ng_max_output_size = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{}, max_output_size[0]);
  NGRAPH_VLOG(5) << "ng_max_output_size " << max_output_size[0];

  auto ng_nmsv = ConstructNgNode<opset::NonMaxSuppression>(
      node.get_name(), ng_boxes_unsqueezed, ng_scores_unsqueezed2,
      ng_max_output_size, ng_iou_threshold,
      opset::NonMaxSuppression::BoxEncodingType::CORNER, false,
      ngraph::element::Type_t::i32);

  auto begin = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{2}, std::vector<int64_t>({0, 2}));
  auto end = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{2},
      std::vector<int64_t>({max_output_size[0], 3}));
  auto ng_nmsv_slice = ConstructNgNode<opset::StridedSlice>(
      node.get_name(), ng_nmsv, begin, end, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{0, 1});

  Builder::SetTracingInfo(node.get_name(), ng_nmsv_slice);
  SaveNgOp(ng_op_map, node.get_name(), ng_nmsv_slice);
  return Status::OK();
}

#endif

static OutputVector TranslateReduceOp(
    const NodeContext& node,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>, ng::Output<ng::Node>, const bool)> create_ng_node) {
    ng::Output<ng::Node> ng_input = node.get_ng_input(0);
    auto tf_keep_dims = node.get_attribute<bool>("keep_dims", false);

    std::vector<int64_t> axes;
    GetStaticInputVector(node, 1, &axes);

    ng::Shape input_shape = ng_input.get_shape();
    size_t input_rank = input_shape.size();

    TF_RETURN_IF_ERROR(CheckAxisDimInRange(axes, input_rank));

    std::vector<size_t> ng_reduction_axes_vect(axes.size());
    std::transform(axes.begin(), axes.end(), ng_reduction_axes_vect.begin(), [input_rank](int idx) {
        return idx + (idx < 0 ? (int)input_rank : 0);
    });
    auto ng_reduction_axes = ConstructNgNode<opset::Constant>(node.get_name(),
                                                              ng::element::i64,
                                                              ng::Shape{ng_reduction_axes_vect.size()},
                                                              ng_reduction_axes_vect);

    ng::Output<ng::Node> ng_node = create_ng_node(ng_input, ng_reduction_axes, tf_keep_dims);

    return {ng_node};
}

template <typename T>
static OutputVector TranslateDirectReduceOp(const NodeContext& node) {
    // ensure its either an arithmetic or a logical reduction
    if (!(std::is_base_of<ngraph::op::util::ArithmeticReduction, T>::value ||
          std::is_base_of<ngraph::op::util::LogicalReduction, T>::value)) {
        throw errors::InvalidArgument("Expected node to be either a valid logical or arithmetic reduction "
                                      "type");
    }
    return TranslateReduceOp(
        node,
        [&node](ng::Output<ng::Node> ng_input, ng::Output<ng::Node> ng_reduction_axes, const bool keep_dims) {
            return ConstructNgNode<T>(node.get_name(), ng_input, ng_reduction_axes, keep_dims);
        });
}

#if 0

static OutputVector TranslateOneHotOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_features, ng_unused, ng_on, ng_off, ng_depth;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_features, ng_unused, ng_on, ng_off));

  auto ng_features_shape = ng_features.get_shape();
  std::vector<int> depth;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &depth));

  // Depth must be scalar
  if (depth.size() != 1) {
    return errors::InvalidArgument(
        "OneHot Op: depth of one hot dimension must be scalar " + to_string(depth.size()));
  }

  auto const_depth = ConstructNgNode<ng::op::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{}, depth);

  int one_hot_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &one_hot_axis));

  auto ng_onehot = ConstructNgNode<opset::OneHot>(
      node.get_name(), ng_features, const_depth, ng_on, ng_off, one_hot_axis);
  SaveNgOp(ng_op_map, node.get_name(), ng_onehot);
  return Status::OK();
}

static Status TranslatePackOp(const TFNodeDecoder* op, const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

  int32_t tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  auto ng_axis = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{1},
      std::vector<int64_t>({tf_axis}));

  ng::OutputVector ng_concat_inputs;
  for (int32_t i = 0; i < op->num_inputs(); ++i) {
    ng::Output<ng::Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_input));
    auto unsqueezed_input =
        ConstructNgNode<opset::Unsqueeze>(node.get_name(), ng_input, ng_axis);
    ng_concat_inputs.push_back(unsqueezed_input);
  }

  // if inputs shape is (2, 3, 4), and axis is 1, then we want
  // to create output_shape (2, num_inputs, 3, 4)
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<opset::Concat>(
                                      node.get_name(), ng_concat_inputs, tf_axis));
  return Status::OK();
}

#endif

// 3 different Pad Ops: Pad, PadV2, MirrorPad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad-v2
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mirror-pad
static OutputVector TranslatePadOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_paddings_op = node.get_ng_input(1);
    ng::Output<ng::Node> pad_val_op;

    // Set inputs and pad_val_op
    auto op_type = node.get_op_type();
    if (op_type == "Pad" || op_type == "MirrorPad") {
        pad_val_op = ConstructNgNode<opset::Constant>(node.get_name(),
                                                      ng_input.get_element_type(),
                                                      ng::Shape(),
                                                      std::vector<int>({0}));
    } else if (op_type == "PadV2") {
        pad_val_op = node.get_ng_input(2);
    } else {
        throw errors::InvalidArgument("Incorrect TF Pad OpType: " + node.get_op_type());
    }

    // Set pad_mode
    auto pad_mode = ng::op::PadMode::CONSTANT;
    if (op_type == "MirrorPad") {
        auto pad_mode_str = node.get_attribute<std::string>("mode");
        if (pad_mode_str == "REFLECT") {
            pad_mode = ng::op::PadMode::REFLECT;
        } else if (pad_mode_str == "SYMMETRIC") {
            pad_mode = ng::op::PadMode::SYMMETRIC;
        } else {
            throw errors::InvalidArgument(pad_mode_str + " is not an allowed padding mode.");
        }
    }

    // Set pads_begin & pads_end (from the pad_val_op)
    std::vector<int64_t> paddings;
    GetStaticInputVector(node, 1, &paddings);
    NGRAPH_VLOG(3) << node.get_name() << " pads {" << ng::join(paddings) << "}";
    if (paddings.size() % 2 != 0) {
        throw errors::InvalidArgument("Constant node for paddings does not have an even number of "
                                      "elements");
    }
    std::vector<int64_t> pad_begin(paddings.size() / 2);
    std::vector<int64_t> pad_end(paddings.size() / 2);
    for (size_t i = 0; i < paddings.size() / 2; i++) {
        pad_begin[i] = paddings[2 * i];
        pad_end[i] = paddings[2 * i + 1];
    }
    auto pads_begin_node =
        ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64, ng::Shape{pad_begin.size()}, pad_begin);
    auto pads_end_node =
        ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64, ng::Shape{pad_end.size()}, pad_end);

    // Create final Op
    auto result_pad_op =
        ConstructNgNode<opset::Pad>(node.get_name(), ng_input, pads_begin_node, pads_end_node, pad_val_op, pad_mode);

    return {result_pad_op};
}

#if 0

static OutputVector TranslateRangeOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_start, ng_stop, ng_step;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_start, ng_stop, ng_step));

  //DataType start_type = op->input_type(0);
  //DataType stop_type = op->input_type(1);
  //DataType step_type = op->input_type(2);
  ng::element::Type out_type;
  TF_RETURN_IF_ERROR(
      TFDataTypeToNGraphElementType(op->output_type(0), &out_type));
  //ng::Output<ng::Node> start_node, stop_node, step_node;
  //TF_RETURN_IF_ERROR(
  //    GetStaticInputNode(op, 0, static_input_map, start_type, start_node));
  //TF_RETURN_IF_ERROR(
  //    GetStaticInputNode(op, 1, static_input_map, stop_type, stop_node));
  //TF_RETURN_IF_ERROR(
  //    GetStaticInputNode(op, 2, static_input_map, step_type, step_node));
  auto ng_range = ConstructNgNode<opset::Range>(node.get_name(), ng_start,
                                                ng_stop, ng_step, out_type);

  SaveNgOp(ng_op_map, node.get_name(), ng_range);
  return Status::OK();
}

static Status TranslateRankOp(const TFNodeDecoder* op, const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  ng::Shape input_shape = ng_input.get_shape();
  auto input_rank = static_cast<int>(input_shape.size());

  auto ng_rank = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i32, ng::Shape(),
      std::vector<int>({input_rank}));

  SaveNgOp(ng_op_map, node.get_name(), ng_rank);
  return Status::OK();
}

static OutputVector TranslateReciprocalOp(
    const NodeContext& node) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        // Create a constant tensor populated with the value -1.
        // (1/x = x^(-1))
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-1");
        auto ng_exponent = ConstructNgNode<opset::Constant>(
            node.get_name(), et, shape, constant_values);

        // Raise each element of the input to the power -1.
        return ConstructNgNode<opset::Power>(node.get_name(), n, ng_exponent);
      });
}

static Status TranslateRelu6Op(const TFNodeDecoder* op,
                               const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                               Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<opset::Clamp>(node.get_name(), ng_input, 0, 6));
  return Status::OK();
}

static OutputVector TranslateReshapeOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_input, ng_shape_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_shape_op));

  NGRAPH_VLOG(3) << "Input shape: " << ng::join(ng_input.get_shape());

  std::vector<int64_t> shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &shape));

  NGRAPH_VLOG(3) << "Requested result shape: " << ng::join(shape);

  auto ng_shape = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{shape.size()}, shape);
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<opset::Reshape>(
                                      node.get_name(), ng_input, ng_shape, false));
  return Status::OK();
}

static OutputVector TranslateRsqrtOp(
    const NodeContext& node) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        // Create a constant tensor populated with the value -1/2.
        // (1/sqrt(x) = x^(-1/2))
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-0.5");
        auto ng_exponent = ConstructNgNode<opset::Constant>(
            node.get_name(), et, shape, constant_values);

        // Raise each element of the input to the power -0.5.
        return ConstructNgNode<opset::Power>(node.get_name(), n, ng_exponent);
      });
}

static Status TranslateShapeOp(const TFNodeDecoder* op,
                               const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                               Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  // default output_type = element::i64
  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<opset::ShapeOf>(node.get_name(), ng_input, type));
  return Status::OK();
}

static Status TranslateSizeOp(const TFNodeDecoder* op, const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  // Size has an attribute to specify output, int32_t or int64_t
  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  auto ng_input_shape = ng_input.get_shape();
  int64_t result = 1;
  for (auto dim : ng_input_shape) {
    result *= dim;
  }

  // make a scalar with value equals to result
  auto ng_result = ConstructNgNode<opset::Constant>(
      node.get_name(), type, ng::Shape(0), std::vector<int64_t>({result}));

  SaveNgOp(ng_op_map, node.get_name(), ng_result);
  return Status::OK();
}

static OutputVector TranslateSliceOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_input, ng_begin, ng_size;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_begin, ng_size));

  std::vector<int64_t> begin_vec;
  std::vector<int64_t> size_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &begin_vec));
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 2, static_input_map, &size_vec));

  if (begin_vec.size() != size_vec.size())
    return errors::InvalidArgument(
        "Cannot translate slice op: size of begin = " + to_string(begin_vec.size()) +
        ", size of size_vec = " + to_string(size_vec.size()) + ". Expected them to match.");

  NGRAPH_VLOG(3) << "Begin input for Slice: " << ng::join(begin_vec);
  NGRAPH_VLOG(3) << "Size input for Slice: " << ng::join(size_vec);

  std::vector<int64_t> end_vec(begin_vec.size());
  const auto ng_input_shape = ng_input.get_shape();
  stringstream err_stream;
  string err_msg;
  for (size_t i = 0; i < size_vec.size(); i++) {
    if (size_vec[i] != -1) {
      end_vec[i] = begin_vec[i] + size_vec[i];
    } else {
      // support -1 for size_vec, to the end of the tensor
      end_vec[i] = ng_input_shape[i];
    }

    // check for this condition: 0 <= begin[i] <= begin[i] + size[i] <= Di
    if (0 > begin_vec[i])
      err_stream << "lower < 0: " << begin_vec[i]
                 << ". It should have been positive.\n";
    if (begin_vec[i] > end_vec[i])
      err_stream << "upper < lower: upper = " << end_vec[i]
                 << ", lower = " << begin_vec[i] << "\n";
    if (begin_vec[i] > ng_input_shape[i])
      err_stream << "dim < upper: dim = " << ng_input_shape[i]
                 << ", upper = " << end_vec[i] << "\n";

    err_msg = err_stream.str();
    if (!err_msg.empty())
      return errors::InvalidArgument("Cannot translate slice op at position " +
                                     to_string(i) + " of " + to_string(size_vec.size()) +
                                     ". The reasons are:\n" + err_msg);
  }

  auto begin = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{begin_vec.size()}, begin_vec);
  auto end = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{end_vec.size()}, end_vec);

  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<opset::StridedSlice>(node.get_name(), ng_input, begin,
                                                end, std::vector<int64_t>{},
                                                std::vector<int64_t>{}));
  return Status::OK();
}

#endif

static OutputVector TranslateSoftmaxOp(const NodeContext& node) {
    auto ng_inp = node.get_ng_input(0);
    auto inp_shape = ng_inp.get_shape();
    size_t rank = inp_shape.size();
    int64_t axes = rank - 1;
    if (rank < 1) {
        throw errors::InvalidArgument("TF Softmax logits must be >=1 dimension");
    }

    return {ConstructNgNode<opset::Softmax>(node.get_name(), ng_inp, axes)};
}

#if 0

// Translate SpaceToDepthOp
static Status TranslateSpaceToDepthOp(const TFNodeDecoder* op,
                                      const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                      Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  // Get the attributes
  int64_t block_size;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "block_size", &block_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "DepthToSpace data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);
  auto ng_mode = opset::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
  auto space_to_depth = ConstructNgNode<opset::SpaceToDepth>(
      node.get_name(), ng_input, ng_mode, block_size);
  NCHWtoNHWC(node.get_name(), is_nhwc, space_to_depth);
  SaveNgOp(ng_op_map, node.get_name(), space_to_depth);
  return Status::OK();
}

static OutputVector TranslateSplitOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, ng_input));
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int32_t num_split;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_split", &num_split));

  ng::Shape shape = ng_input.get_shape();
  int rank = shape.size();

  std::vector<int> split_dim_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 0, static_input_map, &split_dim_vec));
  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64_t)rank : 0);
  auto ng_split_dim = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::u64, ng::Shape{}, split_dim);
  auto ng_split = make_shared<opset::Split>(ng_input, ng_split_dim, num_split);

  for (int i = 0; i < num_split; ++i) {
    auto out = ng_split->output(i);
    Builder::SetTracingInfo(node.get_name(), out);
    SaveNgOp(ng_op_map, node.get_name(), out);
  }
  return Status::OK();
}

static OutputVector TranslateSplitVOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_input, ng_split_length, ng_split_dim;

  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  ng::Shape shape = ng_input.get_shape();
  int rank = shape.size();

  std::vector<int64_t> split_dim_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 2, static_input_map, &split_dim_vec));
  // there should be at least one element specified as axis and not more than
  // one as axis is 0-D
  if (split_dim_vec.size() != 1) {
    return errors::InvalidArgument(
        "split_dim_tensor must have "
        "exactly one element.");
  }
  TF_RETURN_IF_ERROR(CheckAxisDimInRange(split_dim_vec, rank));
  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64_t)rank : 0);
  ng_split_dim = ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i32,
                                                  ng::Shape{}, split_dim);

  std::vector<int> split_lengths_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 1, static_input_map, &split_lengths_vec));

  // length: Length of size_splits
  int length = 0;
  int idx = -1;

  // Find out the total length of the splits and locate -1 's index, if any
  bool has_one_neg = false;
  for (size_t i = 0; i < split_lengths_vec.size(); ++i) {
    if (split_lengths_vec[i] != -1) {
      length += split_lengths_vec[i];
    } else {
      if (has_one_neg) {
        return errors::InvalidArgument("size_splits can only have one -1");
      } else {
        idx = i;
        has_one_neg = true;
      }
    }
  }

  // Size splits must sum to the dimension of value along split_dim
  if (idx > 0) {
    split_lengths_vec[idx] = shape[split_dim] - length;
  }

  if ((!has_one_neg && length != shape[split_dim]) ||
      (has_one_neg && split_lengths_vec[idx] < 0)) {
    return errors::InvalidArgument(
        "The length of size_splits must sum to the value of the dimension "
        "along split_dim");
  }

  ng_split_length = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i32, ng::Shape{split_lengths_vec.size()},
      split_lengths_vec);

  if (split_lengths_vec.size() != 1) {
    auto ng_split = make_shared<opset::VariadicSplit>(ng_input, ng_split_dim,
                                                      ng_split_length);
    for (size_t i = 0; i < split_lengths_vec.size(); ++i) {
      auto out = ng_split->output(i);
      Builder::SetTracingInfo(node.get_name(), out);
      SaveNgOp(ng_op_map, node.get_name(), out);
    }
  } else {
    SaveNgOp(ng_op_map, node.get_name(), ng_input);
  }

  return Status::OK();
}

static OutputVector TranslateSquareOp(
    const NodeContext& node) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        return ConstructNgNode<opset::Multiply>(node.get_name(), n, n);
      });
}

#endif

static OutputVector TranslateSqueezeOp(const NodeContext& node) {
    ng::Output<ng::Node> ng_input = node.get_ng_input(0);
    size_t input_dims = ng_input.get_shape().size();

    auto tf_axis = node.get_attribute<std::vector<int32_t>>("squeeze_dims");

    // If input dimension is negative, make it positive
    for (size_t i = 0; i < tf_axis.size(); i++) {
        tf_axis[i] = tf_axis[i] < 0 ? (int32_t)(input_dims) + tf_axis[i] : tf_axis[i];
    }

    auto ng_const =
        ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i32, ng::Shape{tf_axis.size()}, tf_axis);

    return {ConstructNgNode<opset::Squeeze>(node.get_name(), ng_input, ng_const)};
}

/*
static OutputVector ArgOp(const NodeContext& node) {
    auto ng_et = node.get_attribute<ngraph::element::Type>("T");
    auto overridden_shape = node.get_overridden_shapes().find(node.get_name());
    auto index = node.get_attribute<int>("index");
    auto shape = node.get_indexed_shapes().at(index);
    auto ng_shape = overridden_shape == node.get_overridden_shapes().end() ?
                    shape :
                    overridden_shape->second;
    return {ConstructNgNode<opset::Parameter>(node.get_name(), ng_et, ng_shape)};
}
 */

static OutputVector PlaceholderOp(const NodeContext& node) {
    auto ng_et = node.get_attribute<ngraph::element::Type>("dtype");
    auto overridden_shape = node.get_overridden_shapes().find(node.get_name());
    auto ng_shape = overridden_shape == node.get_overridden_shapes().end()
                        ? node.get_attribute<ngraph::PartialShape>("shape", ngraph::PartialShape())
                        : overridden_shape->second;
    return {ConstructNgNode<opset::Parameter>(node.get_name(), ng_et, ng_shape)};

#if 0  // Old code
    // TODO: Remove this section completely when all code is absorbed to main flow
    DataType dtype;
    // TODO: replace dtype by T when converting Arg
    if (GetNodeAttr(parm->attrs(), "dtype", &dtype) != Status::OK()) {
      throw errors::InvalidArgument("No data type defined for _Arg");
    }

    // TODO: use this code for Arg
    //if (GetNodeAttr(parm->attrs(), "index", &index) != Status::OK()) {
    //  return errors::InvalidArgument("No index defined for _Arg");
    //}

    ng::element::Type ng_et;
    TFDataTypeToNGraphElementType(dtype, &ng_et);

    ng::PartialShape ng_shape;
    auto overridenInputShape = inputs.find(parm->name());
    if(overridenInputShape == inputs.end()) {
        try {
            GetNodeAttr(parm->attrs(), "shape", &ng_shape);
        }
        catch (google::protobuf::FatalException) {
            // suppose there is no shape
            // TODO: do it in a good way
        }
    } else {
        ng_shape = overridenInputShape->second;
    }

#    if 0
    string prov_tag;
    GetNodeAttr(parm->attrs(), "_prov_tag", &prov_tag);
#    endif
    auto ng_param =
        ConstructNgNode<opset::Parameter>(parm->name(), ng_et, ng_shape);
    SaveNgOp(ng_op_map, parm->name(), ng_param);
#endif
}

static OutputVector RetvalOp(const NodeContext& node) {
    // Make sure that this _Retval only has one input node.
    if (node.get_ng_input_size() != 1) {
        throw errors::InvalidArgument("_Retval has " + to_string(node.get_ng_input_size()) + " inputs, should have 1");
    }

    // auto ret_val_index = node.get_attribute<int>("index");
    // TODO: Put ret_val_index to RT info that should be later utilized to order outpus by indices

    return {ConstructNgNode<opset::Result>(node.get_name(), node.get_ng_input(0))};
}

#if 0

static OutputVector TranslateStridedSliceOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  int32_t begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "new_axis_mask", &new_axis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "shrink_axis_mask", &shrink_axis_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ellipsis_mask", &ellipsis_mask));

  NGRAPH_VLOG(5) << "strided slice attributes: "
                 << "  begin mask: " << begin_mask << "  end mask: " << end_mask
                 << "  new axis mask: " << new_axis_mask
                 << "  shrink axis mask: " << shrink_axis_mask
                 << "  ellipsis mask: " << ellipsis_mask;

  std::vector<int64_t> begin_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &begin_vec));
  std::vector<int64_t> end_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 2, static_input_map, &end_vec));
  std::vector<int64_t> stride_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 3, static_input_map, &stride_vec));

  auto begin = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{begin_vec.size()}, begin_vec);
  auto end = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{end_vec.size()}, end_vec);
  auto strides = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{stride_vec.size()}, stride_vec);

  auto mask_to_vec = [](int32_t mask) {
    auto length = sizeof(mask) * CHAR_BIT;
    std::vector<int64_t> vec(length, 0);
    if (mask == 0) {
      return vec;
    }
    for (auto i = 0; i < length; ++i) {
      if ((unsigned char)(mask >> i & 0x01) == 1) {
        vec[i] = 1;
      }
    }
    return vec;
  };

  SaveNgOp(
      ng_op_map, node.get_name(),
      ConstructNgNode<opset::StridedSlice>(
          node.get_name(), ng_input, begin, end, strides, mask_to_vec(begin_mask),
          mask_to_vec(end_mask), mask_to_vec(new_axis_mask),
          mask_to_vec(shrink_axis_mask), mask_to_vec(ellipsis_mask)));
  return Status::OK();
}

static OutputVector TranslateTileOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_input, ng_multiples;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_multiples));

  std::vector<int64_t> multiples;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &multiples));

  auto ng_repeats = ConstructNgNode<opset::Constant>(
      node.get_name(), ng::element::i64, ng::Shape{multiples.size()}, multiples);
  SaveNgOp(ng_op_map, node.get_name(),
           ConstructNgNode<opset::Tile>(node.get_name(), ng_input, ng_repeats));
  return Status::OK();
}

// Translate TopKV2 Op using ngraph core op TopK
static OutputVector TranslateTopKV2Op(
    const NodeContext& node) {
  ng::Output<ngraph::Node> ng_input;

  TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  // axis along which to compute top k indices
  int64_t k_axis = ng_input.get_shape().size() - 1;

  // scalar input tensor specifying how many max/min elts should be computed
  // CPU backend only supports element type i64
  std::vector<int64_t> ng_k_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &ng_k_vec));
  auto ng_k = ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64,
                                               ng::Shape{}, ng_k_vec[0]);

  std::string mode = "max";

  std::string sort = "value";
  bool sorted = true;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "sorted", &sorted));
  if (!sorted) {
    sort = "index";
  }

  auto ng_result =
      std::make_shared<opset::TopK>(ng_input, ng_k, k_axis, mode, sort);

  ng::Output<ng::Node> ng_values = ng_result->output(0);
  Builder::SetTracingInfo(node.get_name(), ng_values);
  ng::Output<ng::Node> ng_indices = ng_result->output(1);
  Builder::SetTracingInfo(node.get_name(), ng_indices);

  SaveNgOp(ng_op_map, node.get_name(), ng_values);
  SaveNgOp(ng_op_map, node.get_name(), ng_indices);

  return Status::OK();
}

static OutputVector TranslateTransposeOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_input, ng_permutation;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_permutation));
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<opset::Transpose>(
                                      node.get_name(), ng_input, ng_permutation));
  return Status::OK();
}

static Status TranslateUnpackOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  int32_t tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  int32_t num_outputs;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num", &num_outputs));

  auto input_shape = ng_input.get_shape();
  auto rank = input_shape.size();
  for (int i = 0; i < num_outputs; ++i) {
    std::vector<int64_t> begin(rank, 0);
    std::vector<int64_t> end(rank, 0);
    begin[tf_axis] = i;
    end[tf_axis] = i + 1;
    auto ng_begin = ConstructNgNode<opset::Constant>(
        node.get_name(), ng::element::i64, ng::Shape{begin.size()}, begin);
    auto ng_end = ConstructNgNode<opset::Constant>(node.get_name(), ng::element::i64,
                                                   ng::Shape{end.size()}, end);
    std::vector<int64_t> begin_mask(rank, 1);
    begin_mask[tf_axis] = 0;
    std::vector<int64_t> end_mask(rank, 1);
    end_mask[tf_axis] = 0;
    std::vector<int64_t> new_axis_mask(rank, 0);
    std::vector<int64_t> shrink_axis_mask(rank, 0);
    shrink_axis_mask[tf_axis] = 1;
    auto slice = ConstructNgNode<opset::StridedSlice>(
        node.get_name(), ng_input, ng_begin, ng_end, begin_mask, end_mask,
        new_axis_mask, shrink_axis_mask);
    SaveNgOp(ng_op_map, node.get_name(), slice);
  }
  return Status::OK();
}

static OutputVector TranslateXdivyOp(
    const NodeContext& node) {
  ng::Output<ngraph::Node> ng_x, ng_y;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_y));
  auto zero =
      ConstructNgNode<opset::Constant>(node.get_name(), ng_x.get_element_type(),
                                       ngraph::Shape{}, std::vector<int>({0}));
  auto x_is_zero = ConstructNgNode<opset::Equal>(node.get_name(), ng_x, zero);
  auto ng_xdivy = ConstructNgNode<opset::Divide>(node.get_name(), ng_x, ng_y);
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<opset::Select>(
                                      node.get_name(), x_is_zero, ng_x, ng_xdivy));
  return Status::OK();
}

static Status TranslateSelectOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input1, ng_input2, ng_input3;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input1, ng_input2, ng_input3));
  auto ng_select = ConstructNgNode<opset::Select>(node.get_name(), ng_input1,
                                                  ng_input2, ng_input3);
  SaveNgOp(ng_op_map, node.get_name(), ng_select);
  return Status::OK();
}

static OutputVector TranslateWhereOp(
    const NodeContext& node) {
  ng::Output<ng::Node> ng_cond;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_cond));
  auto non_zero = ConstructNgNode<opset::NonZero>(node.get_name(), ng_cond);
  auto transpose_order = ConstructNgNode<opset::Constant>(
      node.get_name(), ngraph::element::i64, ngraph::Shape{2},
      std::vector<int64_t>({1, 0}));
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<opset::Transpose>(
                                      node.get_name(), non_zero, transpose_order));
  return Status::OK();
}

static Status TranslateZerosLikeOp(const TFNodeDecoder* op,
                                   const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                   Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  ng::Shape input_shape = ng_input.get_shape();
  std::vector<std::string> const_values(ng::shape_size(input_shape), "0");
  auto ng_result = ConstructNgNode<opset::Constant>(
      node.get_name(), ng_input.get_element_type(), input_shape, const_values);
  SaveNgOp(ng_op_map, node.get_name(), ng_result);
  return Status::OK();
}

#endif

static std::map<const string, const function<ngraph::OutputVector(const NodeContext&)>> TRANSLATE_OP_MAP{
    {"Abs", TranslateUnaryOp<opset::Abs>},
    {"Acos", TranslateUnaryOp<opset::Acos>},
    {"Acosh", TranslateUnaryOp<opset::Acosh>},
    {"Add", TranslateBinaryOp<opset::Add>},
    {"AddN", TranslateAddNOp},
    {"AddV2", TranslateBinaryOp<opset::Add>},
    {"Any", TranslateDirectReduceOp<opset::ReduceLogicalOr>},
    {"All", TranslateDirectReduceOp<opset::ReduceLogicalAnd>},
    {"ArgMax", TranslateArgMaxOp},
    {"ArgMin", TranslateArgMinOp},
    {"Asin", TranslateUnaryOp<opset::Asin>},
    {"Asinh", TranslateUnaryOp<opset::Asinh>},
    {"Atan", TranslateUnaryOp<opset::Atan>},
    {"Atanh", TranslateUnaryOp<opset::Atanh>},
    {"AvgPool", TranslateAvgPoolOp},
    {"BiasAdd", TranslateBiasAddOp},
    {"Cast", TranslateCastOp},
    {"Ceil", TranslateUnaryOp<opset::Ceiling>},
    {"ConcatV2", TranslateConcatV2Op},
    {"Const", TranslateConstOp},
    {"Conv2D", TranslateConv2DOp},
    {"Conv2DBackpropInput", TranslateConv2DBackpropInputOp},
    {"Conv3D", TranslateConv3DOp},
    {"Cos", TranslateUnaryOp<opset::Cos>},
    {"Cosh", TranslateUnaryOp<opset::Cosh>},
    {"Cumsum", TranslateCumsumOp},
    {"DepthToSpace", TranslateDepthToSpaceOp},
    {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
    {"Equal", TranslateBinaryOp<opset::Equal>},
    {"Exp", TranslateUnaryOp<opset::Exp>},
    {"ExpandDims", TranslateExpandDimsOp},
    {"Fill", TranslateFillOp},
    {"Floor", TranslateUnaryOp<opset::Floor>},
    {"FloorDiv", TranslateFloorDivOp},
    {"FloorMod", TranslateBinaryOp<opset::FloorMod>},
    {"FusedBatchNorm", TranslateFusedBatchNormOp},
    {"FusedBatchNormV2", TranslateFusedBatchNormOp},
    {"FusedBatchNormV3", TranslateFusedBatchNormOp},
    {"Gather", TranslateGatherOp},
    {"GatherV2", TranslateGatherV2Op},
    {"_FusedConv2D", TranslateFusedConv2DOp},
    {"_FusedMatMul", TranslateFusedMatMulOp},
    {"Greater", TranslateBinaryOp<opset::Greater>},
    {"GreaterEqual", TranslateBinaryOp<opset::GreaterEqual>},
    {"Identity", TranslateIdentityOp},
    //{"IsFinite", TranslateIsFiniteOp},
    //{"L2Loss", TranslateL2LossOp},
    {"LogSoftmax", TranslateLogSoftmaxOp},
    {"Less", TranslateBinaryOp<opset::Less>},
    {"LessEqual", TranslateBinaryOp<opset::LessEqual>},
    {"Log", TranslateUnaryOp<opset::Log>},
    //{"Log1p", TranslateLog1pOp},
    {"LogicalAnd", TranslateBinaryOp<opset::LogicalAnd>},
    {"LogicalNot", TranslateUnaryOp<opset::LogicalNot>},
    {"LogicalOr", TranslateBinaryOp<opset::LogicalOr>},
    //{"LRN", TranslateLRNOp},
    //{"MatMul", TranslateMatMulOp},
    {"Max", TranslateDirectReduceOp<opset::ReduceMax>},
    {"Maximum", TranslateBinaryOp<opset::Maximum>},
    {"MaxPool", TranslateMaxPoolOp<2>},
    {"MaxPool3D", TranslateMaxPoolOp<3>},
    //{"NonMaxSuppressionV2", TranslateNonMaxSuppressionV2Op},
    {"Mean", TranslateDirectReduceOp<opset::ReduceMean>},
    {"Min", TranslateDirectReduceOp<opset::ReduceMin>},
    {"Minimum", TranslateBinaryOp<opset::Minimum>},
    {"MirrorPad", TranslatePadOp},
    {"Mul", TranslateBinaryOp<opset::Multiply>},
    {"Mod", TranslateBinaryOp<opset::Mod>},
    {"Neg", TranslateUnaryOp<opset::Negative>},
    {"NotEqual", TranslateBinaryOp<opset::NotEqual>},
    // Do nothing! NoOps sometimes get placed on nGraph for bureaucratic
    // reasons, but they have no data flow inputs or outputs.
    {"NoOp",
     [](const NodeContext& node) {
         if (node.get_ng_input_size() == 0) {
             return OutputVector{};
         }
         if (node.get_ng_input_size() != 1) {
             throw errors::InvalidArgument("NoOp has " + to_string(node.get_ng_input_size()) +
                                           " inputs, should have 1");
         }
         return OutputVector{node.get_ng_input(0)};
     }},
    //{"OneHot", TranslateOneHotOp},
    //{"Pack", TranslatePackOp},
    {"Pad", TranslatePadOp},
    {"PadV2", TranslatePadOp},
    //{"_Arg", ArgOp}, // should be registered as an extension in OVTF
    {"Placeholder", PlaceholderOp},
    {"Pow", TranslateBinaryOp<opset::Power>},
    // PreventGradient is just Identity in dataflow terms, so reuse that.
    {"PreventGradient", TranslateIdentityOp},
    {"Prod", TranslateDirectReduceOp<opset::ReduceProd>},
    //{"Range", TranslateRangeOp},
    //{"Rank", TranslateRankOp},
    {"RealDiv", TranslateBinaryOp<opset::Divide>},
    //{"Reciprocal", TranslateReciprocalOp},
    {"Relu", TranslateUnaryOp<opset::Relu>},
    //{"Relu6", TranslateRelu6Op},
    //{"Reshape", TranslateReshapeOp},
    {"_Retval", RetvalOp},
    //{"Rsqrt", TranslateRsqrtOp},
    //{"Select", TranslateSelectOp},
    //{"SelectV2", TranslateSelectOp},
    //{"Shape", TranslateShapeOp},
    {"Sigmoid", TranslateUnaryOp<opset::Sigmoid>},
    {"Sin", TranslateUnaryOp<opset::Sin>},
    {"Sinh", TranslateUnaryOp<opset::Sinh>},
    //{"Size", TranslateSizeOp},
    {"Sign", TranslateUnaryOp<opset::Sign>},
    //{"Slice", TranslateSliceOp},
    //{"Snapshot", TranslateIdentityOp},
    {"Softmax", TranslateSoftmaxOp},
    {"Softplus", TranslateUnaryOp<opset::SoftPlus>},
    //{"SpaceToDepth", TranslateSpaceToDepthOp},
    //{"Split", TranslateSplitOp},
    //{"SplitV", TranslateSplitVOp},
    {"Sqrt", TranslateUnaryOp<opset::Sqrt>},
    //{"Square", TranslateSquareOp},
    {"SquaredDifference", TranslateBinaryOp<opset::SquaredDifference>},
    {"Squeeze", TranslateSqueezeOp},
    //{"StridedSlice", TranslateStridedSliceOp},
    {"Sub", TranslateBinaryOp<opset::Subtract>},
    {"Sum", TranslateDirectReduceOp<opset::ReduceSum>},
    {"Tan", TranslateUnaryOp<opset::Tan>},
    {"Tanh", TranslateUnaryOp<opset::Tanh>},
    //{"Tile", TranslateTileOp},
    //{"TopKV2", TranslateTopKV2Op},
    //{"Transpose", TranslateTransposeOp},
    //{"Unpack", TranslateUnpackOp},
    //{"Where", TranslateWhereOp},
    //{"Xdivy", TranslateXdivyOp},
    //{"ZerosLike", TranslateZerosLikeOp},
};

void Builder::TranslateGraph(
    const std::shared_ptr<ngraph::frontend::InputModelTF>& model,
    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map,
    const std::string model_name,
    bool fail_fast,
    bool no_conversion,
    std::shared_ptr<ngraph::Function>& ng_function) {
    // a map from operation names to generated nGraph Output<TFNodeDecoder>
    Builder::OpMap ng_op_map;

    ngraph::ParameterVector params;
    ngraph::ResultVector results;
    const auto& operation_places = model->get_op_places();
    const auto& model_inputs = model->get_inputs();
    const auto& model_outputs = model->get_outputs();
    const auto& model_frozen_inputs = model->get_tensor_values();

    std::map<const string, const function<ngraph::OutputVector(const NodeContext&)>> translate_map;

    if (no_conversion) {
        const std::set<std::string> required_types{"Placeholder", "_Retval", "NoOp"};
        for (auto& name : required_types) {
            translate_map.emplace(name, TRANSLATE_OP_MAP.at(name));
        }
    } else {
        translate_map = TRANSLATE_OP_MAP;
    }

    // fill ng_op_map with Constant outputs for frozen inputs
    for (const auto& frozen_input : model_frozen_inputs) {
        const auto& frozen_input_name = frozen_input.first;
        const auto& frozen_input_value = frozen_input.second;
        ng_op_map[frozen_input_name] = {frozen_input_value};
    }

    // create parameter nodes for all tensor places corresponding to inputs
    for (const auto& input_place : model_inputs) {
        FRONT_END_GENERAL_CHECK(input_place->get_names().size() == 1, "Input place must have one name.");
        auto input_name = input_place->get_names()[0];
        if (ng_op_map.count(input_name)) {
            // probably this input is frozen
            continue;
        }
        const auto& input_tensor_place = std::dynamic_pointer_cast<TensorPlaceTF>(input_place);
        auto input_shape = input_tensor_place->get_partial_shape();
        auto input_type = input_tensor_place->get_element_type();

        auto input_ng_output = ConstructNgNode<opset::Parameter>(input_name, input_type, input_shape);
        auto input_ng_node = std::dynamic_pointer_cast<opset::Parameter>(input_ng_output.get_node_shared_ptr());
        params.push_back(input_ng_node);
        ng_op_map[input_name] = {input_ng_output};
    }

    // create the nGraph ops from TensorFlow ops    
    for (auto& operation_place : operation_places) {
        auto operation_decoder = operation_place->get_desc();
        auto operation_name = operation_place->get_names()[0];

        // output for parameter nodes has been already generated
        if (ng_op_map.count(operation_name)) {
            continue;
        }

#if 0
    // TODO: Investigate why do we need it
      if (n->IsSink() || n->IsSource()) {
      continue;
    }
#endif

        // prepare a list of nGraph node inputs for each node
        ngraph::OutputVector ng_inputs;
        for (size_t input_port_idx = 0; input_port_idx < operation_decoder->num_inputs(); ++input_port_idx) {
            std::string producer_name;
            size_t producer_port_idx;
            try {
                operation_decoder->input_node(input_port_idx, &producer_name, &producer_port_idx);
            } catch (const std::exception& e) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(input_port_idx) +
                                " for op '"
                                + operation_decoder->name() + "', expected input name: '" + producer_name
                                + "', expected input port index: " + std::to_string(producer_port_idx) + '\n');
            }
            // TODO: re-implement the logic below once Place graph structure is implemented
            // Using Place graph structure (OpPlace, In/OutPortPlace places and their connections) can give
            // names of ports and operations that can be used for further check about existence in ng_op_map

            // check if output vector for places have been already defined and the order of this check is important
            // it moves from places corresponding to input port of the current operation node to output port of original
            // producers
            if (ng_op_map.count(std::to_string(input_port_idx) + ":" + operation_name)) {
                const auto& input_outputs_vector = ng_op_map.at(std::to_string(input_port_idx) + ":" + operation_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(0));
            } else if (ng_op_map.count(producer_name + ":" + std::to_string(producer_port_idx))) {
                const auto& input_outputs_vector =
                    ng_op_map.at(producer_name + ":" + std::to_string(producer_port_idx));
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(0));
            } else if (ng_op_map.count(producer_name)) {
                const auto& input_outputs_vector = ng_op_map.at(producer_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() > producer_port_idx,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(producer_port_idx));
            } else {
                FRONT_END_GENERAL_CHECK(false,
                                        "No input is found for node \"" + operation_name + "\" by port" +
                                            std::to_string(producer_port_idx));
            }
        }

        // generate nGraph node output vector for the current operation node
        ngraph::OutputVector ng_outputs;
        try {
            if (operation_decoder->IsControlFlow()) {
                FRONT_END_THROW("Encountered a control flow op in the nGraph bridge: " +
                                operation_decoder->DebugString());
            }

            FRONT_END_OP_CONVERSION_CHECK(translate_map.count(operation_decoder->type_string()),
                                          "No translator found for " + operation_decoder->type_string() + " node.");
            auto op_fun = &(translate_map[operation_decoder->type_string()]);
            NodeContext node_context(ng_inputs, operation_decoder, model_inputs);

            // generate nGraph node output vector using translator for given operation type
            ng_outputs = (*op_fun)(node_context);
        } catch (...) {
            if (fail_fast) {
                // re-throw any exception
                throw;
            } else {
                auto ng_node =
                    std::make_shared<ngraph::frontend::TFFrameworkNode>(operation_decoder,
                                                                        ng_inputs,
                                                                        operation_place->get_output_ports().size());
                Builder::SetTracingInfo(operation_name, ng_node);
                ng_outputs = ng_node->outputs();
            }
        }

        // register nGraph node outputs in the map for new operation node
        for (auto output : ng_outputs) {
            if (auto result = std::dynamic_pointer_cast<opset::Result>(output.get_node_shared_ptr())) {
                // do not add RetVal type operation to ng_op_map
                results.push_back(result);
            } else {
                if (auto param = std::dynamic_pointer_cast<opset::Parameter>(output.get_node_shared_ptr())) {
                    params.push_back(param);
                }
                ng_op_map[operation_name].push_back(output);
            }
        }
    }

    // create Result nodes for all model outputs
    for (const auto& model_output : model_outputs) {
        auto model_output_tensor_place = std::dynamic_pointer_cast<TensorPlaceTF>(model_output);
        auto model_output_name = model_output_tensor_place->get_names()[0];
        std::string operation_name;
        std::string port_type;
        size_t port_index;
        ngraph::frontend::tf::extract_operation_name_and_port(model_output_name, operation_name, port_index, port_type);

        if (port_type == "none") {
            for (const auto& node_output : ng_op_map[operation_name]) {
                results.push_back(std::make_shared<default_opset::Result>(node_output));
            }
        } else if (port_type == "out") {
            const auto& node_outputs = ng_op_map[operation_name];
            FRONT_END_GENERAL_CHECK(node_outputs.size() > port_index,
                                    "Output port with index " + std::to_string(port_index) + " of " + operation_name +
                                        "node specified as custom output does not exist");
            results.push_back(std::make_shared<default_opset::Result>(node_outputs[port_index]));
        } else if (port_type == "in") {
            // TODO: avoid this traversing by having a map for OpPlace objects, for example
            std::shared_ptr<OpPlaceTF> operation_place = nullptr;
            for (const auto& op_place : operation_places) {
                if (op_place->get_names()[0].compare(operation_name) == 0) {
                    operation_place = op_place;
                }
            }
            FRONT_END_GENERAL_CHECK(operation_place, "There is no operation place with a name: " + operation_name);
            auto operation_decoder = operation_place->get_desc();

            // get to know a producer node and by which its output port data is generated
            std::string producer_name;
            size_t producer_port_idx;
            try {
                operation_decoder->input_node(port_index, &producer_name, &producer_port_idx);
            } catch (const std::exception& e) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(port_index) +
                                " for op '" + operation_decoder->name() + "', expected input name: '" + producer_name +
                                "', expected input port index: " + std::to_string(producer_port_idx) + '\n');
            }

            // add Result node for this producer output port
            const auto& node_outputs = ng_op_map[producer_name];
            FRONT_END_GENERAL_CHECK(node_outputs.size() > producer_port_idx,
                                    "Output port with index " + std::to_string(producer_port_idx) + " of " +
                                        producer_name +
                                        "node specified as custom output does not exist");
            results.push_back(std::make_shared<default_opset::Result>(node_outputs[producer_port_idx]));
        }
    }
    
    // find all terminal nodes in ngraph graph to complete list of results
    if (results.empty()) {
        for (const auto& node_output_vector : ng_op_map) {
            for (auto output : node_output_vector.second) {
                if (output.get_target_inputs().empty() &&
                    !std::dynamic_pointer_cast<opset::Result>(output.get_node_shared_ptr())) {
                    results.push_back(std::make_shared<default_opset::Result>(output));                
                }
            }
        }
    }

    // TODO: reorder results and params according to indices given in RT info (if any)

    // create the nGraph function
    ng_function = make_shared<ng::Function>(results, params, model_name);

    // TODO: request row-major layout on results.
    // why do we need this?
    // for (auto result : ng_function->get_results()) {
    //  result->set_needs_default_layout(true);
    // }
    NGRAPH_VLOG(5) << "Done with translations";
}

void Builder::TranslateFWNode(const std::shared_ptr<TFFrameworkNode>& node) {
    auto type = node->get_op_type();

    auto translator_it = TRANSLATE_OP_MAP.find(type);
    FRONT_END_OP_CONVERSION_CHECK(translator_it != TRANSLATE_OP_MAP.end(), "No translator found for ", type, " node.");

    ngraph::OutputVector ng_inputs;
    for (auto& input : node->inputs()) {
        ng_inputs.push_back(input.get_source_output());
    }

    NodeContext node_ctx(ng_inputs, node->get_decoder(), {}, {});
    auto new_node_outputs = translator_it->second(node_ctx);
    Builder::SetTracingInfo(node_ctx.get_name(), new_node_outputs.front());

    auto new_output = new_node_outputs.begin();
    auto old_outputs = node->outputs();
    auto old_output = old_outputs.begin();

    for (; new_output != new_node_outputs.end() && old_output != old_outputs.end(); ++old_output, ++new_output) {
        old_output->replace(*new_output);
    }
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
