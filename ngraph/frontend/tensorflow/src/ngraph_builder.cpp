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

#include <numeric>
#include "graph.pb.h"
#include "tensor.pb.h"

#include "ngraph/op/util/logical_reduction.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/slice_plan.hpp"

#include "ngraph_builder.h"
#include "ngraph_conversions.h"
#include "default_opset.h"


using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

static bool VecStrCmp(const std::vector<string>& a,
                      const std::vector<string>& b) {
  return a == b;
}

static Status ValidateInputCount(const TFNodeDecoder* op, int32_t count) {
  if (op->num_inputs() != count) {
      std::ostringstream buf;
      buf << "\"" << op->name() << "\" requires " << count <<
              " input(s), got " << op->num_inputs() <<
              " instead";
    return errors::InvalidArgument(buf.str());
  }
  return Status::OK();
}

static Status ValidateInputCountMin(const TFNodeDecoder* op, int32_t count) {
  if (op->num_inputs() < count) {
      std::ostringstream buf;
      buf << "\"" << op->name() << "\" requires at least " <<
              count << " input(s), got " << op->num_inputs() <<
              " instead";
    return errors::InvalidArgument(buf.str());
  }
  return Status::OK();
}

// Check to make sure the axis dimension for reduction are in within range.
// Returns error if axis is out of range. Otherwise returns Status::OK().
static Status CheckAxisDimInRange(std::vector<int64_t> axes, size_t rank) {
  for (auto i : axes) {
    if (i < (int)-rank || i >= (int)rank) {
        std::ostringstream buf;
        buf << "Axis Dimension is out of range. Got " << i <<
                ", should be in range [-" << rank << ", " <<
                rank << ")";
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

static void SaveNgOp(Builder::OpMap& ng_op_map, const std::string& op_name,
                     ng::Output<ng::Node> output_node) {
  // no need to try-catch, map[key] will create vector object
  // if not exists
  ng_op_map[op_name].push_back(output_node);
}

void Builder::SetTracingInfo(const std::string& op_name,
                             const ng::Output<ng::Node> ng_node) {
  auto node = ng_node.get_node_shared_ptr();
  node->set_friendly_name(op_name + "/" + node->get_name());
  node->add_provenance_tag(op_name);
#if 0
  if (api::IsLoggingPlacement()) {
    cout << "TF_to_NG: " << op_name << " --> " << node << "\n";
  }
#endif
}

template <class TOpType, class... TArg>
ng::Output<ng::Node> ConstructNgNode(const std::string& op_name,
                                     TArg&&... Args) {
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

static Status GetInputNode(const Builder::OpMap& ng_op_map, const TFNodeDecoder* op,
                           size_t input_idx, ng::Output<ng::Node>& result) {
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

  const TFNodeDecoder* tf_input;
  size_t src_output_idx;
  TF_RETURN_IF_ERROR(op->input_node(input_idx, &tf_input, &src_output_idx));
  std::vector<ng::Output<ng::Node>> ng_op;
  try {
    ng_op = ng_op_map.at(tf_input->name());
  } catch (const out_of_range&) {
    return Status(string("Ngraph op not found for ") + tf_input->name());
  }
  try {
    result = ng_op.at(src_output_idx);
  } catch (const out_of_range&) {
    return Status(string("Input node not found at index ") +
                                        to_string(src_output_idx));
  }
  return Status::OK();



    NGRAPH_TF_FE_NOT_IMPLEMENTED
}

namespace detail {
static Status GetInputNodes(const Builder::OpMap&, const TFNodeDecoder*, size_t) {
  return Status::OK();
}

template <typename... Arguments>
static Status GetInputNodes(const Builder::OpMap& ng_op_map, const TFNodeDecoder* op,
                            size_t index, ng::Output<ng::Node>& result,
                            Arguments&... remaining) {
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, index, result));
  return GetInputNodes(ng_op_map, op, index + 1, remaining...);
}
}  // namespace detail

template <typename... Arguments>
static Status GetInputNodes(const Builder::OpMap& ng_op_map, const TFNodeDecoder* op,
                            Arguments&... remaining) {
  constexpr size_t args_len = sizeof...(Arguments);
  TF_RETURN_IF_ERROR(ValidateInputCount(op, args_len));
  return detail::GetInputNodes(ng_op_map, op, 0, remaining...);
}

static Status GetStaticNodeTensor(
    const TFNodeDecoder* node, const std::vector<const TensorWrapper*>& static_input_map,
    TensorWrapper* result) {
  if (node->IsArg()) {
    int arg_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &arg_index));
    const TensorWrapper* source_tensor = static_input_map[arg_index];
    if (source_tensor == nullptr) {
      return errors::Internal(
          "GetStaticNodeTensor called on _Arg but input tensor is missing from "
          "static input map");
    }
    *result = *source_tensor;
    return Status::OK();
  } else if (node->type_string() == "Const") {
    if (GetNodeAttr(node->attrs(), "value", &result).status != 0) {
      return errors::Internal(
          "GetStaticNodeTensor: Const tensor proto parsing failed");
    }
    return Status::OK();
  } else {
    return errors::Internal("GetStaticNodeTensor called on node with type " +
                            node->type_string() + "; _Arg or Const expected");
  }
}

template <typename Ttensor, typename Tvector>
static void ConvertTensorDataToVector(const TensorWrapper& tensor,
                                      std::vector<Tvector>* vector) {
  const Ttensor* data = tensor.flat<Ttensor>().data();
  vector->resize(tensor.NumElements());
  for (int64_t i = 0; i < tensor.NumElements(); i++) {
    (*vector)[i] = Tvector(data[i]);
  }
}

template <typename T>
static Status TensorDataToVector(const TensorWrapper& tensor, std::vector<T>* vector) {
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
static Status GetStaticInputVector(
        Builder::OpMap& ng_op_map,
        const TFNodeDecoder* op, int64_t input_index,
        const std::vector<const TensorWrapper*>& static_input_map,
        std::vector<T>* vector) {
    ng::Output<ng::Node> ng_input;
    GetInputNode(ng_op_map, op, input_index, ng_input);
    if(auto constant = std::dynamic_pointer_cast<ngraph::opset5::Constant>(ng_input.get_node_shared_ptr()))
    {
        *vector = constant->cast_vector<T>();
        return Status::OK();
    }

    NGRAPH_TF_FE_NOT_IMPLEMENTED;
/*
    TFNodeDecoder* input_node;
    TF_RETURN_IF_ERROR(op->input_node(input_index, &input_node));
    TensorWrapper* input_tensor;
    TF_RETURN_IF_ERROR(
            GetStaticNodeTensor(input_node, static_input_map, &input_tensor));
    TF_RETURN_IF_ERROR(TensorDataToVector(input_tensor, vector));
    return Status::OK();*/
}

#if 0
template <typename T>
static Status GetStaticInputVector(
    const TFNodeDecoder* op, int64_t input_index,
    const std::vector<const TensorWrapper*>& static_input_map,
    std::vector<T>* vector) {
  TFNodeDecoder* input_node;
  TF_RETURN_IF_ERROR(op->input_node(input_index, &input_node));
  TensorWrapper* input_tensor;
  TF_RETURN_IF_ERROR(
      GetStaticNodeTensor(input_node, static_input_map, &input_tensor));
  TF_RETURN_IF_ERROR(TensorDataToVector(input_tensor, vector));
  return Status::OK();
}

static Status GetStaticInputNode(
    const TFNodeDecoder* op, int64_t input_index,
    const std::vector<const TensorWrapper*>& static_input_map, DataType dt,
    ng::Output<ng::Node>& node_) {
  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dt, &type));
  switch (dt) {
    case DataType::DT_FLOAT: {
      std::vector<float> vec_float;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_float));
      node_ = ConstructNgNode<opset::Constant>(op->name(), type, ng::Shape{},
                                               vec_float[0]);
    } break;
    case DataType::DT_DOUBLE: {
      std::vector<double> vec_double;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_double));
      node_ = ConstructNgNode<opset::Constant>(op->name(), type, ng::Shape{},
                                               vec_double[0]);
    } break;
    case DataType::DT_INT32: {
      std::vector<int32_t> vec_i32;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_i32));
      node_ = ConstructNgNode<opset::Constant>(op->name(), type, ng::Shape{},
                                               vec_i32[0]);
    } break;
    case DataType::DT_INT64: {
      std::vector<int64_t> vec_i64;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_i64));
      node_ = ConstructNgNode<opset::Constant>(op->name(), type, ng::Shape{},
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
            node->getAttrValue("dtype", &dt);


            /*
            if (dt != DataTypeToEnum<T>::value) {
              std::stringstream ss;
              ss << "Invalid data type defined for Const. Defined: "
                 << node.attr().at("dtype").type();
              return errors::InvalidArgument(ss.str());
            }
             */

  // TensorWrapper represents the content of the tensor in either <type>_val or
  // tensor_content.
  TensorWrapper* tensor;
  node->getAttrValue("value", &tensor);
  //typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
  //    checkpoint::MutableTensorProtoData<T>(const_cast<TensorWrapper*>(&tensor));

  const TensorShapeProto& shape = tensor->tensor_def->tensor_shape();
  ngraph::PartialShape pshape;
    TFTensorShapeToNGraphShape(shape, &pshape);
            *const_tensor_shape = pshape.get_shape();
    if(pshape.is_dynamic())
        NGRAPH_TF_FE_NOT_IMPLEMENTED;
    auto tensor_content = tensor->tensor_def->tensor_content();
    std::vector<char> tensor_values_plain(tensor_content.begin(), tensor_content.end());
    const T* tensor_values = reinterpret_cast<const T*>(tensor_values_plain.data());

  if (!tensor_values_plain.empty() && tensor->tensor_def->has_tensor_shape()) {
    // When tensor_shape is set, theoretically the representation of the data
    // could be compressed. So, before copying values to the returned vector,
    // make sure no compression happens.
    //if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values_plain.size()/sizeof(T)) {
      values->insert(values->end(), tensor_values,
                     tensor_values + tensor_values_plain.size()/sizeof(T));
      return Status::OK();
    //}
  }

  const auto tensor_content_size = tensor->tensor_def->tensor_content().size();
  if(tensor_content_size % sizeof(VecT)) {
      std::cerr << "[ ERROR ] tensor_content_size (" << tensor_content_size
                << ") is not a multiple of " << sizeof(VecT);
  }

  // If tensor_content_size is zero, we'll have to take the values from
  // int_val, float_val, etc.
  if (tensor_content_size == 0) {
    int64_t n_elements = 1;
    for (auto i = 0; i < shape.dim_size(); i++) {
      if (shape.dim(i).size() < 0) {
        return errors::InvalidArgument(
            "Const node has empty tensor and an unknown dimension size");
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
          if (val_size > 0) val_i = tensor_proto.int_val()[i];
          break;
        case DT_INT64:
          val_size = tensor_proto.int64_val_size();
          if (val_size > 0) val_i = tensor_proto.int64_val()[i];
          break;
        case DT_FLOAT:
          val_size = tensor_proto.float_val_size();
          if (val_size > 0) val_i = tensor_proto.float_val()[i];
          break;
        case DT_BOOL:
          val_size = tensor_proto.bool_val_size();
          if (val_size > 0) val_i = tensor_proto.bool_val()[i];
          break;
        case DT_DOUBLE:
          val_size = tensor_proto.double_val_size();
          if (val_size > 0) val_i = tensor_proto.double_val()[i];
          break;
        default:
          NGRAPH_VLOG(0)
              << "Const node has empty tensor_proto and we don't know how to "
                 "handle this element type";
          NGRAPH_VLOG(0) << node->DebugString();
          NGRAPH_VLOG(0) << shape.DebugString();
          return errors::Unimplemented("Encountered unknown element type " +
                                       DataType_Name(dt) +
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
      //values->resize(tensor_content_size / sizeof(VecT));
    //port::CopyToArray(tensor.tensor_content(),
    //                  reinterpret_cast<char*>(values->data()));
  }

  return Status::OK();
#endif


}

// Helper for Builder::TranslateGraph ("Const" op)
template <typename T, typename VecT = T>
static Status MakeConstOp(const TFNodeDecoder* op, ng::element::Type et,
                          ng::Output<ng::Node>& ng_node) {
  vector<VecT> const_values;
  ngraph::Shape ng_shape;

  TF_RETURN_IF_ERROR(
  (ValuesFromConstNode<T, VecT>(op, &ng_shape, &const_values)));

  ng_node =
      ConstructNgNode<opset::Constant>(op->name(), et, ng_shape, const_values);
  return Status::OK();
}

const Builder::ConstMap& Builder::TF_NGRAPH_CONST_MAP() {
  static const Builder::ConstMap the_map = {
      {DataType::DT_FLOAT, make_pair(MakeConstOp<float>, ng::element::f32)},
      {DataType::DT_DOUBLE, make_pair(MakeConstOp<double>, ng::element::f64)},
      {DataType::DT_INT8, make_pair(MakeConstOp<int8_t>, ng::element::i8)},
      {DataType::DT_INT16, make_pair(MakeConstOp<int16_t>, ng::element::i16)},
#if 0
      {DataType::DT_QINT8, make_pair(MakeConstOp<qint8>, ng::element::i8)},
      {DataType::DT_QUINT8, make_pair(MakeConstOp<quint8>, ng::element::u8)},
      {DataType::DT_QUINT16, make_pair(MakeConstOp<quint16>, ng::element::u16)},
#endif
      {DataType::DT_INT32, make_pair(MakeConstOp<int32_t>, ng::element::i32)},
      {DataType::DT_INT64, make_pair(MakeConstOp<int64_t>, ng::element::i64)},
      {DataType::DT_UINT8, make_pair(MakeConstOp<uint8_t>, ng::element::u8)},
      {DataType::DT_UINT16, make_pair(MakeConstOp<uint16_t>, ng::element::u16)},
      {DataType::DT_BOOL,
       make_pair(MakeConstOp<bool, char>, ng::element::boolean)}};
  return the_map;
}

// Helper function to translate a unary op.
//
// Parameters:
//
//    TFNodeDecoder* op                   - TF op being translated. Must have one input.
//    const std::vector<const TensorWrapper*>& static_input_map
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
static Status TranslateUnaryOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>&,
    Builder::OpMap& ng_op_map,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>)> create_unary_op) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  auto ng_node = create_unary_op(ng_input);
  if (ng_node != ng_input) {
    Builder::SetTracingInfo(op->name(), ng_node);
  }
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
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
static Status TranslateUnaryOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(op, static_input_map, ng_op_map,
                          [&op](ng::Output<ng::Node> n) {
                            return ConstructNgNode<T>(op->name(), n);
                          });
}

// Helper function to translate a binary op
// Parameters:
//
//    TFNodeDecoder* op               - TF op being translated. Must have only two
//    inputs.
//    const std::vector<const TensorWrapper*>& static_input_map - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-nGraph op map.
//    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>,
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

static Status TranslateBinaryOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>&,
    Builder::OpMap& ng_op_map,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>&,
                                       ng::Output<ng::Node>&)>
        create_binary_op) {
  ng::Output<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs));
  auto ng_node = create_binary_op(ng_lhs, ng_rhs);
  if (ng_node != ng_lhs && ng_node != ng_rhs) {
    Builder::SetTracingInfo(op->name(), ng_node);
  }
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
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
static Status TranslateBinaryOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateBinaryOp(
      op, static_input_map, ng_op_map,
      [&op](ng::Output<ng::Node>& ng_lhs, ng::Output<ng::Node>& ng_rhs) {
        return ConstructNgNode<T>(op->name(), ng_lhs, ng_rhs);
      });
}

static Status TranslateAddNOp(const TFNodeDecoder* op, const std::vector<const TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
  std::vector<ng::Output<ng::Node>> ng_arg_vec(op->num_inputs());

  for (int inp_idx = 0; inp_idx < op->num_inputs(); inp_idx++)
    TF_RETURN_IF_ERROR(
        GetInputNode(ng_op_map, op, inp_idx, ng_arg_vec[inp_idx]));
  auto ng_addn = std::accumulate(
      std::next(ng_arg_vec.begin()), ng_arg_vec.end(), ng_arg_vec.at(0),
      [&op](ng::Output<ng::Node> a, ng::Output<ng::Node> b) {
        return ConstructNgNode<opset::Add>(op->name(), a, b);
      });  // accumulation: start with
           // first element. default op is
           // addition
  SaveNgOp(ng_op_map, op->name(), ng_addn);
  return Status::OK();
}
static Status TranslateArgMinMax(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map, std::string mode) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  std::vector<int64_t> tf_dim;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &tf_dim));

  ng::Shape input_shape = ng_input.get_shape();
  size_t input_rank = input_shape.size();

  if (tf_dim.size() != 1) {
    return errors::InvalidArgument(
        "ArgMax Op: dimension must be scalar, operates on a single axis");
  }

  // If input dimension is negative, make it positive
  if (tf_dim[0] < 0) {
    NGRAPH_VLOG(3) << "Input dimension is negative, make it positive "
                   << tf_dim[0];
    tf_dim[0] = (int64_t)input_rank + tf_dim[0];
  }
  NGRAPH_VLOG(3) << "Axis along which to compute " << tf_dim[0];
  size_t k_axis = tf_dim[0];

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "output_type", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  auto ng_k = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{}, std::vector<int64_t>({1}));

  std::string sort = "none";
  auto ng_topk =
      std::make_shared<opset::TopK>(ng_input, ng_k, k_axis, mode, sort, ng_et);
  auto ng_indices = ng_topk->output(1);
  int axis = ng_topk->get_axis();
  auto axis_to_remove = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{1}, std::vector<int64_t>({axis}));
  auto reshaped_indices =
      ConstructNgNode<opset::Squeeze>(op->name(), ng_indices, axis_to_remove);
  Builder::SetTracingInfo(op->name(), reshaped_indices);
  SaveNgOp(ng_op_map, op->name(), reshaped_indices);
  return Status::OK();
}

static Status TranslateArgMaxOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return (TranslateArgMinMax(op, static_input_map, ng_op_map, "max"));
}

static Status TranslateArgMinOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return (TranslateArgMinMax(op, static_input_map, ng_op_map, "min"));
}

static Status TranslateAvgPoolOp(const TFNodeDecoder* op,
                                 const std::vector<const TensorWrapper*>&,
                                 Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<int32_t> tf_strides;
  std::vector<int32_t> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "AvgPool data format is neither NHWC nor NCHW");
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
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff padding_below;
  ng::CoordinateDiff padding_above;
  ng::Shape ng_dilations{1, 1};
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, padding_below, padding_above);

  // TODO: remove this once nGraph supports negative padding
  // (CoordinateDiff) for AvgPool
  ng::Shape ng_padding_below(padding_below.begin(), padding_below.end());
  ng::Shape ng_padding_above(padding_above.begin(), padding_above.end());

  ng::Output<ng::Node> ng_avgpool = ConstructNgNode<opset::AvgPool>(
      op->name(), ng_input, ng_strides, ng_padding_below, ng_padding_above,
      ng_kernel_shape, true, ng::op::RoundingType::FLOOR);

  NCHWtoNHWC(op->name(), is_nhwc, ng_avgpool);
  NGRAPH_VLOG(3) << "avgpool outshape: {" << ng::join(ng_avgpool.get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_avgpool);
  return Status::OK();
}

static Status TranslateBiasAddOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_bias;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_bias));

  std::string tf_data_format;
  if (GetNodeAttr(op->attrs(), "data_format", &tf_data_format) !=
      Status::OK()) {
    tf_data_format = "NHWC";
  }

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "BiasAdd data format is neither NHWC nor NCHW");
  }

  auto ng_input_shape = ng_input.get_shape();
  auto ng_bias_shape = ng_bias.get_shape();
  if (ng_bias_shape.size() != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
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
    auto target_shape_node = make_shared<opset::Constant>(
        ng::element::i64, ng::Shape{ng_input_shape.size()}, target_shape);
    ng_bias_reshaped = ConstructNgNode<opset::Reshape>(
        op->name(), ng_bias, target_shape_node, false);
  }

  ng::Output<ng::Node> ng_add =
      ConstructNgNode<opset::Add>(op->name(), ng_input, ng_bias_reshaped);

  SaveNgOp(ng_op_map, op->name(), ng_add);
  return Status::OK();
}

static Status TranslateCastOp(const TFNodeDecoder* op, const std::vector<const TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "DstT", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  try {
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<opset::Convert>(op->name(), ng_input, ng_et));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Failed to convert TF data type: " +
                                 DataType_Name(dtype));
  }
  return Status::OK();
}

static Status TranslateConcatV2Op(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 2));

  std::vector<int64_t> tf_concat_axis_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(
          ng_op_map, op, op->num_inputs() - 1, static_input_map, &tf_concat_axis_vec));

  int64_t concat_axis = tf_concat_axis_vec[0];

  if (concat_axis < 0) {
    ng::Output<ng::Node> ng_first_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_first_arg));

    concat_axis += int64_t(ng_first_arg.get_shape().size());
  }

  ng::OutputVector ng_args;

  for (int i = 0; i < op->num_inputs() - 1; i++) {
    ng::Output<ng::Node> ng_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_arg));
    ng_args.push_back(ng_arg);
  }

  SaveNgOp(
      ng_op_map, op->name(),
      ConstructNgNode<opset::Concat>(op->name(), ng_args, size_t(concat_axis)));
  return Status::OK();
}

static Status TranslateConstOp(const TFNodeDecoder* op,
                               const std::vector<const TensorWrapper*>&,
                               Builder::OpMap& ng_op_map) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dtype", &dtype));

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
    const auto& func_param = Builder::TF_NGRAPH_CONST_MAP().at(dtype);
    TF_RETURN_IF_ERROR(func_param.first(op, func_param.second, ng_node));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Failed to translate Constant with TF type:" +
                                 DataType_Name(dtype));
  }

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateConv2DOp(const TFNodeDecoder* op,
                                const std::vector<const TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter));

  std::vector<int32_t> tf_strides;
  std::vector<int32_t> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  // TF Kernel Test Checks
  // Strides in the batch and depth dimension is not supported
  if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
    return errors::InvalidArgument(
        "Strides in batch and depth dimensions is not supported: " +
        op->type_string());
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
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below;
  ng::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  ng::Output<ng::Node> ng_conv = ConstructNgNode<opset::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_nhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateConv2DBackpropInputOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_filter, ng_out_backprop, ng_unused;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_unused, ng_filter, ng_out_backprop));

  // TODO: refactor me to be less redundant with other convolution ops
  std::vector<int32_t> tf_strides;
  std::vector<int32_t> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2DBackpropInput data format is neither NHWC nor NCHW: %s" +
        tf_data_format);
  }

  std::vector<int64_t> tf_input_sizes;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 0, static_input_map, &tf_input_sizes));

  if (std::any_of(tf_input_sizes.begin(), tf_input_sizes.end(),
                  [](int32_t size) { return size <= 0; })) {
    return errors::InvalidArgument(
        "Conv2DBackpropInput input sizes must be positive integers");
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
  NHWCtoNCHW(op->name(), is_nhwc, ng_out_backprop);
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
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below;
  ng::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  auto ng_output_shape = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{ng_batch_shape.size() - 2},
      vector<size_t>(ng_batch_shape.begin() + 2, ng_batch_shape.end()));

  auto ng_data = ConstructNgNode<opset::ConvolutionBackpropData>(
      op->name(), ng_out_backprop, ng_filter, ng_output_shape, ng_strides,
      ng_padding_below, ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_nhwc, ng_data);
  SaveNgOp(ng_op_map, op->name(), ng_data);
  return Status::OK();
}

// Translate Conv3D Op
static Status TranslateConv3DOp(const TFNodeDecoder* op,
                                const std::vector<const TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter));

  std::vector<int32_t> tf_strides;
  std::vector<int32_t> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NDHWC" && tf_data_format != "NCDHW") {
    return errors::InvalidArgument(
        "Conv3D data format is neither NDHWC nor NCDHW");
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
  NHWCtoNCHW(op->name(), is_ndhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  ng_kernel_shape[2] = ng_filter_shape[2];
  Transpose3D<4, 3, 0, 1, 2>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below;
  ng::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  ng::Output<ng::Node> ng_conv = ConstructNgNode<opset::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_ndhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateCumsumOp(const TFNodeDecoder* op,
                                const std::vector<const TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_x, ng_axis;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_axis));
  bool exclusive, reverse;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "exclusive", &exclusive));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "reverse", &reverse));

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::CumSum>(op->name(), ng_x, ng_axis, exclusive,
                                          reverse));
  return Status::OK();
}

// Translate DepthToSpace op
static Status TranslateDepthToSpaceOp(const TFNodeDecoder* op,
                                      const std::vector<const TensorWrapper*>&,
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

  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  auto ng_mode = opset::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
  ng::Output<ng::Node> depth_to_space = ConstructNgNode<opset::DepthToSpace>(
      op->name(), ng_input, ng_mode, block_size);
  NCHWtoNHWC(op->name(), is_nhwc, depth_to_space);
  SaveNgOp(ng_op_map, op->name(), depth_to_space);
  return Status::OK();
}

static Status TranslateDepthwiseConv2dNativeOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>&,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter));

  std::vector<int32_t> tf_strides;
  std::vector<int32_t> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "DepthwiseConv2D data format is neither NHWC nor NCHW");
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
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below;
  ng::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  // H W I M -> H W I 1 M
  auto filter_shape = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::u64, ng::Shape{5},
      ngraph::Shape{ng_filter_shape[0], ng_filter_shape[1], ng_filter_shape[2],
                    1, ng_filter_shape[3]});
  auto reshaped_filter = ConstructNgNode<opset::Reshape>(op->name(), ng_filter,
                                                         filter_shape, false);

  // H W I 1 M -> I M 1 H W
  auto order = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{5}, vector<int64_t>{2, 4, 3, 0, 1});
  auto transposed_filter =
      ConstructNgNode<opset::Transpose>(op->name(), reshaped_filter, order);

  auto ng_conv = ConstructNgNode<opset::GroupConvolution>(
      op->name(), ng_input, transposed_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_nhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateExpandDimsOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &dims));
  auto ng_dims = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ngraph::Shape{dims.size()}, dims);
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Unsqueeze>(op->name(), ng_input, ng_dims));
  return Status::OK();
}

static Status TranslateFillOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_value, ng_dims;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_dims, ng_value));
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Broadcast>(op->name(), ng_value, ng_dims));
  return Status::OK();
}

static Status TranslateFloorDivOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  auto floordiv_fn = [&op](ng::Output<ng::Node> x, ng::Output<ng::Node> y) {
    return ConstructNgNode<opset::Floor>(
        op->name(), ConstructNgNode<opset::Divide>(op->name(), x, y));
  };
  return TranslateBinaryOp(op, static_input_map, ng_op_map, floordiv_fn);
}

static Status TranslateFusedBatchNormOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_scale, ng_offset, ng_mean, ng_variance;
  bool is_v3 = op->type_string() == "FusedBatchNormV3";

  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_scale, ng_offset,
                                   ng_mean, ng_variance));

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << "data_format: " << tf_data_format;

  float tf_epsilon;
  if (GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon) != Status::OK()) {
    NGRAPH_VLOG(3) << "epsilon attribute not present, setting to 0.0001";
    // TensorFlow default
    tf_epsilon = 0.0001;
  }

  NGRAPH_VLOG(3) << "epsilon: " << tf_epsilon;

  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  auto ng_batch_norm = ConstructNgNode<opset::BatchNormInference>(
      op->name(), ng_input, ng_scale, ng_offset, ng_mean, ng_variance,
      tf_epsilon);
  NCHWtoNHWC(op->name(), is_nhwc, ng_batch_norm);
  SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
  SaveNgOp(ng_op_map, op->name(), ng_mean);
  SaveNgOp(ng_op_map, op->name(), ng_variance);
  SaveNgOp(ng_op_map, op->name(), ng_mean);      // reserve_space_1
  SaveNgOp(ng_op_map, op->name(), ng_variance);  // reserve_space_2
  if (is_v3) {
    // FusedBatchNormV3 has 6 outputs
    SaveNgOp(ng_op_map, op->name(), ng_mean);  // reserve_space_3
  }
  return Status::OK();
}

static Status TranslateFusedMatMulOp(const TFNodeDecoder* op,
                                     const std::vector<const TensorWrapper*>&,
                                     Builder::OpMap& ng_op_map) {
  int num_args;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_args", &num_args));

  std::vector<string> fused_ops;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "fused_ops", &fused_ops));

  // Transpose arguments if requested.
  bool transpose_a = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_a", &transpose_a));

  bool transpose_b = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_b", &transpose_b));

  ng::Output<ng::Node> ng_lhs, ng_rhs, ng_bias, ng_matmul;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs, ng_bias));
  ng_matmul = ConstructNgNode<opset::MatMul>(op->name(), ng_lhs, ng_rhs,
                                             transpose_a, transpose_b);

  auto ng_matmul_shape = ng_matmul.get_shape();
  auto ng_bias_shape = ng_bias.get_shape();

  if (ng_bias_shape.size() != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
  }

  auto ng_add = ConstructNgNode<opset::Add>(op->name(), ng_matmul, ng_bias);
  if (fused_ops.size() == 1) {  // Only fusing BiasAdd
    SaveNgOp(ng_op_map, op->name(), ng_add);
  } else if (fused_ops.size() == 2) {  // Also has activation
    if (fused_ops[1] == "Relu") {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<opset::Relu>(op->name(), ng_add));
    } else if (fused_ops[1] == "Relu6") {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<opset::Clamp>(op->name(), ng_add, 0, 6));
    } else {
      return errors::Internal(
          "Expected activation to be Relu or Relu6 but got " + fused_ops[1]);
    }
  } else {
    // Adding this here to catch future changes in _FusedMatMul
    return errors::Internal("Unsupported combination");
  }

  return Status::OK();
}

// See .../tensorflow/include/tensorflow/cc/ops/array_ops.h
// and .../openvino/ngraph/core/include/ngraph/op/gather.hpp
static Status TranslateGatherOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_input_indices;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_input_indices));

  auto ng_axis = ConstructNgNode<opset::Constant>(op->name(), ng::element::i64,
                                                  ng::Shape{}, 0);

  auto gather_op = ConstructNgNode<opset::Gather>(op->name(), ng_input,
                                                  ng_input_indices, ng_axis);

  SaveNgOp(ng_op_map, op->name(), gather_op);
  return Status::OK();
}

static Status TranslateGatherV2Op(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_input_coords, ng_unused;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input, ng_input_coords, ng_unused));

  std::vector<int64_t> tf_axis;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 2, static_input_map, &tf_axis));

  if (tf_axis.size() > 1) {
      std::ostringstream buf;
      buf << "Found axis in GatherV2 op (" << op->name() <<
              ") translation to be non scalar, of size " <<
              tf_axis.size();
    return errors::Internal(buf.str());
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
      std:ostringstream buf;
      buf << "Expected axis in the range [-" <<
              ng_input_rank << ", " << ng_input_rank <<
              "), but got " << tf_axis[0];
    return errors::InvalidArgument(buf.str());
  }

  auto ng_axis = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{tf_axis.size()}, tf_axis);

  auto gather_op = ConstructNgNode<opset::Gather>(op->name(), ng_input,
                                                  ng_input_coords, ng_axis);

  SaveNgOp(ng_op_map, op->name(), gather_op);
  return Status::OK();
}

static Status TranslateFusedConv2DOp(const TFNodeDecoder* op,
                                     const std::vector<const TensorWrapper*>&,
                                     Builder::OpMap& ng_op_map) {
  int num_args;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_args", &num_args));

  std::vector<string> fused_ops;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "fused_ops", &fused_ops));

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));
  bool is_nhwc = (tf_data_format == "NHWC");

  auto CreateNgConv = [&](ng::Output<ng::Node>& ng_input,
                          ng::Output<ng::Node>& ng_filter,
                          ng::Output<ng::Node>& ng_conv) {
    std::vector<int32_t> tf_strides;
    std::vector<int32_t> tf_dilations;
    std::string tf_padding_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
      return errors::InvalidArgument(
          "Conv2D data format is neither NHWC nor NCHW");
    }

    // TF Kernel Test Checks
    // Strides in the batch and depth dimension is not supported
    if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
      return errors::InvalidArgument(
          "Strides in batch and depth dimensions is not supported: " +
          op->type_string());
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
    NHWCtoNCHW(op->name(), is_nhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    Transpose<3, 2, 0, 1>(ng_filter);
    Builder::SetTracingInfo(op->name(), ng_filter);

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff ng_padding_below;
    ng::CoordinateDiff ng_padding_above;
    Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_dilations, ng_padding_below,
                         ng_padding_above);

    ng_conv = ConstructNgNode<opset::Convolution>(
        op->name() + "_FusedConv2D_Conv", ng_input, ng_filter, ng_strides,
        ng_padding_below, ng_padding_above, ng_dilations);

    return Status::OK();
  };

  if (VecStrCmp(fused_ops, {"BiasAdd"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
    if (num_args != 1) {
      return errors::InvalidArgument(
          "FusedConv2DBiasAdd has incompatible num_args");
    }

    ng::Output<ng::Node> ng_input, ng_filter, ng_bias, ng_conv;
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, ng_input, ng_filter, ng_bias));

    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    auto ng_conv_shape = ng_conv.get_shape();
    auto ng_bias_shape = ng_bias.get_shape();
    if (ng_bias_shape.size() != 1) {
      return errors::InvalidArgument(
          "Bias argument to BiasAdd does not have one dimension");
    }

    std::vector<size_t> reshape_pattern_values(ng_conv_shape.size(), 1U);
    reshape_pattern_values[1] = ng_bias.get_shape().front();
    auto reshape_pattern = make_shared<opset::Constant>(
        ng::element::u64, ng::Shape{reshape_pattern_values.size()},
        reshape_pattern_values);
    auto ng_bias_reshaped = ConstructNgNode<opset::Reshape>(
        op->name(), ng_bias, reshape_pattern, false);

    auto ng_add = ConstructNgNode<opset::Add>(
        op->name() + "_FusedConv2D_BiasAdd", ng_conv, ng_bias_reshaped);

    if (VecStrCmp(fused_ops, {"BiasAdd", "Relu"})) {
      auto ng_relu = ConstructNgNode<opset::Relu>(
          op->name() + "_FusedConv2D_Relu", ng_add);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu);
      SaveNgOp(ng_op_map, op->name(), ng_relu);
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
      auto ng_relu6 = ConstructNgNode<opset::Clamp>(
          op->name() + "_FusedConv2D_Relu6", ng_add, 0, 6);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu6);
      SaveNgOp(ng_op_map, op->name(), ng_relu6);
    } else {
      NCHWtoNHWC(op->name(), is_nhwc, ng_add);
      SaveNgOp(ng_op_map, op->name(), ng_add);
    }
  } else if (VecStrCmp(fused_ops, {"FusedBatchNorm"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
    if (num_args != 4) {
      return errors::InvalidArgument(
          "FusedConv2D with FusedBatchNorm has incompatible num_args");
    }

    ng::Output<ng::Node> ng_input, ng_filter, ng_conv, ng_scale, ng_offset,
        ng_mean, ng_variance;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter,
                                     ng_scale, ng_offset, ng_mean,
                                     ng_variance));
    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    float tf_epsilon;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon));

    auto ng_batch_norm = ConstructNgNode<opset::BatchNormInference>(
        op->name() + "_FusedConv2D_BatchNorm", ng_conv, ng_scale, ng_offset,
        ng_mean, ng_variance, tf_epsilon);

    if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"})) {
      auto ng_relu = ConstructNgNode<opset::Relu>(
          op->name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu);
      SaveNgOp(ng_op_map, op->name(), ng_relu);
    } else if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
      auto ng_relu6 = ConstructNgNode<opset::Clamp>(
          op->name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm, 0, 6);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu6);
      SaveNgOp(ng_op_map, op->name(), ng_relu6);
    } else {
      NCHWtoNHWC(op->name(), is_nhwc, ng_batch_norm);
      SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
    }
  } else {
    return errors::Unimplemented("Unsupported _FusedConv2D " +
                                 StrJoin(fused_ops, ","));
  }
  return Status::OK();
}

static Status TranslateIdentityOp(const TFNodeDecoder* op,
                                  const std::vector<const TensorWrapper*>&,
                                  Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_arg;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_arg));
  SaveNgOp(ng_op_map, op->name(), ng_arg);
  return Status::OK();
}

static Status TranslateIsFiniteOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // Implemented tf.is_finite by checking:
  // (in != inf) && (in != -inf) && (in == in)
  //                                 ^^^^^^^^ checks for NaN's
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto const_inf = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ng::Shape{},
      std::vector<float>{std::numeric_limits<float>::infinity()});

  auto const_neg_inf = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ng::Shape{},
      std::vector<float>{-std::numeric_limits<float>::infinity()});

  auto neq_inf =
      ConstructNgNode<opset::NotEqual>(op->name(), ng_input, const_inf);
  auto neq_neg_inf =
      ConstructNgNode<opset::NotEqual>(op->name(), ng_input, const_neg_inf);
  auto eq_nan = ConstructNgNode<opset::Equal>(op->name(), ng_input, ng_input);

  auto neq_inf_and_neq_neg_inf =
      ConstructNgNode<opset::LogicalAnd>(op->name(), neq_inf, neq_neg_inf);
  auto is_finite = ConstructNgNode<opset::LogicalAnd>(
      op->name(), neq_inf_and_neq_neg_inf, eq_nan);

  SaveNgOp(ng_op_map, op->name(), is_finite);
  return Status::OK();
}

static Status TranslateL2LossOp(const TFNodeDecoder* op,
                                const std::vector<const TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<float> val;
  val.push_back(2.0);
  auto const_2 = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ng::Shape{}, val[0]);

  auto ng_pow =
      ConstructNgNode<opset::Multiply>(op->name(), ng_input, ng_input);

  size_t input_rank = ng_input.get_shape().size();
  std::vector<int64_t> axes;
  for (size_t i = 0; i < input_rank; ++i) {
    axes.push_back(i);
  }

  auto ng_reduction_axes = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{axes.size()}, axes);
  auto ng_sum =
      ConstructNgNode<opset::ReduceSum>(op->name(), ng_pow, ng_reduction_axes);
  auto ng_l2loss = ConstructNgNode<opset::Divide>(op->name(), ng_sum, const_2);
  SaveNgOp(ng_op_map, op->name(), ng_l2loss);
  return Status::OK();
}

static Status TranslateLog1pOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> val_1(ng::shape_size(shape), "1");
        auto ng_const1 =
            ConstructNgNode<opset::Constant>(op->name(), et, shape, val_1);
        auto ng_add = ConstructNgNode<opset::Add>(op->name(), ng_const1, n);
        return ConstructNgNode<opset::Log>(op->name(), ng_add);
      });
}

static Status TranslateLRNOp(const TFNodeDecoder* op,
                             const std::vector<const TensorWrapper*>& static_input_map,
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
  NHWCtoNCHW(op->name(), true, ng_inp);
  auto ng_output = ConstructNgNode<opset::LRN>(op->name(), ng_inp, alpha, beta,
                                               bias, (size_t)size);
  NCHWtoNHWC(op->name(), true, ng_output);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateLogSoftmaxOp(const TFNodeDecoder* op,
                                    const std::vector<const TensorWrapper*>&,
                                    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp));
  auto inp_shape = ng_inp.get_shape();
  size_t rank = inp_shape.size();
  int64_t axes = rank - 1;

  auto ng_output = ConstructNgNode<opset::LogSoftmax>(op->name(), ng_inp, axes);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateMatMulOp(const TFNodeDecoder* op,
                                const std::vector<const TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs));

  // Transpose arguments if requested.
  bool transpose_a = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_a", &transpose_a));

  bool transpose_b = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_b", &transpose_b));

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::MatMul>(op->name(), ng_lhs, ng_rhs,
                                          transpose_a, transpose_b));
  return Status::OK();
}

template <unsigned int N>
static Status TranslateMaxPoolOp(const TFNodeDecoder* op,
                                 const std::vector<const TensorWrapper*>&,
                                 Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<int32_t> tf_strides;
  std::vector<int32_t> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

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
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff padding_below;
  ng::CoordinateDiff padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, padding_below, padding_above);

  // TODO: remove this once nGraph supports negative padding
  // (CoordinateDiff) for MaxPool
  ng::Shape ng_padding_below(padding_below.begin(), padding_below.end());
  ng::Shape ng_padding_above(padding_above.begin(), padding_above.end());

  auto ng_maxpool = ConstructNgNode<opset::MaxPool>(
      op->name(), ng_input, ng_strides, ng_padding_below, ng_padding_above,
      ng_kernel_shape, ng::op::RoundingType::FLOOR);

  NCHWtoNHWC(op->name(), is_nhwc, ng_maxpool);

  NGRAPH_VLOG(3) << "maxpool outshape: {" << ng::join(ng_maxpool.get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_maxpool);
  return Status::OK();
}

static Status TranslateNonMaxSuppressionV2Op(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_boxes, ng_scores, ng_unused, ng_iou_threshold;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_boxes, ng_scores,
                                   ng_unused, ng_iou_threshold));

  auto ng_axis_boxes = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{1}, std::vector<int64_t>({0}));
  auto ng_boxes_unsqueezed =
      ConstructNgNode<opset::Unsqueeze>(op->name(), ng_boxes, ng_axis_boxes);

  auto ng_axis_scores = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{1}, std::vector<int64_t>({0}));
  auto ng_scores_unsqueezed1 =
      ConstructNgNode<opset::Unsqueeze>(op->name(), ng_scores, ng_axis_scores);
  auto ng_scores_unsqueezed2 = ConstructNgNode<opset::Unsqueeze>(
      op->name(), ng_scores_unsqueezed1, ng_axis_scores);

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
      op->name(), ng::element::i64, ng::Shape{}, max_output_size[0]);
  NGRAPH_VLOG(5) << "ng_max_output_size " << max_output_size[0];

  auto ng_nmsv = ConstructNgNode<opset::NonMaxSuppression>(
      op->name(), ng_boxes_unsqueezed, ng_scores_unsqueezed2,
      ng_max_output_size, ng_iou_threshold,
      opset::NonMaxSuppression::BoxEncodingType::CORNER, false,
      ngraph::element::Type_t::i32);

  auto begin = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{2}, std::vector<int64_t>({0, 2}));
  auto end = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{2},
      std::vector<int64_t>({max_output_size[0], 3}));
  auto ng_nmsv_slice = ConstructNgNode<opset::StridedSlice>(
      op->name(), ng_nmsv, begin, end, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{0, 1});

  Builder::SetTracingInfo(op->name(), ng_nmsv_slice);
  SaveNgOp(ng_op_map, op->name(), ng_nmsv_slice);
  return Status::OK();
}

static Status TranslateReduceOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>,
                                       ng::Output<ng::Node>, const bool)>
        create_ng_node) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64_t> axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &axes));

  ng::Shape input_shape = ng_input.get_shape();
  size_t input_rank = input_shape.size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(axes.size());
  std::transform(
      axes.begin(), axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  auto ng_reduction_axes = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{ng_reduction_axes_vect.size()},
      ng_reduction_axes_vect);

  ng::Output<ng::Node> ng_node =
      create_ng_node(ng_input, ng_reduction_axes, tf_keep_dims);

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

template <typename T>
static Status TranslateDirectReduceOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // ensure its either an arithmetic or a logical reduction
  if (!(std::is_base_of<ngraph::op::util::ArithmeticReduction, T>::value ||
        std::is_base_of<ngraph::op::util::LogicalReduction, T>::value)) {
    return errors::InvalidArgument(
        "Expected node to be either a valid logical or arithmetic reduction "
        "type");
  }
  return TranslateReduceOp(
      op, static_input_map, ng_op_map,
      [&op](ng::Output<ng::Node> ng_input,
            ng::Output<ng::Node> ng_reduction_axes, const bool keep_dims) {
        return ConstructNgNode<T>(op->name(), ng_input, ng_reduction_axes,
                                  keep_dims);
      });
}

static Status TranslateOneHotOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
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
      op->name(), ng::element::i64, ng::Shape{}, depth);

  int one_hot_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &one_hot_axis));

  auto ng_onehot = ConstructNgNode<opset::OneHot>(
      op->name(), ng_features, const_depth, ng_on, ng_off, one_hot_axis);
  SaveNgOp(ng_op_map, op->name(), ng_onehot);
  return Status::OK();
}

static Status TranslatePackOp(const TFNodeDecoder* op, const std::vector<const TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

  int32_t tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  auto ng_axis = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{1},
      std::vector<int64_t>({tf_axis}));

  ng::OutputVector ng_concat_inputs;
  for (int32_t i = 0; i < op->num_inputs(); ++i) {
    ng::Output<ng::Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_input));
    auto unsqueezed_input =
        ConstructNgNode<opset::Unsqueeze>(op->name(), ng_input, ng_axis);
    ng_concat_inputs.push_back(unsqueezed_input);
  }

  // if inputs shape is (2, 3, 4), and axis is 1, then we want
  // to create output_shape (2, num_inputs, 3, 4)
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Concat>(
                                      op->name(), ng_concat_inputs, tf_axis));
  return Status::OK();
}

// 3 different Pad Ops: Pad, PadV2, MirrorPad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad-v2
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mirror-pad
static Status TranslatePadOp(const TFNodeDecoder* op,
                             const std::vector<const TensorWrapper*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_paddings_op, pad_val_op, result_pad_op;

  // Set inputs and pad_val_op
  if (op->type_string() == "Pad" || op->type_string() == "MirrorPad") {
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_paddings_op));
    pad_val_op = ConstructNgNode<opset::Constant>(
        op->name(), ng_input.get_element_type(), ng::Shape(),
        std::vector<int>({0}));
  } else if (op->type_string() == "PadV2") {
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, ng_input, ng_paddings_op, pad_val_op));
  } else {
    return errors::InvalidArgument("Incorrect TF Pad OpType: " +
                                   op->type_string());
  }

  // Set pad_mode
  auto pad_mode = ng::op::PadMode::CONSTANT;
  if (op->type_string() == "MirrorPad") {
    std::string pad_mode_str;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "mode", &pad_mode_str));
    if (pad_mode_str == "REFLECT") {
      pad_mode = ng::op::PadMode::REFLECT;
    } else if (pad_mode_str == "SYMMETRIC") {
      pad_mode = ng::op::PadMode::SYMMETRIC;
    } else {
      return errors::InvalidArgument(pad_mode_str +
                                     " is not an allowed padding mode.");
    }
  }

  // Set pads_begin & pads_end (from the pad_val_op)
  std::vector<int64_t> paddings;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &paddings));
  NGRAPH_VLOG(3) << op->name() << " pads {" << ng::join(paddings) << "}";
  if (paddings.size() % 2 != 0) {
    return errors::InvalidArgument(
        "Constant node for paddings does not have an even number of "
        "elements");
  }
  std::vector<int64_t> pad_begin(paddings.size() / 2);
  std::vector<int64_t> pad_end(paddings.size() / 2);
  for (size_t i = 0; i < paddings.size() / 2; i++) {
    pad_begin[i] = paddings[2 * i];
    pad_end[i] = paddings[2 * i + 1];
  }
  auto pads_begin_node = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{pad_begin.size()}, pad_begin);
  auto pads_end_node = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{pad_end.size()}, pad_end);

  // Create final Op
  result_pad_op =
      ConstructNgNode<opset::Pad>(op->name(), ng_input, pads_begin_node,
                                  pads_end_node, pad_val_op, pad_mode);

  SaveNgOp(ng_op_map, op->name(), result_pad_op);
  return Status::OK();
}

static Status TranslateRangeOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
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
  auto ng_range = ConstructNgNode<opset::Range>(op->name(), ng_start,
                                                ng_stop, ng_step, out_type);

  SaveNgOp(ng_op_map, op->name(), ng_range);
  return Status::OK();
}

static Status TranslateRankOp(const TFNodeDecoder* op, const std::vector<const TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  ng::Shape input_shape = ng_input.get_shape();
  auto input_rank = static_cast<int>(input_shape.size());

  auto ng_rank = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i32, ng::Shape(),
      std::vector<int>({input_rank}));

  SaveNgOp(ng_op_map, op->name(), ng_rank);
  return Status::OK();
}

static Status TranslateReciprocalOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        // Create a constant tensor populated with the value -1.
        // (1/x = x^(-1))
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-1");
        auto ng_exponent = ConstructNgNode<opset::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -1.
        return ConstructNgNode<opset::Power>(op->name(), n, ng_exponent);
      });
}

static Status TranslateRelu6Op(const TFNodeDecoder* op,
                               const std::vector<const TensorWrapper*>&,
                               Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Clamp>(op->name(), ng_input, 0, 6));
  return Status::OK();
}

static Status TranslateReshapeOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_shape_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_shape_op));

  NGRAPH_VLOG(3) << "Input shape: " << ng::join(ng_input.get_shape());

  std::vector<int64_t> shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &shape));

  NGRAPH_VLOG(3) << "Requested result shape: " << ng::join(shape);

  auto ng_shape = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{shape.size()}, shape);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Reshape>(
                                      op->name(), ng_input, ng_shape, false));
  return Status::OK();
}

static Status TranslateRsqrtOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        // Create a constant tensor populated with the value -1/2.
        // (1/sqrt(x) = x^(-1/2))
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-0.5");
        auto ng_exponent = ConstructNgNode<opset::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -0.5.
        return ConstructNgNode<opset::Power>(op->name(), n, ng_exponent);
      });
}

static Status TranslateShapeOp(const TFNodeDecoder* op,
                               const std::vector<const TensorWrapper*>&,
                               Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  // default output_type = element::i64
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::ShapeOf>(op->name(), ng_input, type));
  return Status::OK();
}

static Status TranslateSizeOp(const TFNodeDecoder* op, const std::vector<const TensorWrapper*>&,
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
      op->name(), type, ng::Shape(0), std::vector<int64_t>({result}));

  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

static Status TranslateSliceOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
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
      op->name(), ng::element::i64, ng::Shape{begin_vec.size()}, begin_vec);
  auto end = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{end_vec.size()}, end_vec);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::StridedSlice>(op->name(), ng_input, begin,
                                                end, std::vector<int64_t>{},
                                                std::vector<int64_t>{}));
  return Status::OK();
}

static Status TranslateSoftmaxOp(const TFNodeDecoder* op,
                                 const std::vector<const TensorWrapper*>&,
                                 Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto input_shape = ng_input.get_shape();
  auto rank = input_shape.size();
  if (rank < 1) {
    return errors::InvalidArgument("TF Softmax logits must be >=1 dimension");
  }

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Softmax>(op->name(), ng_input, rank - 1));
  return Status::OK();
}

// Translate SpaceToDepthOp
static Status TranslateSpaceToDepthOp(const TFNodeDecoder* op,
                                      const std::vector<const TensorWrapper*>&,
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

  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  auto ng_mode = opset::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
  auto space_to_depth = ConstructNgNode<opset::SpaceToDepth>(
      op->name(), ng_input, ng_mode, block_size);
  NCHWtoNHWC(op->name(), is_nhwc, space_to_depth);
  SaveNgOp(ng_op_map, op->name(), space_to_depth);
  return Status::OK();
}

static Status TranslateSplitOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
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
      op->name(), ng::element::u64, ng::Shape{}, split_dim);
  auto ng_split = make_shared<opset::Split>(ng_input, ng_split_dim, num_split);

  for (int i = 0; i < num_split; ++i) {
    auto out = ng_split->output(i);
    Builder::SetTracingInfo(op->name(), out);
    SaveNgOp(ng_op_map, op->name(), out);
  }
  return Status::OK();
}

static Status TranslateSplitVOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
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
  ng_split_dim = ConstructNgNode<opset::Constant>(op->name(), ng::element::i32,
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
      op->name(), ng::element::i32, ng::Shape{split_lengths_vec.size()},
      split_lengths_vec);

  if (split_lengths_vec.size() != 1) {
    auto ng_split = make_shared<opset::VariadicSplit>(ng_input, ng_split_dim,
                                                      ng_split_length);
    for (size_t i = 0; i < split_lengths_vec.size(); ++i) {
      auto out = ng_split->output(i);
      Builder::SetTracingInfo(op->name(), out);
      SaveNgOp(ng_op_map, op->name(), out);
    }
  } else {
    SaveNgOp(ng_op_map, op->name(), ng_input);
  }

  return Status::OK();
}

static Status TranslateSquareOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        return ConstructNgNode<opset::Multiply>(op->name(), n, n);
      });
}

static Status TranslateSqueezeOp(const TFNodeDecoder* op,
                                 const std::vector<const TensorWrapper*>&,
                                 Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  size_t input_dims = ng_input.get_shape().size();

  std::vector<int32_t> tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "squeeze_dims", &tf_axis));

  // If input dimension is negative, make it positive
  for (size_t i = 0; i < tf_axis.size(); i++) {
    tf_axis[i] = tf_axis[i] < 0 ? (int32_t)(input_dims) + tf_axis[i] : tf_axis[i];
  }

  auto ng_const = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i32, ng::Shape{tf_axis.size()}, tf_axis);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Squeeze>(op->name(), ng_input, ng_const));
  return Status::OK();
}

static Status TranslateStridedSliceOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
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
      op->name(), ng::element::i64, ng::Shape{begin_vec.size()}, begin_vec);
  auto end = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{end_vec.size()}, end_vec);
  auto strides = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{stride_vec.size()}, stride_vec);

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
      ng_op_map, op->name(),
      ConstructNgNode<opset::StridedSlice>(
          op->name(), ng_input, begin, end, strides, mask_to_vec(begin_mask),
          mask_to_vec(end_mask), mask_to_vec(new_axis_mask),
          mask_to_vec(shrink_axis_mask), mask_to_vec(ellipsis_mask)));
  return Status::OK();
}

static Status TranslateTileOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_multiples;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_multiples));

  std::vector<int64_t> multiples;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &multiples));

  auto ng_repeats = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{multiples.size()}, multiples);
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Tile>(op->name(), ng_input, ng_repeats));
  return Status::OK();
}

// Translate TopKV2 Op using ngraph core op TopK
static Status TranslateTopKV2Op(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ngraph::Node> ng_input;

  TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  // axis along which to compute top k indices
  int64_t k_axis = ng_input.get_shape().size() - 1;

  // scalar input tensor specifying how many max/min elts should be computed
  // CPU backend only supports element type i64
  std::vector<int64_t> ng_k_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &ng_k_vec));
  auto ng_k = ConstructNgNode<opset::Constant>(op->name(), ng::element::i64,
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
  Builder::SetTracingInfo(op->name(), ng_values);
  ng::Output<ng::Node> ng_indices = ng_result->output(1);
  Builder::SetTracingInfo(op->name(), ng_indices);

  SaveNgOp(ng_op_map, op->name(), ng_values);
  SaveNgOp(ng_op_map, op->name(), ng_indices);

  return Status::OK();
}

static Status TranslateTransposeOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_permutation;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_permutation));
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Transpose>(
                                      op->name(), ng_input, ng_permutation));
  return Status::OK();
}

static Status TranslateUnpackOp(const TFNodeDecoder* op,
                                const std::vector<const TensorWrapper*>&,
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
        op->name(), ng::element::i64, ng::Shape{begin.size()}, begin);
    auto ng_end = ConstructNgNode<opset::Constant>(op->name(), ng::element::i64,
                                                   ng::Shape{end.size()}, end);
    std::vector<int64_t> begin_mask(rank, 1);
    begin_mask[tf_axis] = 0;
    std::vector<int64_t> end_mask(rank, 1);
    end_mask[tf_axis] = 0;
    std::vector<int64_t> new_axis_mask(rank, 0);
    std::vector<int64_t> shrink_axis_mask(rank, 0);
    shrink_axis_mask[tf_axis] = 1;
    auto slice = ConstructNgNode<opset::StridedSlice>(
        op->name(), ng_input, ng_begin, ng_end, begin_mask, end_mask,
        new_axis_mask, shrink_axis_mask);
    SaveNgOp(ng_op_map, op->name(), slice);
  }
  return Status::OK();
}

static Status TranslateXdivyOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ngraph::Node> ng_x, ng_y;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_y));
  auto zero =
      ConstructNgNode<opset::Constant>(op->name(), ng_x.get_element_type(),
                                       ngraph::Shape{}, std::vector<int>({0}));
  auto x_is_zero = ConstructNgNode<opset::Equal>(op->name(), ng_x, zero);
  auto ng_xdivy = ConstructNgNode<opset::Divide>(op->name(), ng_x, ng_y);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Select>(
                                      op->name(), x_is_zero, ng_x, ng_xdivy));
  return Status::OK();
}

static Status TranslateSelectOp(const TFNodeDecoder* op,
                                const std::vector<const TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input1, ng_input2, ng_input3;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input1, ng_input2, ng_input3));
  auto ng_select = ConstructNgNode<opset::Select>(op->name(), ng_input1,
                                                  ng_input2, ng_input3);
  SaveNgOp(ng_op_map, op->name(), ng_select);
  return Status::OK();
}

static Status TranslateWhereOp(
    const TFNodeDecoder* op, const std::vector<const TensorWrapper*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_cond;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_cond));
  auto non_zero = ConstructNgNode<opset::NonZero>(op->name(), ng_cond);
  auto transpose_order = ConstructNgNode<opset::Constant>(
      op->name(), ngraph::element::i64, ngraph::Shape{2},
      std::vector<int64_t>({1, 0}));
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Transpose>(
                                      op->name(), non_zero, transpose_order));
  return Status::OK();
}

static Status TranslateZerosLikeOp(const TFNodeDecoder* op,
                                   const std::vector<const TensorWrapper*>&,
                                   Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  ng::Shape input_shape = ng_input.get_shape();
  std::vector<std::string> const_values(ng::shape_size(input_shape), "0");
  auto ng_result = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), input_shape, const_values);
  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

const static std::map<
    const string,
    const function<Status(const TFNodeDecoder*, const std::vector<const TensorWrapper*>&,
                          Builder::OpMap&)>>
    TRANSLATE_OP_MAP{
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
        {"IsFinite", TranslateIsFiniteOp},
        {"L2Loss", TranslateL2LossOp},
        {"LogSoftmax", TranslateLogSoftmaxOp},
        {"Less", TranslateBinaryOp<opset::Less>},
        {"LessEqual", TranslateBinaryOp<opset::LessEqual>},
        {"Log", TranslateUnaryOp<opset::Log>},
        {"Log1p", TranslateLog1pOp},
        {"LogicalAnd", TranslateBinaryOp<opset::LogicalAnd>},
        {"LogicalNot", TranslateUnaryOp<opset::LogicalNot>},
        {"LogicalOr", TranslateBinaryOp<opset::LogicalOr>},
        {"LRN", TranslateLRNOp},
        {"MatMul", TranslateMatMulOp},
        {"Max", TranslateDirectReduceOp<opset::ReduceMax>},
        {"Maximum", TranslateBinaryOp<opset::Maximum>},
        {"MaxPool", TranslateMaxPoolOp<2>},
        {"MaxPool3D", TranslateMaxPoolOp<3>},
        {"NonMaxSuppressionV2", TranslateNonMaxSuppressionV2Op},
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
        {"NoOp", [](const TFNodeDecoder*, const std::vector<const TensorWrapper*>&,
                    Builder::OpMap&) { return Status::OK(); }},
        {"OneHot", TranslateOneHotOp},
        {"Pack", TranslatePackOp},
        {"Pad", TranslatePadOp},
        {"PadV2", TranslatePadOp},
        {"Pow", TranslateBinaryOp<opset::Power>},
        // PreventGradient is just Identity in dataflow terms, so reuse that.
        {"PreventGradient", TranslateIdentityOp},
        {"Prod", TranslateDirectReduceOp<opset::ReduceProd>},
        {"Range", TranslateRangeOp},
        {"Rank", TranslateRankOp},
        {"RealDiv", TranslateBinaryOp<opset::Divide>},
        {"Reciprocal", TranslateReciprocalOp},
        {"Relu", TranslateUnaryOp<opset::Relu>},
        {"Relu6", TranslateRelu6Op},
        {"Reshape", TranslateReshapeOp},
        {"Rsqrt", TranslateRsqrtOp},
        {"Select", TranslateSelectOp},
        {"SelectV2", TranslateSelectOp},
        {"Shape", TranslateShapeOp},
        {"Sigmoid", TranslateUnaryOp<opset::Sigmoid>},
        {"Sin", TranslateUnaryOp<opset::Sin>},
        {"Sinh", TranslateUnaryOp<opset::Sinh>},
        {"Size", TranslateSizeOp},
        {"Sign", TranslateUnaryOp<opset::Sign>},
        {"Slice", TranslateSliceOp},
        {"Snapshot", TranslateIdentityOp},
        {"Softmax", TranslateSoftmaxOp},
        {"Softplus", TranslateUnaryOp<opset::SoftPlus>},
        {"SpaceToDepth", TranslateSpaceToDepthOp},
        {"Split", TranslateSplitOp},
        {"SplitV", TranslateSplitVOp},
        {"Sqrt", TranslateUnaryOp<opset::Sqrt>},
        {"Square", TranslateSquareOp},
        {"SquaredDifference", TranslateBinaryOp<opset::SquaredDifference>},
        {"Squeeze", TranslateSqueezeOp},
        {"StridedSlice", TranslateStridedSliceOp},
        {"Sub", TranslateBinaryOp<opset::Subtract>},
        {"Sum", TranslateDirectReduceOp<opset::ReduceSum>},
        {"Tan", TranslateUnaryOp<opset::Tan>},
        {"Tanh", TranslateUnaryOp<opset::Tanh>},
        {"Tile", TranslateTileOp},
        {"TopKV2", TranslateTopKV2Op},
        {"Transpose", TranslateTransposeOp},
        {"Unpack", TranslateUnpackOp},
        {"Where", TranslateWhereOp},
        {"Xdivy", TranslateXdivyOp},
        {"ZerosLike", TranslateZerosLikeOp}};



class NodeProtoWrapper : public TFNodeDecoder
{
    const NodeDef* node_def;
    const GraphDef* graph_def;
    std::vector<TFNodeDecoder*>* nodes;
public:

    NodeProtoWrapper(const NodeDef* _node_def, const GraphDef* _graph_def, std::vector<TFNodeDecoder*>* _nodes) :
        node_def(_node_def), graph_def(_graph_def), nodes(_nodes) {}

#define GET_ATTR_VALUE(TYPE, FIELD) virtual void getAttrValue (const char* name, TYPE* x) const override \
    { *x = node_def->attr().at(name).FIELD(); }
#define GET_ATTR_VALUE_VECTOR(TYPE, FIELD) virtual void getAttrValue (const char* name, std::vector<TYPE>* x) const override \
    {\
        const auto& list = node_def->attr().at(name).list();\
        x->reserve(/*node_def->attr().at(name).FIELD##_size()*/list.FIELD##_size());\
        for(size_t i = 0; i < list.FIELD##_size(); ++i)\
        {\
            x->push_back(list.FIELD(i));\
        }\
    }

            GET_ATTR_VALUE_VECTOR(int32_t, i)
            GET_ATTR_VALUE_VECTOR(float, f)
    //virtual void getAttrValue (const char* name, std::vector<int32_t>* x) const override { NGRAPH_TF_FE_NOT_IMPLEMENTED; }
    //virtual void getAttrValue (const char* name, std::vector<float>* x) const override { NGRAPH_TF_FE_NOT_IMPLEMENTED; }
    GET_ATTR_VALUE(int32_t, i)

    virtual void getAttrValue (const char* name, DataType* x) const override
    {
        *x = node_def->attr().at(name).type();
    }

    virtual void getAttrValue (const char* name, ngraph::PartialShape* x) const override {
        TFTensorShapeToNGraphShape(node_def->attr().at(name).shape(), x);
    }

    GET_ATTR_VALUE(std::string, s)
    GET_ATTR_VALUE(bool, b)
    GET_ATTR_VALUE(long int, i)
    GET_ATTR_VALUE(float, f)

    virtual void getAttrValue (const char* name, std::vector<std::string>* x) const override { NGRAPH_TF_FE_NOT_IMPLEMENTED; }

    // a way to read Const value as a tensor
    virtual void getAttrValue (const char* name, TensorWrapper** x) const override
    {
        // TODO: use std::shared_ptr! memory is lost!
        *x = new TensorWrapper(&node_def->attr().at(name).tensor());
    }

    virtual std::string op () const override
    {
        return node_def->op();
    }

    virtual unsigned int num_inputs () const override { return node_def->input_size(); }

    virtual std::string name () const override
    {
        return node_def->name();
    }

    virtual std::string type_string () const override
    {
        return node_def->op();
    }

    virtual Status input_node (size_t index, TFNodeDecoder const * *) const override { NGRAPH_TF_FE_NOT_IMPLEMENTED; }

    virtual Status input_node (size_t index, TFNodeDecoder const * * retnode, size_t* outputPortIndex) const override
    {
        std::string input_name = node_def->input(index);
        if(input_name.find(':') != std::string::npos) {
            NGRAPH_TF_FE_NOT_IMPLEMENTED;
        }
        // TODO: don't search linearly every time!!!
        for(auto node: *nodes)
        {
            if(node->name() == input_name)
            {
                *retnode = node;
                *outputPortIndex = 0;
                return Status::OK();
            }
        }
        return Status("Node is not found " + input_name + " when searched as an input for node " + name());
    }

    virtual DataType input_type (size_t index) const override { NGRAPH_TF_FE_NOT_IMPLEMENTED; }
    virtual DataType output_type (size_t index) const override { NGRAPH_TF_FE_NOT_IMPLEMENTED; }

    virtual bool IsSink () const override
    {
        // TODO: recognize special op in TF runtime; don't know similar node for proto graph representation
        return false;
    }

    virtual bool IsSource () const override
    {
        // TODO: populate with other source operation types
        return node_def->op() == "Placeholder";
    }

    virtual bool IsControlFlow () const override
    {
        // TODO
        return false;
    }

    virtual std::string DebugString () const override
    {
        return node_def->op() + "(with name " + node_def->name() + ")";
    }

    virtual bool IsArg () const override
    {
        // TODO
        return IsSource();
    }

    virtual bool IsRetval () const override
    {
        // TODO
        return IsSink();
    }
};

void PopulateNodesTopologicallySorted (const GraphDef* input_graph, std::vector<TFNodeDecoder*>* result)
{
    // WARNING! We suppose that input_graph contains nodes in topologically sorted order
    // TODO: sort it if it is not the case

    result->reserve(input_graph->node_size());
    for(int i = 0; i < input_graph->node_size(); ++i)
    {
        result->push_back(new NodeProtoWrapper(&input_graph->node(i), input_graph, result));
    }
}

Status Builder::TranslateGraph(
        const std::map<std::string, ngraph::PartialShape>& inputs,
        const std::vector<const TensorWrapper*>& static_input_map, const GraphDef* input_graph,
        const std::string name, std::shared_ptr<ngraph::Function>& ng_function) {
  //
  // We will visit ops in topological order.
  //
  // ought to be `const TFNodeDecoder*`, but GetReversePostOrder doesn't use `const`

  std::vector<TFNodeDecoder*> ordered;
  //GetReversePostOrder(*input_graph, &ordered, NodeComparatorName());
  PopulateNodesTopologicallySorted(input_graph, &ordered);

  //
  // Split ops into params, retvals, and all others.
  //
  vector<const TFNodeDecoder*> tf_params;
  vector<const TFNodeDecoder*> tf_ret_vals;
  vector<const TFNodeDecoder*> tf_ops;

  for (const auto n : ordered) {
#if 0
      // TODO: Investigate why do we need it
      if (n->IsSink() || n->IsSource()) {
      continue;
    }
#endif

    if (n->IsControlFlow()) {
      return errors::Unimplemented(
          "Encountered a control flow op in the nGraph bridge: " +
          n->DebugString());
    }

    if (n->IsArg()) {
      tf_params.push_back(n);
    } else if (n->IsRetval()) {
      tf_ret_vals.push_back(n);
    } else {
      tf_ops.push_back(n);
    }
  }

  //
  // The op map holds a mapping from TensorFlow op names (strings) to
  // vector of generated nGraph Output<TFNodeDecoder>.
  //
  Builder::OpMap ng_op_map;

  //
  // Populate the parameter list, and also put parameters into the op map.
  //
  std::cerr << "[ INFO ] Detected " << tf_params.size() << " parameters\n";
  ng::ParameterVector ng_parameter_list(tf_params.size());
  // enumerate placeholders in some random order, count them and use counter as an index
  int index = 0;

  for (auto parm : tf_params) {
    DataType dtype;
    // TODO: replace dtype by T when converting Arg
    if (GetNodeAttr(parm->attrs(), "dtype", &dtype) != Status::OK()) {
      return errors::InvalidArgument("No data type defined for _Arg");
    }

    // TODO: use this code for Arg
    //if (GetNodeAttr(parm->attrs(), "index", &index) != Status::OK()) {
    //  return errors::InvalidArgument("No index defined for _Arg");
    //}

    ng::element::Type ng_et;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

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

#if 0
    string prov_tag;
    GetNodeAttr(parm->attrs(), "_prov_tag", &prov_tag);
#endif
    auto ng_param =
        ConstructNgNode<opset::Parameter>(parm->name(), ng_et, ng_shape);
    SaveNgOp(ng_op_map, parm->name(), ng_param);
    ng_parameter_list[index] =
        ngraph::as_type_ptr<opset::Parameter>(ng_param.get_node_shared_ptr());

    index++;
  }

  //
  // Now create the nGraph ops from TensorFlow ops.
  //
  for (auto op : tf_ops) {
    NGRAPH_VLOG(2) << "Constructing op " << op->name() << " which is "
                   << op->type_string() << "\n";

    const function<Status(const TFNodeDecoder*, const std::vector<const TensorWrapper*>&,
                          Builder::OpMap&)>* op_fun;

    try {
      op_fun = &(TRANSLATE_OP_MAP.at(op->type_string()));
    } catch (const std::out_of_range&) {
      // -----------------------------
      // Catch-all for unsupported ops
      // -----------------------------
      NGRAPH_VLOG(3) << "No translation handler registered for op: "
                     << op->name() << " (" << op->type_string() << ")";
      NGRAPH_VLOG(3) << op->DebugString();
      return errors::InvalidArgument(
          "No translation handler registered for op: " + op->name() + " (" +
          op->type_string() + ")\n" + op->DebugString());
    }

    try {
      TF_RETURN_IF_ERROR((*op_fun)(op, static_input_map, ng_op_map));
    } catch (const std::exception& e) {
      return errors::Internal("Unhandled exception in op handler: " + op->name() +
                              " (" + op->type_string() + ")\n" +
                              op->DebugString() + "\n" + "what(): " +
                              e.what());
    }
  }

  //
  // Populate the result list.
  //
  ng::ResultVector ng_result_list(tf_ret_vals.size());

  for (auto n : tf_ret_vals) {
    // Make sure that this _Retval only has one input node.
    if (n->num_inputs() != 1) {
      return errors::InvalidArgument("_Retval has " + to_string(n->num_inputs()) +
                                     " inputs, should have 1");
    }

    int index;
    if (GetNodeAttr(n->attrs(), "index", &index) != Status::OK()) {
      return errors::InvalidArgument("No index defined for _Retval");
    }

    ng::Output<ng::Node> result;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, n, 0, result));
    auto ng_result = ConstructNgNode<opset::Result>(n->name(), result);
    ng_result_list[index] =
        ngraph::as_type_ptr<opset::Result>(ng_result.get_node_shared_ptr());
  }

  // Find all terminal nodes in ngraph graph to complete list of results
  for(auto op: tf_ops)
  {
      auto p = ng_op_map.find(op->name());
      if(p != ng_op_map.end())
      {
          for(auto output: p->second)
          {
              if(output.get_target_inputs().empty())
                  ng_result_list.push_back(std::make_shared<default_opset::Result>(output));
          }
      }
  }

  //
  // Create the nGraph function.
  //
  ng_function =
      make_shared<ng::Function>(ng_result_list, ng_parameter_list, name);

  //
  // Apply additional passes on the nGraph function here.
  //
  {
#if 0
    ngraph::pass::Manager passes;
    if (util::GetEnv("NGRAPH_TF_CONSTANT_FOLDING") == "1") {
      passes.register_pass<ngraph::pass::ConstantFolding>();
    }
    if (util::GetEnv("NGRAPH_TF_TRANSPOSE_SINKING") != "0") {
      passes.register_pass<pass::TransposeSinking>();
    }
    passes.run_passes(ng_function);
#endif
  }
  NGRAPH_VLOG(5) << "Done with passes";
  //
  // Request row-major layout on results.
  //
  for (auto result : ng_function->get_results()) {
    result->set_needs_default_layout(true);
  }
  NGRAPH_VLOG(5) << "Done with translations";
  return Status::OK();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
