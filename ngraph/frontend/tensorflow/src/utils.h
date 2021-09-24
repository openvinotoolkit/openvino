// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tensorflow_frontend/node_context.hpp>
#include "node_context_new.hpp"
#include "ngraph/ngraph.hpp"
#include "default_opset.h"
#include "graph.hpp"
#include "ngraph_builder.h"

namespace tensorflow {
namespace ngraph_bridge {

    template<typename Ttensor, typename Tvector>
    static void ConvertTensorDataToVector(const ngraph::frontend::tensorflow::detail::TensorWrapper &tensor,
                                          std::vector<Tvector> *vector) {
        const Ttensor *data = tensor.flat<Ttensor>().data();
        vector->resize(tensor.NumElements());
        for (int64_t i = 0; i < tensor.NumElements(); i++) {
            (*vector)[i] = Tvector(data[i]);
        }
    }

    template<typename T>
    static Status TensorDataToVector(const ngraph::frontend::tensorflow::detail::TensorWrapper &tensor,
                                     std::vector<T> *vector) {
        // stub
#if 0
        DataType dt = tensor.dtype();

      // If dt and T match, we can just copy.
      if (dt == DataTypeToEnum<T>::value) {
        *std::vector = std::vector<T>(tensor.flat<T>().data(),
                                 tensor.flat<T>().data() + tensor.NumElements());
      }
      // Else we have to convert.
      else {
        switch (dt) {
          case DT_FLOAT:
            ConvertTensorDataToVector<float, T>(tensor, std::vector);
            break;
          case DT_DOUBLE:
            ConvertTensorDataToVector<double, T>(tensor, std::vector);
            break;
          case DT_INT8:
            ConvertTensorDataToVector<int8_t, T>(tensor, std::vector);
            break;
          case DT_INT16:
            ConvertTensorDataToVector<int16_t, T>(tensor, std::vector);
            break;
          case DT_INT32:
            ConvertTensorDataToVector<int32_t, T>(tensor, std::vector);
            break;
          case DT_INT64:
            ConvertTensorDataToVector<int64_t, T>(tensor, std::vector);
            break;
          case DT_UINT8:
            ConvertTensorDataToVector<uint8_t, T>(tensor, std::vector);
            break;
          case DT_UINT16:
            ConvertTensorDataToVector<uint16_t, T>(tensor, std::vector);
            break;
          case DT_UINT32:
            ConvertTensorDataToVector<uint32, T>(tensor, std::vector);
            break;
          case DT_UINT64:
            ConvertTensorDataToVector<uint64, T>(tensor, std::vector);
            break;
          case DT_BOOL:
            ConvertTensorDataToVector<bool, T>(tensor, std::vector);
            break;
          default:
            return errors::Internal("TensorDataToVector: tensor has element type ",
                                    DataType_Name(dt), ", std::vector has type ",
                                    DataType_Name(DataTypeToEnum<T>::value),
                                    "; don't know how to convert");
        }
      }
      return Status::OK();
#endif

        NGRAPH_TF_FE_NOT_IMPLEMENTED
    }


    static bool VecStrCmp(const std::vector<std::string>& a, const std::vector<std::string>& b) {
        return a == b;
    }

    static Status ValidateInputCount(const ngraph::frontend::tf::NodeContext& op, size_t count) {
        if (op.get_ng_input_size() != count) {
            std::ostringstream buf;
            buf << "\"" << op.get_names()[0] << "\" requires " << count << " input(s), got " << op.get_ng_input_size()
                << " instead";
            return errors::InvalidArgument(buf.str());
        }
        return Status::OK();
    }

    static void ValidateInputCountMin(const ngraph::frontend::tf::NodeContext& node, size_t count) {
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
// std::vector ng_op_map[op_name] would contain one element.
//
// If storing more than one output_nodes, make sure it's in
// the same order as tensorflow would do that.
//
// Parameters:
//    Builder::OpMap& ng_op_map        - The TF-to-nGraph op map.
//    std::string op_name              - Name of the op.
//
//    ngraph::Output<ngraph::Node> output_node - ngraph::Node to store
//

        static void SaveNgOp(Builder::OpMap& ng_op_map, const std::string& op_name, ngraph::Output<ngraph::Node> output_node) {
            // no need to try-catch, map[key] will create std::vector object
            // if not exists
            ng_op_map[op_name].push_back(output_node);
        }

        template <class TOpType, class... TArg>
        ngraph::Output<ngraph::Node> ConstructNgNode(const std::string& op_name, TArg&&... Args) {
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
//      ngraph::Output<ngraph::Node> ng_input;
//      try {
//        ng_input = ng_op_map.at(tf_input->name());
//      } catch (const std::out_of_range&) {
//        return errors::NotFound(tf_input->name(),
//                                    " is not found in the ng_op_map");
//      }
//
// Into 2 lines:
//
//      ngraph::Output<ngraph::node> ng_input;
//      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input))
//
//
//
// Parameters:
//    Builder::OpMap& ng_op_map     - The TF-to-nGraph op map.
//    TFNodeDecoder* op                  - TF op being translated.
//    input_idx                     - index of input
//
//    ngraph::Output<ngraph::Node> *result  - ngraph::Node pointer where result
//                                    will be written
//
//

        static Status GetInputNode(const NodeContext op, size_t input_idx, ngraph::Output<ngraph::Node>& result) {
// Stub
#if 0
            // input op may have resulted in more than one ngraph::Node (eg. Split)
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
                                        ngraph::Output<ngraph::Node>& result,
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

        template <typename T>
        static void GetStaticInputVector(const ngraph::frontend::tf::NodeContext& node, int64_t input_index, std::vector<T>* vector) {
            ngraph::Output<ngraph::Node> ng_input = node.get_ng_input(input_index);
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
                TF_RETURN_IF_ERROR(TensorDataToVector(input_tensor, std::vector));
                return Status::OK();*/
        }

#if 0
        template <typename T>
static Status GetStaticInputVector(
    const TFNodeDecoder* op, int64_t input_index,
    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map,
    std::vector<T>* std::vector) {
  TFNodeDecoder* input_node;
  TF_RETURN_IF_ERROR(op->input_node(input_index, &input_node));
  ngraph::frontend::tensorflow::detail::TensorWrapper* input_tensor;
  TF_RETURN_IF_ERROR(
      GetStaticNodeTensor(input_node, static_input_map, &input_tensor));
  TF_RETURN_IF_ERROR(TensorDataToVector(input_tensor, std::vector));
  return Status::OK();
}

static Status GetStaticInputNode(
    const TFNodeDecoder* op, int64_t input_index,
    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map, DataType dt,
    ngraph::Output<ngraph::Node>& node_) {
  ngraph::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dt, &type));
  switch (dt) {
    case DataType::DT_FLOAT: {
      std::vector<float> vec_float;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_float));
      node_ = ConstructNgNode<opset::Constant>(node.get_name(), type, ngraph::Shape{},
                                               vec_float[0]);
    } break;
    case DataType::DT_DOUBLE: {
      std::vector<double> vec_double;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_double));
      node_ = ConstructNgNode<opset::Constant>(node.get_name(), type, ngraph::Shape{},
                                               vec_double[0]);
    } break;
    case DataType::DT_INT32: {
      std::vector<int32_t> vec_i32;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_i32));
      node_ = ConstructNgNode<opset::Constant>(node.get_name(), type, ngraph::Shape{},
                                               vec_i32[0]);
    } break;
    case DataType::DT_INT64: {
      std::vector<int64_t> vec_i64;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_i64));
      node_ = ConstructNgNode<opset::Constant>(node.get_name(), type, ngraph::Shape{},
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
// in the std::vector does not match TensorFlow's notion of what the C++ type
// should be (e.g. when T is `bool`, we actually need a std::vector of `char` for
// compatibility with nGraph).
        template <typename T, typename VecT = T>
        static Status ValuesFromConstNode(const DecoderBase* node,
                                          ngraph::Shape* const_tensor_shape,
                                          std::vector<VecT>* values) {
#if 1

            if (node->get_op_type() != "Const") {
                return errors::InvalidArgument("TFNodeDecoder not a Const");
            }
            auto dt1 = node->get_attribute("dtype", ::ov::VariantWrapper<::tensorflow::DataType>::type_info);
            FRONT_END_GENERAL_CHECK(dt1);
            auto dt = std::dynamic_pointer_cast<::ov::VariantWrapper<::tensorflow::DataType>>(dt1)->get();

            /*
            if (dt != DataTypeToEnum<T>::value) {
              std::stringstream ss;
              ss << "Invalid data type defined for Const. Defined: "
                 << node.attr().at("dtype").type();
              return errors::InvalidArgument(ss.str());
            }
            */

            // ngraph::frontend::tensorflow::detail::TensorWrapper represents the content of the tensor in either
            // <type>_val or tensor_content.

            auto tensor_proto_var =
                node->get_attribute("value", ::ov::VariantWrapper<::tensorflow::TensorProto>::type_info);
            FRONT_END_GENERAL_CHECK(tensor_proto_var);
            auto tensor_proto =
                std::dynamic_pointer_cast<::ov::VariantWrapper<::tensorflow::TensorProto>>(tensor_proto_var)->get();

            // typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
            //    checkpoint::MutableTensorProtoData<T>(const_cast<ngraph::frontend::tensorflow::detail::TensorWrapper*>(&tensor));

            const TensorShapeProto& shape = tensor_proto.tensor_shape();
            ngraph::PartialShape pshape;
            TFTensorShapeToNGraphShape(shape, &pshape);
            *const_tensor_shape = pshape.get_shape();
            if (pshape.is_dynamic())
                NGRAPH_TF_FE_NOT_IMPLEMENTED;
            auto tensor_content = tensor_proto.tensor_content();
            std::vector<char> tensor_values_plain(tensor_content.begin(), tensor_content.end());
            const T* tensor_values = reinterpret_cast<const T*>(tensor_values_plain.data());

            if (!tensor_values_plain.empty() && tensor_proto.has_tensor_shape()) {
                // When tensor_shape is set, theoretically the representation of the data
                // could be compressed. So, before copying values to the returned vector,
                // make sure no compression happens.
                // if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values_plain.size()/sizeof(T)) {
                values->insert(values->end(), tensor_values, tensor_values + tensor_values_plain.size() / sizeof(T));
                return Status::OK();
                //}
            }

            const auto tensor_content_size = tensor_proto.tensor_content().size();
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
                        // NGRAPH_VLOG(0) << node->DebugString();
                        // NGRAPH_VLOG(0) << shape.DebugString();
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
        static Status MakeConstOp(const NodeContext& node, ngraph::element::Type et, ngraph::Output<ngraph::Node>& ng_node) {
            std::vector<VecT> const_values;
            ngraph::Shape ng_shape;

            TF_RETURN_IF_ERROR((ValuesFromConstNode<T, VecT>(node._get_decoder(), &ng_shape, &const_values)));

            ng_node = ConstructNgNode<opset::Constant>(node.get_name(), et, ng_shape, const_values);
            return Status::OK();
        }
}
}
