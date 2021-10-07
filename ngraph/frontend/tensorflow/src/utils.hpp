// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset8.hpp>

#include "graph_iterator_proto.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph_conversions.hpp"
#include "node_context.hpp"

namespace ngraph {
namespace frontend {
namespace tf {
namespace detail {
// TODO: avoid using directly:
using ::tensorflow::DataType;
using ::tensorflow::TensorProto;

// TODO: separate interface from proto implementation; here is a proto implementation
class TensorWrapper {
public:
    const TensorProto* tensor_def;

    TensorWrapper(const TensorProto* _tensor_def) : tensor_def(_tensor_def) {}

    // a hack to minimize amount of code
    TensorWrapper& attrs() const {
        return const_cast<TensorWrapper&>(*this);
    }

    template <typename T>
    std::vector<T> flat() const;

    size_t NumElements() const;

    DataType dtype() const;
};

}  // namespace detail
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph

namespace ngraph {
namespace frontend {
namespace tf {
using namespace ::ngraph::frontend::tf::detail;

using OpMap = std::unordered_map<std::string, std::vector<ngraph::Output<ngraph::Node>>>;

void extract_operation_name_and_port(const std::string& port_name,
                                     std::string& operation_name,
                                     size_t& port_index,
                                     std::string& port_type);

class Status {
public:
    int status = 0;
    std::string message;

    static Status OK() {
        return Status();
    }

    Status(const std::string& x) : message(x), status(1) {}
    Status() {}
};

inline bool operator!=(const Status& x, const Status& y) {
    return x.status != y.status;
}

inline std::ostream& operator<<(std::ostream& out, const Status& s) {
    return out << s.message;
}

#define TF_RETURN_IF_ERROR(S) \
    if ((S).status != 0)      \
        throw S;

class errors {
public:
    static Status InvalidArgument(const std::string& x) {
        return Status("InvalidArgument: " + x);
    }

    static Status Internal(const std::string& x) {
        return Status("Internal: " + x);
    }

    static Status Unimplemented(const std::string& x) {
        return Status("Unimplemented: " + x);
    }
};

void SetTracingInfo(const std::string& op_name, const ngraph::Output<ngraph::Node> ng_node);

template <typename T>
static void MakePadding(const std::string& tf_padding_type,
                        const ngraph::Shape& ng_image_shape,
                        const ngraph::Shape& ng_kernel_shape,
                        const ngraph::Strides& ng_strides,
                        const ngraph::Shape& ng_dilations,
                        T& ng_padding_below,
                        T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
        ngraph::Shape img_shape = {0, 0};
        img_shape.insert(img_shape.end(), ng_image_shape.begin(), ng_image_shape.end());
        ngraph::infer_auto_padding(img_shape,
                                   ng_kernel_shape,
                                   ng_strides,
                                   ng_dilations,
                                   ngraph::op::PadType::SAME_UPPER,
                                   ng_padding_above,
                                   ng_padding_below);
    } else if (tf_padding_type == "VALID") {
        ng_padding_below.assign(ng_image_shape.size(), 0);
        ng_padding_above.assign(ng_image_shape.size(), 0);
    }
}

template <typename Ttensor, typename Tvector>
static void ConvertTensorDataToVector(const TensorWrapper& tensor, std::vector<Tvector>* vector) {
    const Ttensor* data = tensor.flat<Ttensor>().data();
    vector->resize(tensor.NumElements());
    for (int64_t i = 0; i < tensor.NumElements(); i++) {
        (*vector)[i] = Tvector(data[i]);
    }
}

static bool VecStrCmp(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    return a == b;
}

static Status ValidateInputCount(const NodeContext& op, size_t count) {
    if (op.get_ng_input_size() != count) {
        std::ostringstream buf;
        buf << "\"" << op.get_name() << "\" requires " << count << " input(s), got " << op.get_ng_input_size()
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
static void SaveNgOp(OpMap& ng_op_map, const std::string& op_name, ngraph::Output<ngraph::Node> output_node) {
    // no need to try-catch, map[key] will create std::vector object
    // if not exists
    ng_op_map[op_name].push_back(output_node);
}

template <class TOpType, class... TArg>
ngraph::Output<ngraph::Node> ConstructNgNode(const std::string& op_name, TArg&&... Args) {
    auto ng_node = std::make_shared<TOpType>(std::forward<TArg>(Args)...);
    SetTracingInfo(op_name, ng_node);
    return ng_node;
}

static Status GetInputNode(const NodeContext op, size_t input_idx, ngraph::Output<ngraph::Node>& result) {
    // Stub
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

void TFTensorShapeToNGraphShape(const ::tensorflow::TensorShapeProto& tf_shape, ngraph::PartialShape* ng_shape);

template <typename T>
static void GetStaticInputVector(const ngraph::frontend::tf::NodeContext& node,
                                 int64_t input_index,
                                 std::vector<T>* vector) {
    ngraph::Output<ngraph::Node> ng_input = node.get_ng_input(input_index);
    if (auto constant = std::dynamic_pointer_cast<ngraph::opset8::Constant>(ng_input.get_node_shared_ptr())) {
        *vector = constant->cast_vector<T>();
        return;
    }
}

// Taken from: tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc
// Extract values from a Const op to `values`. Returns true if succeeds.
//
// Modified with an extra `VecT` parameter to handle the case where the type
// in the std::vector does not match TensorFlow's notion of what the C++ type
// should be (e.g. when T is `bool`, we actually need a std::vector of `char` for
// compatibility with nGraph).
template <typename T, typename VecT = T>
static Status ValuesFromConstNode(const ::ngraph::frontend::DecoderBase* node,
                                  ngraph::Shape* const_tensor_shape,
                                  std::vector<VecT>* values) {
    if (node->get_op_type() != "Const") {
        return errors::InvalidArgument("TFNodeDecoder not a Const");
    }
    auto dt1 = node->get_attribute("dtype", ::ov::VariantWrapper<::tensorflow::DataType>::get_type_info_static());
    FRONT_END_GENERAL_CHECK(dt1);
    auto dt = std::dynamic_pointer_cast<::ov::VariantWrapper<::tensorflow::DataType>>(dt1)->get();

    auto tensor_proto_var =
        node->get_attribute("value", ::ov::VariantWrapper<::tensorflow::TensorProto>::get_type_info_static());
    FRONT_END_GENERAL_CHECK(tensor_proto_var);
    auto tensor_proto =
        std::dynamic_pointer_cast<::ov::VariantWrapper<::tensorflow::TensorProto>>(tensor_proto_var)->get();

    const tensorflow::TensorShapeProto& shape = tensor_proto.tensor_shape();
    ngraph::PartialShape pshape;
    TFTensorShapeToNGraphShape(shape, &pshape);
    *const_tensor_shape = pshape.get_shape();
    if (pshape.is_dynamic())
        FRONT_END_NOT_IMPLEMENTED("ValuesFromConstNode");
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
            case tensorflow::DT_INT32:
                val_size = tensor_proto.int_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.int_val()[i];
                break;
            case tensorflow::DT_INT64:
                val_size = tensor_proto.int64_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.int64_val()[i];
                break;
            case tensorflow::DT_FLOAT:
                val_size = tensor_proto.float_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.float_val()[i];
                break;
            case tensorflow::DT_BOOL:
                val_size = tensor_proto.bool_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.bool_val()[i];
                break;
            case tensorflow::DT_DOUBLE:
                val_size = tensor_proto.double_val_size();
                if (val_size > 0)
                    val_i = tensor_proto.double_val()[i];
                break;
            default:
                NGRAPH_VLOG(0) << "Const node has empty tensor_proto and we don't know how to "
                                  "handle this element type";
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
    }

    return Status::OK();
}

template <typename T, typename VecT = T>
static Status MakeConstOp(const NodeContext& node, ngraph::element::Type et, ngraph::Output<ngraph::Node>& ng_node) {
    std::vector<VecT> const_values;
    ngraph::Shape ng_shape;

    TF_RETURN_IF_ERROR((ValuesFromConstNode<T, VecT>(node.get_decoder(), &ng_shape, &const_values)));

    ng_node = ConstructNgNode<ngraph::opset8::Constant>(node.get_name(), et, ng_shape, const_values);
    return Status::OK();
}
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
