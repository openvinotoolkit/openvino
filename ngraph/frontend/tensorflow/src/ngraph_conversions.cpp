// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_conversions.h"

#include "ngraph_builder.h"

namespace tensorflow {
namespace ngraph_bridge {

void NHWCtoNCHW(const std::string& op_name, bool is_nhwc, ngraph::Output<ngraph::Node>& node) {
    if (is_nhwc) {
        auto rank = node.get_shape().size();
        if (rank == 4) {
            Transpose<0, 3, 1, 2>(node);
        } else if (rank == 5) {
            Transpose3D<0, 4, 1, 2, 3>(node);
        }
        Builder::SetTracingInfo(op_name, node);
    }
}

void NCHWtoNHWC(const std::string& op_name, bool is_nhwc, ngraph::Output<ngraph::Node>& node) {
    if (is_nhwc) {
        auto rank = node.get_shape().size();
        if (rank == 4) {
            Transpose<0, 2, 3, 1>(node);
        } else if (rank == 5) {
            Transpose3D<0, 2, 3, 4, 1>(node);
        }
        Builder::SetTracingInfo(op_name, node);
    }
}

void TFDataTypeToNGraphElementType(DataType tf_dt, ngraph::element::Type* ng_et) {
    switch (tf_dt) {
    case DataType::DT_FLOAT:
        *ng_et = ngraph::element::f32;
        break;
    case DataType::DT_DOUBLE:
        *ng_et = ngraph::element::f64;
        break;
    case DataType::DT_INT32:
        *ng_et = ngraph::element::i32;
        break;
    case DataType::DT_UINT8:
        *ng_et = ngraph::element::u8;
        break;
    case DataType::DT_INT8:
        *ng_et = ngraph::element::i8;
        break;
    case DataType::DT_UINT16:
        *ng_et = ngraph::element::u16;
        break;
    case DataType::DT_INT64:
        *ng_et = ngraph::element::i64;
        break;
    case DataType::DT_UINT32:
        *ng_et = ngraph::element::u32;
        break;
    case DataType::DT_UINT64:
        *ng_et = ngraph::element::u64;
        break;
    case DataType::DT_BOOL:
        *ng_et = ngraph::element::boolean;
        break;
    case DataType::DT_QINT8:
        *ng_et = ngraph::element::i8;
        break;
    case DataType::DT_QUINT8:
        *ng_et = ngraph::element::u8;
        break;
    case DataType::DT_QINT32:
        *ng_et = ngraph::element::i32;
        break;
    case DataType::DT_BFLOAT16:
        *ng_et = ngraph::element::bf16;
        break;
    case DataType::DT_HALF:
        *ng_et = ngraph::element::f16;
        break;
    default:
        throw errors::Unimplemented("Unsupported TensorFlow data type: " + DataType_Name(tf_dt));
    }
}

void TFTensorShapeToNGraphShape(const ::tensorflow::TensorShapeProto& tf_shape, ngraph::PartialShape* ng_shape) {
    std::vector<ngraph::Dimension> dims;
    for (int i = 0; i < tf_shape.dim_size(); i++) {
        dims.push_back(tf_shape.dim(i).size());
    }
    *ng_shape = ngraph::PartialShape(dims);
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
