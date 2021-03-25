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

#include "ngraph_conversions.h"

namespace tensorflow {
namespace ngraph_bridge {

void NHWCtoNCHW(const std::string& op_name, bool is_nhwc,
                ngraph::Output<ngraph::Node>& node) {
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

void NCHWtoNHWC(const std::string& op_name, bool is_nhwc,
                ngraph::Output<ngraph::Node>& node) {
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


    Status TFDataTypeToNGraphElementType(DataType tf_dt,
                                         ngraph::element::Type* ng_et) {
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
                return errors::Unimplemented("Unsupported TensorFlow data type: " +
                                             DataType_Name(tf_dt));
        }
        return Status::OK();
    }

    Status TFTensorShapeToNGraphShape(const ::tensorflow::TensorShapeProto& tf_shape,
                                      ngraph::PartialShape* ng_shape) {
        std::vector<ngraph::Dimension> dims;
        for (int i = 0; i < tf_shape.dim_size(); i++) {
            dims.push_back(tf_shape.dim(i).size());
        }
        *ng_shape = ngraph::PartialShape(dims);
        return Status::OK();
    }

}  // namespace ngraph_bridge
}  // namespace tensorflow
