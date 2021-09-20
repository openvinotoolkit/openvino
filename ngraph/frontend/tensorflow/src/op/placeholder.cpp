// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <default_opset.h>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector PlaceholderOp(const NodeContext& node) {
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

    element::Type ng_et;
    TFDataTypeToNGraphElementType(dtype, &ng_et);

    PartialShape ng_shape;
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
}
}