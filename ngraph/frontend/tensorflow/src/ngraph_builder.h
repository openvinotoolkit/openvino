// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_TF_BRIDGE_BUILDER_H_
#define NGRAPH_TF_BRIDGE_BUILDER_H_

#include <ostream>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>

#include "graph.hpp"

// TODO: remove explicit proto dependency from this common header
//#include "graph.pb.h"

#include "node_context_impl.hpp"
#include "tf_framework_node.hpp"

#include "ngraph/ngraph.hpp"

#include "node_context_new.hpp"

namespace tensorflow {

using ngraph::OutputVector;
//using ngraph::frontend::tensorflow::detail::NodeContext;
using ngraph::frontend::tf::NodeContext;

// TODO: Get rid of direct usage of this structures and remove the following usages:
using ngraph::frontend::tensorflow::detail::TFNodeDecoder;
using ngraph::frontend::tensorflow::detail::TensorWrapper;



namespace ngraph_bridge {



#if 0
#define NGRAPH_VLOG(I) std::cerr
#else
#define NGRAPH_VLOG(I) std::ostringstream()
#endif


class Builder {
 public:
    static void TranslateGraph(
        const std::shared_ptr<ngraph::frontend::InputModelTF>& tf_model,
        const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map,
        const std::string name,
        bool fail_fast,
        bool no_conversion,
        std::shared_ptr<ngraph::Function>& ng_function);
    static void TranslateFWNode(const std::shared_ptr<TFFrameworkNode>& node);

  using OpMap = std::unordered_map<std::string,
                                   std::vector<ngraph::Output<ngraph::Node>>>;
  using ConstMap = std::map<
      ngraph::element::Type,
      std::pair<std::function<Status(const NodeContext&, ngraph::element::Type,
                                     ngraph::Output<ngraph::Node>&)>,
                const ngraph::element::Type>>;
  static const Builder::ConstMap& TF_NGRAPH_CONST_MAP();

  template <typename T>
  static void MakePadding(const std::string& tf_padding_type,
                          const ngraph::Shape& ng_image_shape,
                          const ngraph::Shape& ng_kernel_shape,
                          const ngraph::Strides& ng_strides,
                          const ngraph::Shape& ng_dilations,
                          T& ng_padding_below, T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
      ngraph::Shape img_shape = {0, 0};
      img_shape.insert(img_shape.end(), ng_image_shape.begin(),
                       ng_image_shape.end());
      ngraph::infer_auto_padding(img_shape, ng_kernel_shape, ng_strides,
                                 ng_dilations, ngraph::op::PadType::SAME_UPPER,
                                 ng_padding_above, ng_padding_below);
    } else if (tf_padding_type == "VALID") {
      ng_padding_below.assign(ng_image_shape.size(), 0);
      ng_padding_above.assign(ng_image_shape.size(), 0);
    }
  }

  // This function is used to trace which ng node came from which tf node
  // It does 3 things:
  // 1. Attaches provenance tags. This is guaranteed to propagate the tag info
  // to all nodes.
  // The next 2 are not guaranteed to be present for all nodes.
  // But when present they are correct and agree with provenance tags
  // 2. Attaches friendly names.
  // 3. Prints a log if NGRAPH_TF_LOG_PLACEMENT=1
  static void SetTracingInfo(const std::string& op_name,
                             const ngraph::Output<ngraph::Node> ng_node);
};

inline std::string StrJoin (const std::vector<std::string>& strs, const char* sep)
{
    std::ostringstream str;
    std::copy(strs.begin(), strs.end(), std::ostream_iterator<std::string>(str, sep));
    return str.str();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
