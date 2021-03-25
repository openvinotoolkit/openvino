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
#ifndef NGRAPH_TF_BRIDGE_BUILDER_H_
#define NGRAPH_TF_BRIDGE_BUILDER_H_

#include <ostream>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>

// TODO: remove explicit proto dependency from this common header
#include "graph.pb.h"

#include "ngraph/ngraph.hpp"

namespace tensorflow {

// Stub for TF class
class Status
{
public:
    int status = 0;
    std::string message;

    static Status OK () { return Status(); }

    Status (const std::string& x) : message(x), status(1) {}
    Status () {}
};

inline bool operator!= (const Status& x, const Status& y)
{
    return x.status != y.status;
}

inline std::ostream& operator<< (std::ostream& out, const Status& s)
{
    return out << s.message;
}

#define TF_RETURN_IF_ERROR(S) if((S).status != 0)return S;

// Stub for tf error system
class errors
{
public:

    static Status InvalidArgument (const std::string& x)
    {
        return Status("InvalidArgument: " + x);
    }

    static Status Internal (const std::string& x)
    {
        return Status("Internal: " + x);
    }

    static Status Unimplemented (const std::string& x)
    {
        return Status("Unimplemented: " + x);
    }
};


namespace ngraph_bridge {

    class TensorWrapper;

// ABI-free wrapper for TF node
class TFNodeDecoder
{
public:

    // a hack to minimize amount of code
    TFNodeDecoder& attrs () const { return const_cast<TFNodeDecoder&>(*this); }
    virtual void getAttrValue (const char* name, std::vector<int32_t>* x) const = 0;
    virtual void getAttrValue (const char* name, std::vector<float>* x) const = 0;
    virtual void getAttrValue (const char* name, int32_t* x) const = 0;
    virtual void getAttrValue (const char* name, DataType* x) const = 0;
    virtual void getAttrValue (const char* name, std::string* x) const = 0;
    virtual void getAttrValue (const char* name, bool* x) const = 0;
    virtual void getAttrValue (const char* name, long int* x) const = 0;
    virtual void getAttrValue (const char* name, float* x) const = 0;
    virtual void getAttrValue (const char* name, std::vector<std::string>* x) const = 0;
    virtual void getAttrValue (const char* name, ngraph::PartialShape* x) const = 0;

    virtual std::string op () const = 0;

    // a way to read Const value as a tensor
    virtual void getAttrValue (const char* name, TensorWrapper** x) const = 0;

    virtual unsigned int num_inputs () const = 0;
    virtual std::string name () const = 0;
    virtual bool IsArg () const = 0;
    virtual std::string type_string () const = 0;

    virtual Status input_node (size_t index, TFNodeDecoder const * *) const = 0;
    virtual Status input_node (size_t index, TFNodeDecoder const * *, size_t* outputPortIndex) const = 0;
    virtual DataType input_type (size_t index) const = 0;
    virtual DataType output_type (size_t index) const = 0;

    virtual bool IsSink () const = 0;
    virtual bool IsSource () const = 0;
    virtual bool IsControlFlow () const = 0;
    virtual std::string DebugString () const = 0;
    virtual bool IsRetval () const = 0;
};

// TODO: separate interface from proto implementation; here is a proto implementation
class TensorWrapper
{
public:

    const TensorProto* tensor_def;

    TensorWrapper (const TensorProto* _tensor_def) : tensor_def(_tensor_def) {}

    // a hack to minimize amount of code
    TensorWrapper &attrs() const { return const_cast<TensorWrapper &>(*this); }

    //virtual void getAttrValue(const char *name, std::vector<int32_t> &x) = 0;

    template <typename T>
    std::vector<T> flat () const;

    size_t NumElements () const;

    DataType dtype () const;
};

template <typename T>
Status GetNodeAttr (TFNodeDecoder& attrs, const char* attr_name, T* result)
{
    attrs.getAttrValue(attr_name, result);
    return Status::OK();
}

#if 0
#define NGRAPH_VLOG(I) std::cerr
#else
#define NGRAPH_VLOG(I) std::ostringstream()
#endif


    class Builder {
 public:
  static Status TranslateGraph(
      const std::map<std::string, ngraph::PartialShape>& inputs,
      const std::vector<const TensorWrapper*>& static_input_map, const GraphDef* tf_graph,
      const std::string name, std::shared_ptr<ngraph::Function>& ng_function);

  using OpMap = std::unordered_map<std::string,
                                   std::vector<ngraph::Output<ngraph::Node>>>;
  using ConstMap = std::map<
      DataType,
      std::pair<std::function<Status(const TFNodeDecoder*, ngraph::element::Type,
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
