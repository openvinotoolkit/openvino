// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <tensorflow_frontend/place.hpp>
#include <tensorflow_frontend/utility.hpp>

#include "graph.pb.h"
#include "tensorflow_frontend/node_context.hpp"

namespace tensorflow {

// Stub for TF class
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

// Stub for tf error system
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

}  // namespace tensorflow

namespace ngraph {
namespace frontend {
namespace tensorflow {

namespace detail {
// TODO: avoid using directly:
using ::tensorflow::DataType;
using ::tensorflow::Status;
using ::tensorflow::TensorProto;

class TensorWrapper;

// should be an ABI-free wrapper for TF node (not it is not ABI-free, TODO: eliminate ABI-dependent data structures)
class TFNodeDecoder {
public:
    // a hack to minimize amount of code
    TFNodeDecoder& attrs() const {
        return const_cast<TFNodeDecoder&>(*this);
    }
    virtual void getAttrValue2(const char* name, std::vector<int32_t>* x) const = 0;
    virtual void getAttrValue2(const char* name, std::vector<int64_t>* x) const = 0;
    virtual void getAttrValue2(const char* name, std::vector<float>* x) const = 0;
    virtual void getAttrValue2(const char* name, int32_t* x) const = 0;
    virtual void getAttrValue2(const char* name, int64_t* x) const = 0;
    virtual void getAttrValue2(const char* name, ngraph::element::Type* x) const = 0;
    virtual void getAttrValue2(const char* name, DataType* x) const = 0;
    virtual void getAttrValue2(const char* name, std::string* x) const = 0;
    virtual void getAttrValue2(const char* name, bool* x) const = 0;
    virtual void getAttrValue2(const char* name, float* x) const = 0;
    virtual void getAttrValue2(const char* name, std::vector<std::string>* x) const = 0;
    virtual void getAttrValue2(const char* name, ngraph::PartialShape* x) const = 0;
    virtual void getAttrValue2(const char* name, ngraph::frontend::tensorflow::detail::TensorWrapper** x) const = 0;

    virtual std::string op() const = 0;

    // a way to read Const value as a tensor
    // virtual void getAttrValue2 (const char* name, TensorWrapper** x) const = 0;

    virtual Status input_node(size_t index, std::string* name) const = 0;

    virtual Status input_node(size_t index, std::string* name, size_t* outputPortIndex) const = 0;

    virtual unsigned int num_inputs() const = 0;
    virtual std::string name() const = 0;
    virtual bool IsArg() const = 0;
    virtual std::string type_string() const = 0;

    virtual DataType input_type(size_t index) const = 0;
    virtual DataType output_type(size_t index) const = 0;

    virtual bool IsSink() const = 0;
    virtual bool IsSource() const = 0;
    virtual bool IsControlFlow() const = 0;
    virtual std::string DebugString() const = 0;
    virtual bool IsRetval() const = 0;
};

// TODO: separate interface from proto implementation; here is a proto implementation
class TensorWrapper {
public:
    const TensorProto* tensor_def;

    TensorWrapper(const TensorProto* _tensor_def) : tensor_def(_tensor_def) {}

    // a hack to minimize amount of code
    TensorWrapper& attrs() const {
        return const_cast<TensorWrapper&>(*this);
    }

    // virtual void getAttrValue(const char *name, std::vector<int32_t> &x) = 0;

    template <typename T>
    std::vector<T> flat() const;

    size_t NumElements() const;

    DataType dtype() const;
};

template <typename T>
Status GetNodeAttr(TFNodeDecoder& attrs, const char* attr_name, T* result) {
    attrs.getAttrValue2(attr_name, result);
    return Status::OK();
}

}  // namespace detail
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ngraph
