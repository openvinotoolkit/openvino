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
#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>


// TODO: remove explicit proto dependency from this common header
#include "graph.pb.h"

namespace tensorflow
{

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

#define TF_RETURN_IF_ERROR(S) if((S).status != 0)throw S;

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

}

namespace ngraph {
namespace frontend {
namespace tensorflow {

    namespace detail
    {
        // TODO: avoid using directly:
        using ::tensorflow::DataType;
        using ::tensorflow::Status;
        using ::tensorflow::TensorProto;

        class TensorWrapper;

// should be an ABI-free wrapper for TF node (not it is not ABI-free, TODO: eliminate ABI-dependent data structures)
        class TFNodeDecoder
        {
        public:

            // a hack to minimize amount of code
            TFNodeDecoder& attrs () const { return const_cast<TFNodeDecoder&>(*this); }
            virtual void getAttrValue (const char* name, std::vector<int32_t>* x) const = 0;
            virtual void getAttrValue (const char* name, std::vector<float>* x) const = 0;
            virtual void getAttrValue (const char* name, int32_t* x) const = 0;
            virtual void getAttrValue (const char* name, ngraph::element::Type* x) const = 0;
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

            virtual Status input_node (size_t index, std::string* name) const = 0;

            virtual Status input_node (size_t index, std::string* name, size_t* outputPortIndex) const = 0;

            virtual unsigned int num_inputs () const = 0;
            virtual std::string name () const = 0;
            virtual bool IsArg () const = 0;
            virtual std::string type_string () const = 0;

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
    }

/// Generic NodeContext that hides graph representation
/// It is base class for specific implementations for protobuf and run-time graph
class NodeContext
{
    OutputVector m_ng_inputs;
    std::shared_ptr<detail::TFNodeDecoder> m_decoder;

    // If shape is overridden for a particular node, it exists in the following map
    const std::map<std::string, ngraph::PartialShape>& m_overridden_shapes;

    // For special kind inputs (args) there are shapes defined externally here:
    const std::vector<ngraph::PartialShape>& m_indexed_shapes;

public:

    NodeContext (
            const OutputVector& _ng_inputs,
            std::shared_ptr<detail::TFNodeDecoder> _decoder,
            const std::map<std::string, ngraph::PartialShape>& overridden_shapes,
            const std::vector<ngraph::PartialShape>& indexed_shapes = {}) :
        m_ng_inputs(_ng_inputs),
        m_decoder(_decoder),
        m_overridden_shapes(overridden_shapes),
        m_indexed_shapes(indexed_shapes)
    {}

    size_t get_ng_input_size() const
    {
        return m_ng_inputs.size();
    }

    /// Returns a vector of already converted inputs for this node
    const OutputVector& get_ng_inputs () const
    {
        return m_ng_inputs;
    }

    Output<Node> get_ng_input (size_t input_port) const
    {
        return m_ng_inputs[input_port];
    }

    virtual std::string get_op_type() const
    {
        return m_decoder->op();
    }

    virtual std::vector<std::string> get_output_names() const
    {
        // TODO
        throw "Not implemented";
    }

    virtual std::vector<std::string> get_names() const
    {
        std::vector<std::string> names;
        names.push_back(m_decoder->name());
        return names;
    }

    virtual std::string get_name() const
    {
        return get_names()[0];
    }

    /// Temporary method for the transition period during migration to NodeContext
    // TODO: Remove this method and port all dependent code to the remaining methods
    const detail::TFNodeDecoder* _get_decoder() const
    {
        return m_decoder.get();
    }

    template <typename T>
    T get_attribute(const std::string& name) const
    {
        try {
            T result;
            m_decoder->getAttrValue(name.c_str(), &result);
            // TODO: no real processing of case when there is no default: getAttrValue will provide default even you don't need it
            return result;
        }
        catch(...)
        {
            std::cerr << "[ ERROR ] When accecing attribute '" << name << "' value.\n";
            throw;
        }
    }

    template <typename T>
    T get_attribute(const std::string& name, const T& default_value) const
    {
        T result;
        try {
            m_decoder->getAttrValue(name.c_str(), &result);
        } catch(...)  // TODO: replace by more narrow filter
        {
            result = default_value;
        }
        return result;
    }

    // Meta-attributes like op type, domain, version -- some FW specific but common for all operations properties


    template <typename T>
    T get_meta_attribute(const std::string& name) const;

    template <typename T>
    T get_meta_attribute(const std::string& name, const T& default_value) const;

    const std::map<std::string, ngraph::PartialShape>& get_overridden_shapes () const {
        return m_overridden_shapes;
    }

    const std::vector<ngraph::PartialShape>& get_indexed_shapes () const {
        return m_indexed_shapes;
    }
};

}
}
}

