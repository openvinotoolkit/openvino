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

// Defined if we are building the plugin DLL (instead of using it)
//#ifdef tensorflow_ngraph_frontend_EXPORTS
//#define NGRAPH_HELPER_DLL_IMPORT __declspec(dllimport)
#define NGRAPH_HELPER_DLL_EXPORT __declspec(dllexport)

#define TF_API NGRAPH_HELPER_DLL_EXPORT
//#else
//#define TF_API NGRAPH_HELPER_DLL_IMPORT
//#endif // paddlepaddle_ngraph_frontend_EXPORTS
#define NGRAPH_VARIANT_DECLARATION(TYPE, info)                                                     \
    template <>                                                                                    \
    class VariantWrapper<TYPE> : public VariantImpl<TYPE>                                          \
    {                                                                                              \
    public:                                                                                        \
        static constexpr VariantTypeInfo type_info{info, 0};                                       \
        const VariantTypeInfo& get_type_info() const override { return type_info; }                \
        VariantWrapper<TYPE>(const value_type& value)                                              \
            : VariantImpl<value_type>(value)                                                       \
        {                                                                                          \
        }                                                                                          \
    }

namespace ngraph
{
/*
    NGRAPH_VARIANT_DECLARATION(int32_t, "Variant::int32");
    NGRAPH_VARIANT_DECLARATION(std::vector<int32_t>, "Variant::int32_vector");
    NGRAPH_VARIANT_DECLARATION(float, "Variant::float");
    NGRAPH_VARIANT_DECLARATION(std::vector<float>, "Variant::float_vector");
    NGRAPH_VARIANT_DECLARATION(bool, "Variant::bool");
    NGRAPH_VARIANT_DECLARATION(ngraph::element::Type, "Variant::element_type");
    NGRAPH_VARIANT_DECLARATION(std::vector<int64_t>, "Variant::int64_vector");
*/
namespace frontend
{
namespace tensorflow
{


namespace detail {

    class TFNodeDecoder;

/// Generic NodeContext that hides graph representation
/// It is base class for specific implementations for protobuf and run-time graph
    class TF_API NodeContext
    {
        OutputVector m_ng_inputs;
        std::shared_ptr<detail::TFNodeDecoder> m_decoder;

        // If shape is overridden for a particular node, it exists in the following map
        const std::map<std::string, ngraph::PartialShape> &m_overridden_shapes;

        // For special kind inputs (args) there are shapes defined externally here:
        const std::vector<ngraph::PartialShape> &m_indexed_shapes;

    public:

        NodeContext(
                const OutputVector &_ng_inputs,
                std::shared_ptr<detail::TFNodeDecoder> _decoder,
                const std::map<std::string, ngraph::PartialShape> &overridden_shapes,
                const std::vector<ngraph::PartialShape> &indexed_shapes = {});

        size_t get_ng_input_size() const;

        /// Returns a vector of already converted inputs for this node
        const OutputVector &get_ng_inputs() const;

        Output<Node> get_ng_input(size_t input_port) const;

        virtual std::string get_op_type() const;

        virtual std::vector<std::string> get_output_names() const;

        virtual std::vector<std::string> get_names() const;

        virtual std::string get_name() const;

        /// Temporary method for the transition period during migration to NodeContext
        // TODO: Remove this method and port all dependent code to the remaining methods
        const detail::TFNodeDecoder *_get_decoder() const;

        template<typename T>
        T get_attribute(const std::string &name) const;

        template<typename T>
        T get_attribute(const std::string &name, const T &default_value) const;

        // Meta-attributes like op type, domain, version -- some FW specific but common for all operations properties


        template<typename T>
        T get_meta_attribute(const std::string &name) const;

        template<typename T>
        T get_meta_attribute(const std::string &name, const T &default_value) const;

        const std::map<std::string, ngraph::PartialShape> &get_overridden_shapes() const;

        const std::vector<ngraph::PartialShape> &get_indexed_shapes() const;
    };

}
}
}
}

