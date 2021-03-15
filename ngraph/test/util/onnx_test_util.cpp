//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <exception>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>
#include <sstream>

#include "ngraph/except.hpp"
#include "onnx_test_util.hpp"

using namespace ngraph;
using namespace ngraph::test;

namespace
{
    void parse_from_istream(std::istream& model_stream, ONNX_NAMESPACE::ModelProto& model_proto)
    {
        if (!model_stream.good())
        {
            model_stream.clear();
            model_stream.seekg(0);
            if (!model_stream.good())
            {
                throw ngraph_error("Provided input stream has incorrect state.");
            }
        }

        if (!model_proto.ParseFromIstream(&model_stream))
        {
#ifdef NGRAPH_USE_PROTOBUF_LITE
            throw ngraph_error(
                "Error during import of ONNX model provided as input stream "
                " with binary protobuf message.");
#else
            // Rewind to the beginning and clear stream state.
            model_stream.clear();
            model_stream.seekg(0);
            google::protobuf::io::IstreamInputStream iistream(&model_stream);
            // Try parsing input as a prototxt message
            if (!google::protobuf::TextFormat::Parse(&iistream, &model_proto))
            {
                throw ngraph_error(
                    "Error during import of ONNX model provided as input stream with prototxt "
                    "protobuf message.");
            }
#endif
        }
    }

    void parse_from_file(const std::string& file_path, ONNX_NAMESPACE::ModelProto& model_proto)
    {
        std::ifstream file_stream{file_path, std::ios::in | std::ios::binary};

        if (!file_stream.is_open())
        {
            throw ngraph_error("Could not open the file: " + file_path);
        };

        parse_from_istream(file_stream, model_proto);
    }

    ComparisonResult compare_nodes(const ONNX_NAMESPACE::GraphProto& graph,
                                   const ONNX_NAMESPACE::GraphProto& ref_graph)
    {
        if (graph.node_size() != ref_graph.node_size())
        {
            return ComparisonResult::fail("The number of nodes in compared models doesn't match");
        }
        else
        {
            for (int i = 0; i < graph.node_size(); ++i)
            {
                const auto& lhs = graph.node(i);
                const auto& rhs = ref_graph.node(i);

                if (lhs.op_type() != rhs.op_type())
                {
                    return ComparisonResult::fail("Operation types are different at index " +
                                                  std::to_string(i) + ": " + lhs.op_type() +
                                                  " vs " + rhs.op_type());
                }

                for (int j = 0; j < lhs.input_size(); ++j)
                {
                    if (lhs.input(j) != rhs.input(j))
                    {
                        return ComparisonResult::fail(
                            "Input names don't match for nodes at index " + std::to_string(i) +
                            ": " + lhs.input(j) + " vs " + rhs.input(j));
                    }
                }

                for (int j = 0; j < lhs.output_size(); ++j)
                {
                    if (lhs.output(j) != rhs.output(j))
                    {
                        return ComparisonResult::fail(
                            "Output names don't match for nodes at index " + std::to_string(i) +
                            ": " + lhs.output(j) + " vs " + rhs.output(j));
                    }
                }
            }
        }

        return ComparisonResult::pass();
    }

    ComparisonResult compare_value_info(const ONNX_NAMESPACE::ValueInfoProto& lhs,
                                        const ONNX_NAMESPACE::ValueInfoProto& rhs,
                                        const std::string& item_type)
    {
        if (lhs.name() != rhs.name())
        {
            return ComparisonResult::fail(
                item_type + " names in the graph don't match: " + lhs.name() + " vs " + rhs.name());
        }

        const auto& lhs_tensor = lhs.type().tensor_type();
        const auto& rhs_tensor = rhs.type().tensor_type();
        if (lhs_tensor.elem_type() != rhs_tensor.elem_type())
        {
            return ComparisonResult::fail("Element types don't match for " + item_type + "  " +
                                          lhs.name() + ": " +
                                          std::to_string(lhs_tensor.elem_type()) + " vs " +
                                          std::to_string(rhs_tensor.elem_type()));
        }

        const auto& lhs_shape = lhs_tensor.shape();
        const auto& rhs_shape = rhs_tensor.shape();
        if (lhs_shape.dim_size() != rhs_shape.dim_size())
        {
            return ComparisonResult::fail("Tensor ranks don't match for " + item_type + " " +
                                          lhs.name() + ": " + std::to_string(lhs_shape.dim_size()) +
                                          " vs " + std::to_string(rhs_shape.dim_size()));
        }
        else
        {
            for (int j = 0; j < lhs_shape.dim_size(); ++j)
            {
                const auto& lhs_dim = lhs_shape.dim(j);
                const auto& rhs_dim = rhs_shape.dim(j);
                if ((lhs_dim.has_dim_value() && rhs_dim.has_dim_param()) ||
                    (rhs_dim.has_dim_value() && lhs_dim.has_dim_param()))
                {
                    return ComparisonResult::fail("Dynamic vs static dimension mismatch for " +
                                                  item_type + " " + lhs.name() +
                                                  " at index: " + std::to_string(j));
                }
                else if (lhs_dim.has_dim_value() && lhs_dim.dim_value() != rhs_dim.dim_value())
                {
                    return ComparisonResult::fail("Shape dimensions don't match for " + item_type +
                                                  " " + lhs.name() +
                                                  " at index: " + std::to_string(j) + ". " +
                                                  std::to_string(lhs_dim.dim_value()) + " vs " +
                                                  std::to_string(rhs_dim.dim_value()));
                }
            }
        }

        return ComparisonResult::pass();
    }

    ComparisonResult compare_inputs(const ONNX_NAMESPACE::GraphProto& graph,
                                    const ONNX_NAMESPACE::GraphProto& ref_graph)
    {
        if (graph.input_size() != ref_graph.input_size())
        {
            return ComparisonResult::fail(
                "The number of inputs in compared models doesn't match: " +
                std::to_string(graph.input_size()) + " vs " +
                std::to_string(ref_graph.input_size()));
        }
        else
        {
            for (int i = 0; i < graph.input_size(); ++i)
            {
                const auto& lhs = graph.input(i);
                const auto& rhs = ref_graph.input(i);

                const auto res = compare_value_info(lhs, rhs, "input");
                if (!res.is_ok)
                {
                    return res;
                }
            }

            return ComparisonResult::pass();
        }
    }

    ComparisonResult compare_outputs(const ONNX_NAMESPACE::GraphProto& graph,
                                     const ONNX_NAMESPACE::GraphProto& ref_graph)
    {
        if (graph.output_size() != ref_graph.output_size())
        {
            return ComparisonResult::fail("The number of outputs in compared models doesn't match" +
                                          std::to_string(graph.output_size()) + " vs " +
                                          std::to_string(ref_graph.output_size()));
        }
        else
        {
            for (int i = 0; i < graph.output_size(); ++i)
            {
                const auto& lhs = graph.output(i);
                const auto& rhs = ref_graph.output(i);

                const auto res = compare_value_info(lhs, rhs, "output");
                if (!res.is_ok)
                {
                    return res;
                }
            }

            return ComparisonResult::pass();
        }
    }

    ComparisonResult compare_initializers(const ONNX_NAMESPACE::GraphProto& graph,
                                          const ONNX_NAMESPACE::GraphProto& ref_graph)
    {
        if (graph.initializer_size() != ref_graph.initializer_size())
        {
            return ComparisonResult::fail(
                "The number of initializers in compared models doesn't match" +
                std::to_string(graph.initializer_size()) + " vs " +
                std::to_string(ref_graph.initializer_size()));
        }
        else
        {
            for (int i = 0; i < graph.initializer_size(); ++i)
            {
                const auto& lhs = graph.initializer(i);
                const auto& rhs = ref_graph.initializer(i);

                if (lhs.name() != rhs.name())
                {
                    return ComparisonResult::fail("Initializer names in the graph don't match: " +
                                                  lhs.name() + " vs " + rhs.name());
                }
                else if (lhs.data_type() != rhs.data_type())
                {
                    return ComparisonResult::fail(
                        "Initializer data types in the graph don't match: " +
                        std::to_string(lhs.data_type()) + " vs " + std::to_string(rhs.data_type()));
                }
                else if (lhs.dims_size() != rhs.dims_size())
                {
                    return ComparisonResult::fail("Initializer ranks in the graph don't match: " +
                                                  std::to_string(lhs.dims_size()) + " vs " +
                                                  std::to_string(rhs.dims_size()));
                }
                else
                {
                    for (int j = 0; j < lhs.dims_size(); ++j)
                    {
                        if (lhs.dims(j) != rhs.dims(j))
                        {
                            return ComparisonResult::fail(
                                "Shape dimensions don't match for initializer " + lhs.name() +
                                " at index: " + std::to_string(j) + ". " +
                                std::to_string(lhs.dims(j)) + " vs " + std::to_string(rhs.dims(j)));
                        }
                    }
                }
            }

            return ComparisonResult::pass();
        }
    }

    ComparisonResult compare_onnx_graphs(const ONNX_NAMESPACE::GraphProto& graph,
                                         const ONNX_NAMESPACE::GraphProto& ref_graph)
    {
        ComparisonResult comparison = compare_inputs(graph, ref_graph);
        if (!comparison.is_ok)
        {
            return comparison;
        }

        comparison = compare_outputs(graph, ref_graph);
        if (!comparison.is_ok)
        {
            return comparison;
        }

        comparison = compare_initializers(graph, ref_graph);
        if (!comparison.is_ok)
        {
            return comparison;
        }

        return compare_nodes(graph, ref_graph);
    }
} // namespace
namespace ngraph
{
    namespace test
    {
        ComparisonResult compare_onnx_models(const std::string& model,
                                             const std::string& reference_model_path)
        {
            std::stringstream model_stream{model};
            ONNX_NAMESPACE::ModelProto model_proto, ref_model;
            parse_from_istream(model_stream, model_proto);
            parse_from_file(reference_model_path, ref_model);
            return compare_onnx_graphs(model_proto.graph(), ref_model.graph());
        }
    } // namespace test
} // namespace ngraph
