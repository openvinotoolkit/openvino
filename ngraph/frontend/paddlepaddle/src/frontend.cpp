// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

#include "framework.pb.h"

#include "../include/paddlepaddle_frontend/model.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset6.hpp>

#include "utility.hpp"
#include "decoder.hpp"



namespace ngraph {
    namespace frontend {


typedef std::shared_ptr<ngraph::Node>(*CreatorFunction)(
        std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> &inputs,
        const DecoderPDPDProto &);

template<typename T>
void print(const T &a) {
    std::cerr << "[";
    for (const auto &e: a) {
        std::cerr << e << ", ";
    }
    std::cerr << "]\n";
}

std::shared_ptr<ngraph::Node>
conv2d_creator(std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> &inputs,
               const DecoderPDPDProto &op) {
    std::cout << "Running conv2d creator" << std::endl;
    MY_ASSERT(inputs["Input"].size() == 1 && inputs["Filter"].size() == 1, "More then one input for conv2d");
    MY_ASSERT(inputs["Bias"].size() == 0 && inputs["ResidualData"].size() == 0,
              "Bias and residual have input for conv2d");
    auto data = inputs["Input"][0];
    auto filter = inputs["Filter"][0];
    // TODO: resolve padding according to spec
    auto strides = op.get_ints("strides");
    auto paddings = op.get_ints("paddings");
    auto dilations = op.get_ints("dilations");
    std::cout << "Creating convolution node" << std::endl;
    print(strides);
    print(paddings);
    print(dilations);
    return std::make_shared<ngraph::opset6::Convolution>(data,
                                                         filter,
                                                         ngraph::Strides(strides.begin(), strides.end()),
                                                         ngraph::CoordinateDiff(paddings.begin(),
                                                                                paddings.end()),
                                                         ngraph::CoordinateDiff(paddings.begin(),
                                                                                paddings.end()),
                                                         ngraph::Strides(dilations.begin(), dilations.end()));
}


std::shared_ptr<ngraph::Node>
batch_norm_creator(std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> &inputs,
                   const DecoderPDPDProto &op) {
    MY_ASSERT(inputs["X"].size() == 1 &&
              inputs["Scale"].size() == 1 &&
              inputs["Bias"].size() == 1 &&
              inputs["Mean"].size() == 1 &&
              inputs["Variance"].size() == 1,
              "More then one input for batch_norm");
    auto data = inputs["X"][0];
    auto gamma = inputs["Scale"][0];
    auto beta = inputs["Bias"][0];
    auto mean = inputs["Mean"][0];
    auto variance = inputs["Variance"][0];
    return std::make_shared<ngraph::opset6::BatchNormInference>(data, gamma, beta, mean, variance,
                                                                op.get_float("epsilon"));
}


std::shared_ptr<ngraph::Node>
relu_creator(std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> &inputs,
             const DecoderPDPDProto &op) {
    MY_ASSERT(inputs["X"].size() == 1, "More then one input for relu");
    auto data = inputs["X"][0];
    return std::make_shared<ngraph::opset6::Relu>(data);
}

std::shared_ptr<ngraph::Node>
pool2d_creator(std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> &inputs,
               const DecoderPDPDProto &op) {
    MY_ASSERT(inputs["X"].size() == 1, "More then one input for pool2d");
    auto data = inputs["X"][0];
    // TODO : resolve padding according to spec
    auto pooling_type = op.get_str("pooling_type");
    auto global_pooling = op.get_bool("global_pooling");
    if (pooling_type == "max" && !global_pooling) {
        auto strides = op.get_ints("strides");
        auto paddings = op.get_ints("paddings");
        auto kernel_shape = op.get_ints("ksize");
        return std::make_shared<ngraph::opset6::MaxPool>(data,
                                                         ngraph::Strides(strides.begin(), strides.end()),
                                                         ngraph::Shape(paddings.begin(), paddings.end()),
                                                         ngraph::Shape(paddings.begin(), paddings.end()),
                                                         ngraph::Shape(kernel_shape.begin(),
                                                                       kernel_shape.end()));
    } else if (pooling_type == "avg" && global_pooling) {
        // TODO : resolve axes according to rank
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i64, {2}, {2, 3});
        return std::make_shared<ngraph::opset6::ReduceMean>(data, axes, true);
    } else {
        throw std::runtime_error("Unsupported pooling type");
    }
}

std::shared_ptr<ngraph::Node>
elementwise_add_creator(std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> &inputs,
                        const DecoderPDPDProto &op) {
    MY_ASSERT(inputs["X"].size() == 1 && inputs["Y"].size() == 1, "More then one input for elementwise_add");
    auto x = inputs["X"][0];
    auto y = inputs["Y"][0];
    // TODO : resolve broadcast
    return std::make_shared<ngraph::opset6::Add>(x, y);
}

std::shared_ptr<ngraph::Node>
mul_creator(std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> &inputs,
            const DecoderPDPDProto &op) {
    MY_ASSERT(inputs["X"].size() == 1 && inputs["Y"].size() == 1, "More then one input for mul");
    auto x = inputs["X"][0];
    auto y = inputs["Y"][0];
    MY_ASSERT(x->output(0).get_partial_shape().rank().is_static());
    int64_t x_rank = x->output(0).get_partial_shape().rank().get_length();
    MY_ASSERT(y->output(0).get_partial_shape().rank().is_static() &&
              y->output(0).get_partial_shape().rank().get_length() == 2);
    if (x_rank > 2) {
        auto shape = std::make_shared<ngraph::opset6::ShapeOf>(x);
        int64_t x_num_col_dims = op.get_int("x_num_col_dims");
        auto axis = ngraph::opset6::Constant::create(ngraph::element::i64, {}, {0});
        auto split_lengths = ngraph::opset6::Constant::create(ngraph::element::i64, {2},
                                                              {x_num_col_dims, x_rank - x_num_col_dims});
        auto split = std::make_shared<ngraph::opset6::VariadicSplit>(shape, axis, split_lengths);
        auto f_dim_red_axis = ngraph::opset6::Constant::create(ngraph::element::i64, {}, {0});
        auto first_dim_reduce = std::make_shared<ngraph::opset6::ReduceProd>(split->output(0), f_dim_red_axis);
        auto f_dim_shape = ngraph::opset6::Constant::create(ngraph::element::i64, {1}, {1});
        auto first_dim = std::make_shared<ngraph::opset6::Reshape>(first_dim_reduce, f_dim_shape, false);
        auto s_dim_red_axis = ngraph::opset6::Constant::create(ngraph::element::i64, {}, {0});
        auto second_dim_reduce = std::make_shared<ngraph::opset6::ReduceProd>(split->output(1), s_dim_red_axis);
        auto s_dim_shape = ngraph::opset6::Constant::create(ngraph::element::i64, {1}, {1});
        auto second_dim = std::make_shared<ngraph::opset6::Reshape>(second_dim_reduce, s_dim_shape, false);
        auto out_shape = std::make_shared<ngraph::opset6::Concat>(ngraph::NodeVector{first_dim, second_dim}, 0);
        auto x_reshaped = std::make_shared<ngraph::opset6::Reshape>(x, out_shape, false);
        return std::make_shared<ngraph::opset6::MatMul>(x_reshaped, y);
    }
    return std::make_shared<ngraph::opset6::MatMul>(x, y);
}

std::shared_ptr<ngraph::Node>
scale_creator(std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> &inputs,
              const DecoderPDPDProto &op) {
    MY_ASSERT(inputs["X"].size() == 1, "More then one input for scale");
    auto data = inputs["X"][0];
    auto scale = ngraph::opset6::Constant::create(ngraph::element::f32, {1}, {op.get_float("scale")});
    return std::make_shared<ngraph::opset6::Multiply>(data, scale);
}

std::shared_ptr<ngraph::Node>
make_ng_node(std::map<std::string, google::protobuf::RepeatedPtrField<std::string>> &inputs,
             std::map<std::string, std::shared_ptr<ngraph::Node>> &nodes,
             const paddle::framework::proto::OpDesc &op,
             const paddle::framework::proto::BlockDesc &block) {
    std::cout << "Making node: " << op.type() << std::endl;
    std::map<std::string, CreatorFunction> CREATORS_MAP = {
            {"conv2d",          conv2d_creator},
            {"batch_norm",      batch_norm_creator},
            {"relu",            relu_creator},
            {"pool2d",          pool2d_creator},
            {"elementwise_add", elementwise_add_creator},
            {"mul",             mul_creator},
            {"scale",           scale_creator}
    };
    MY_ASSERT(CREATORS_MAP.find(op.type()) != CREATORS_MAP.end(), "No creator found");
    std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> inputs_preproc;
    for (const auto &item : inputs) {
        inputs_preproc[item.first] = std::vector<std::shared_ptr<ngraph::Node>>();
        for (const auto &input_name : item.second) {
            // TODO: refactor to not search every time
            inputs_preproc[item.first].push_back(nodes[input_name]);
        }
    }
    return CREATORS_MAP[op.type()](inputs_preproc, DecoderPDPDProto(op));
}

std::shared_ptr<ngraph::opset6::Constant>
read_tensor(const paddle::framework::proto::VarDesc &var, const std::string &model_dir) {
    std::cout << "Reading tensor " << var.name() << std::endl;
    MY_ASSERT(var.type().type() == paddle::framework::proto::VarType::LOD_TENSOR);
    auto tensor = var.type().lod_tensor().tensor();

    std::ifstream is(model_dir + "/" + var.name(), std::ios::in | std::ifstream::binary);
    if (!is || !is.is_open()) {
        std::cout << "File not opened" << std::endl;
    }
    // get length of file:
    is.seekg(0, std::ios::end);
    auto length = is.tellg();
    auto tensor_length = std::accumulate(tensor.dims().cbegin(), tensor.dims().cend(), 1,
                                         std::multiplies<int64_t>());
    std::cout << "length: " << length << ", ten_len: " << tensor_length << std::endl;
    is.seekg((size_t) length - tensor_length * 4, std::ios::beg);

    std::vector<float> tensor_data(tensor_length, 0);
    is.read(reinterpret_cast<char *>(&tensor_data[0]), tensor_length * 4);
    is.close();
    auto shape = std::vector<size_t>(tensor.dims().cbegin(), tensor.dims().cend());
    return ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape(shape), tensor_data);
}

bool endsWith(const std::string &str, const std::string &suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    return false;
}

std::shared_ptr<ngraph::Function> convert_model(const std::string &model_dir) {
    std::cout << "Convert Model Start" << std::endl;
    paddle::framework::proto::ProgramDesc fw_model;
    std::ifstream pb_stream(model_dir + "/__model__", std::ios::binary);
    std::cout << "Model Parsed: " << fw_model.ParseFromIstream(&pb_stream) << std::endl;

    std::map<std::string, std::shared_ptr<ngraph::Node>> nodes_dict;
    ngraph::ParameterVector parameter_nodes;
    ngraph::ResultVector result_nodes;

    std::cout << "Blocks number: " << fw_model.blocks().size() << std::endl;
    const auto &global_block = fw_model.blocks()[0];
    for (const auto &var : global_block.vars()) {
        if (endsWith(var.name(), "feed") || endsWith(var.name(), "fetch"))
            continue;
        if (!var.persistable())
            continue;
        nodes_dict[var.name()] = read_tensor(var, model_dir);
    }
    std::cout << "Reading consts finished" << std::endl;

    for (const auto &block : fw_model.blocks()) {
        std::map<std::string, paddle::framework::proto::VarType> vars_dict;
        for (const auto &var : block.vars()) {
            vars_dict[var.name()] = var.type();
        }
        for (int i = 0; i < block.ops_size(); i++) {
            std::cerr << "Observing index i = " << i << "\n";
            const auto &op = block.ops()[i];
            std::cerr << "Observing " << op.type() << "\n";
            std::map<std::string, google::protobuf::RepeatedPtrField<std::string>> outputs_dict;
            for (const auto &output : op.outputs()) {
                outputs_dict[output.parameter()] = output.arguments();
                std::cerr << output.parameter() << "\n";
            }
            std::map<std::string, google::protobuf::RepeatedPtrField<std::string>> inputs_dict;
            for (const auto &input : op.inputs()) {
                inputs_dict[input.parameter()] = input.arguments();
            }
            if (op.type() == "feed") {
                auto layer_name = outputs_dict["Out"][0];
                std::cout << "Creating parameter: " << layer_name << std::endl;
                auto var = vars_dict[layer_name];
                MY_ASSERT(var.type() == paddle::framework::proto::VarType::LOD_TENSOR);
                auto tensor_desc = var.lod_tensor().tensor();
                auto dtype = tensor_desc.data_type();
                std::vector<size_t> shape;
                // set all -1 dims to 1
                for (auto dim : tensor_desc.dims()) {
                    if (dim >= 0) {
                        shape.push_back(dim);
                    } else {
                        shape.push_back(1);
                    }
                }
                auto param = std::make_shared<ngraph::opset6::Parameter>(TYPE_MAP[dtype], ngraph::Shape(shape));
                param->set_friendly_name(layer_name);
                nodes_dict[layer_name] = param;
                parameter_nodes.push_back(param);
                std::cout << "Parameter created" << std::endl;
            } else if (op.type() == "fetch") {
                auto input_node = inputs_dict["X"][0];
                MY_ASSERT(nodes_dict.find(input_node) != nodes_dict.end());
                result_nodes.push_back(std::make_shared<ngraph::opset6::Result>(nodes_dict[input_node]));
            } else {
                auto node = make_ng_node(inputs_dict, nodes_dict, op, block);
                std::cerr << "Node created: " << node << "\n";
                node->set_friendly_name(op.outputs()[0].parameter());
                std::cerr << "Named with " << node->get_friendly_name() << "\n";
                for (const auto &item : outputs_dict) {
                    MY_ASSERT(item.second.size() <= 1);
                    if (item.second.size() == 1) {
                        nodes_dict[item.second[0]] = node;
                    }
                }
            }
        }
    }
    return std::make_shared<ngraph::Function>(result_nodes, parameter_nodes);
}

std::shared_ptr<ngraph::Function> ngraph::frontend::FrontEndPDPD::convert(InputModel::Ptr model) const {
    std::string path = std::dynamic_pointer_cast<ngraph::frontend::InputModelPDPD>(model)->path;
    std::cerr << "[ INFO ] PFrontEndPDPD::convert invoked\n";
    auto f = convert_model(path);
    std::cerr << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << "\n";
    return f;
}

}
}