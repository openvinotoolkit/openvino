#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include "composite_tssn.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace ov;

// Helper to convert dense weights to sparse format
void dense_to_sparse(const std::shared_ptr<ov::op::v0::Constant>& weight_node, 
                     std::shared_ptr<Node>& indices_node,
                     std::shared_ptr<Node>& values_node,
                     std::shared_ptr<Node>& sensitivity_node,
                     std::shared_ptr<Node>& counts_node,
                     std::shared_ptr<Node>& starts_node,
                     std::shared_ptr<Node>& func_ids_node,
                     float sparsity_target = 0.1,
                     bool transpose = false) {
    
    auto weight_data = weight_node->cast_vector<float>();
    auto shape = weight_node->get_shape();
    
    // MatMul weights are usually [In, Out] if transpose_b=False
    // Or [Out, In] if transpose_b=True
    // We want to iterate such that we find inputs for each output.
    
    size_t out_dim, in_dim;
    if (transpose) {
        // Weights are [Out, In]
        out_dim = shape[0];
        in_dim = shape[1];
    } else {
        // Weights are [In, Out]
        in_dim = shape[0];
        out_dim = shape[1];
    }
    
    // Simple magnitude pruning
    std::vector<float> abs_weights = weight_data;
    std::transform(abs_weights.begin(), abs_weights.end(), abs_weights.begin(), [](float v){ return std::abs(v); });
    
    // Find threshold
    std::vector<float> sorted_abs = abs_weights;
    std::sort(sorted_abs.begin(), sorted_abs.end());
    size_t threshold_idx = (size_t)(sorted_abs.size() * sparsity_target);
    float threshold = sorted_abs[threshold_idx];
    
    struct Element {
        int32_t r, c; // r=Input, c=Output
        float val;
    };
    std::vector<Element> elements;
    
    for (size_t i = 0; i < weight_data.size(); ++i) {
        if (std::abs(weight_data[i]) >= threshold) {
            size_t r, c;
            if (transpose) {
                // [Out, In] -> i = out * In + in
                c = i / in_dim; // Output
                r = i % in_dim; // Input
            } else {
                // [In, Out] -> i = in * Out + out
                r = i / out_dim; // Input
                c = i % out_dim; // Output
            }
            elements.push_back({(int32_t)r, (int32_t)c, weight_data[i]});
        }
    }
    
    // Sort by Output Index (c) for CSC
    std::sort(elements.begin(), elements.end(), [](const Element& a, const Element& b) {
        if (a.c != b.c) return a.c < b.c;
        return a.r < b.r;
    });
    
    size_t n_synapses = elements.size();
    std::vector<int32_t> indices_flat(2 * n_synapses);
    std::vector<float> values_flat(n_synapses);
    std::vector<float> sensitivity_flat(n_synapses);
    
    for (size_t i = 0; i < n_synapses; ++i) {
        indices_flat[i] = elements[i].r;              // Row 0: Input Index
        indices_flat[i + n_synapses] = elements[i].c; // Row 1: Output Index
        
        // Ternary Quantization
        float val = elements[i].val;
        float sign = (val > 0) ? 1.0f : -1.0f;
        values_flat[i] = sign;
        sensitivity_flat[i] = std::abs(val);
    }
    
    // Compute Counts and Starts
    std::vector<int32_t> counts(out_dim, 0);
    for (const auto& el : elements) {
        if (el.c < out_dim) counts[el.c]++;
    }
    
    std::vector<int32_t> starts(out_dim, 0);
    int32_t current_start = 0;
    for (size_t i = 0; i < out_dim; ++i) {
        starts[i] = current_start;
        current_start += counts[i];
    }
    
    // Generate Function IDs (Random 0-4)
    std::vector<int32_t> func_ids(out_dim);
    for (size_t i = 0; i < out_dim; ++i) {
        func_ids[i] = rand() % 5;
    }
    
    indices_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2, n_synapses}, indices_flat);
    values_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{n_synapses}, values_flat);
    sensitivity_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{n_synapses}, sensitivity_flat);
    counts_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{out_dim}, counts);
    starts_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{out_dim}, starts);
    func_ids_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{out_dim}, func_ids);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: apply_incision_tool <input_model.xml> <output_model.xml> [target_layer_substring] [sparsity]" << std::endl;
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string target_layer = (argc > 3) ? argv[3] : "down_proj";
    float sparsity = (argc > 4) ? std::stof(argv[4]) : 0.1f;
    
    Core core;
    
    std::cout << "Reading model from " << input_path << "..." << std::endl;
    std::shared_ptr<Model> model = core.read_model(input_path);
    
    int replaced_count = 0;
    
    // Iterate over ops. Note: modifying the graph while iterating can be tricky.
    // Better to collect ops to replace first.
    std::vector<std::shared_ptr<Node>> ops_to_replace;
    
    for (auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("MatMul")) {
            std::string name = op->get_friendly_name();
            if (name.find(target_layer) != std::string::npos) {
                ops_to_replace.push_back(op);
            }
        }
    }
    
    for (auto& op : ops_to_replace) {
        std::string name = op->get_friendly_name();
        std::cout << "Replacing layer: " << name << " with sparsity " << sparsity << std::endl;
        
        // Get inputs
        auto input_node = op->input_value(0);
        auto weight_node_output = op->input_value(1);
        auto node_ptr = weight_node_output.get_node_shared_ptr();
        
        // Look through Convert
        if (node_ptr->get_type_name() == std::string("Convert")) {
            weight_node_output = node_ptr->input_value(0);
            node_ptr = weight_node_output.get_node_shared_ptr();
        }
        
        auto weight_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node_ptr);
        
        if (!weight_node) {
            std::cout << "Skipping " << name << ": Weights are not Constant. Type: " << node_ptr->get_type_name() << std::endl;
            continue;
        }
        
        // Check transpose_b attribute
        bool transpose_b = false;
        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(op);
        if (matmul) {
            transpose_b = matmul->get_transpose_b();
        }
        
        std::shared_ptr<Node> indices, values, sensitivity, counts, starts, func_ids_node;
        dense_to_sparse(weight_node, indices, values, sensitivity, counts, starts, func_ids_node, sparsity, transpose_b);
        
        // Create CompositeTSSN
        // Output dim is the size of counts
        size_t output_dim = counts->get_shape()[0];
        std::vector<int64_t> func_ids_vec = {}; // Empty for now
        
        auto tssn = std::make_shared<ov::op::v0::CompositeTSSN>(
            input_node,
            indices,
            values,
            sensitivity,
            counts,
            starts,
            func_ids_node,
            output_dim,
            func_ids_vec
        );
        
        tssn->set_friendly_name(name + "_tssn");
        
        // Replace
        replace_node(op, tssn);
        replaced_count++;
    }
    
    std::cout << "Replaced " << replaced_count << " layers." << std::endl;
    
    ov::save_model(model, output_path);
    std::cout << "Saved model to " << output_path << std::endl;
    
    return 0;
}
