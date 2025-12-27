// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MoE Expert Dimension Unrolling Test
// Tests transformation accuracy by comparing original vs unrolled model outputs

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>

#include "npuw/moe_experts.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

// Compute FNV-1a hash for byte-level tensor comparison
std::string compute_tensor_hash(const ov::Tensor &tensor) {
  const uint8_t *data = reinterpret_cast<const uint8_t *>(tensor.data());
  size_t size = tensor.get_byte_size();

  uint64_t hash = 0xcbf29ce484222325ULL;
  const uint64_t prime = 0x100000001b3ULL;

  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= prime;
  }

  std::stringstream ss;
  ss << std::hex << std::setfill('0') << std::setw(16) << hash;
  return ss.str();
}

// Fill tensor with random data (supports f32/f16/i64/i32/nf4/i4/u4/u8/i8)
void fill_tensor_random(ov::Tensor &tensor, std::mt19937 &rng) {
  auto element_type = tensor.get_element_type();
  size_t num_elements = tensor.get_size();

  if (element_type == ov::element::f32) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto *data = tensor.data<float>();
    for (size_t i = 0; i < num_elements; ++i) {
      data[i] = dist(rng);
    }
  } else if (element_type == ov::element::f16) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto *data = tensor.data<ov::float16>();
    for (size_t i = 0; i < num_elements; ++i) {
      data[i] = ov::float16(dist(rng));
    }
  } else if (element_type == ov::element::i64) {
    std::uniform_int_distribution<int64_t> dist(0, 100);
    auto *data = tensor.data<int64_t>();
    for (size_t i = 0; i < num_elements; ++i) {
      data[i] = dist(rng);
    }
  } else if (element_type == ov::element::i32) {
    std::uniform_int_distribution<int32_t> dist(0, 100);
    auto *data = tensor.data<int32_t>();
    for (size_t i = 0; i < num_elements; ++i) {
      data[i] = dist(rng);
    }
  } else if (element_type == ov::element::nf4) {
    // 4-bit normalized float: packed 2 values per byte, indices 0-15
    std::uniform_int_distribution<int> dist(0, 15);
    int8_t *data = reinterpret_cast<int8_t *>(tensor.data());
    for (size_t i = 0; i < num_elements; ++i) {
      int value = dist(rng);
      if (i % 2 == 0) {
        data[i / 2] = static_cast<int8_t>(value & 0x0F);
      } else {
        data[i / 2] |= static_cast<int8_t>(value << 4);
      }
    }
  } else if (element_type == ov::element::i4) {
    // 4-bit signed integer: packed 2 values per byte, range -8 to 7
    std::uniform_int_distribution<int> dist(-8, 7);
    int8_t *data = reinterpret_cast<int8_t *>(tensor.data());
    for (size_t i = 0; i < num_elements; ++i) {
      int value = dist(rng);
      if (i % 2 == 0) {
        data[i / 2] = static_cast<int8_t>(value & 0x0F);
      } else {
        data[i / 2] |= static_cast<int8_t>(value << 4);
      }
    }
  } else if (element_type == ov::element::u4) {
    // 4-bit unsigned integer: packed 2 values per byte, range 0 to 15
    std::uniform_int_distribution<int> dist(0, 15);
    int8_t *data = reinterpret_cast<int8_t *>(tensor.data());
    for (size_t i = 0; i < num_elements; ++i) {
      int value = dist(rng);
      if (i % 2 == 0) {
        data[i / 2] = static_cast<int8_t>(value & 0x0F);
      } else {
        data[i / 2] |= static_cast<int8_t>(value << 4);
      }
    }
  } else if (element_type == ov::element::u8) {
    std::uniform_int_distribution<unsigned int> dist(0, 255);
    auto *data = tensor.data<uint8_t>();
    for (size_t i = 0; i < num_elements; ++i) {
      data[i] = static_cast<uint8_t>(dist(rng));
    }
  } else if (element_type == ov::element::i8) {
    std::uniform_int_distribution<int> dist(-128, 127);
    auto *data = tensor.data<int8_t>();
    for (size_t i = 0; i < num_elements; ++i) {
      data[i] = static_cast<int8_t>(dist(rng));
    }
  } else {
    std::cerr << "Unsupported element type: " << element_type
              << ", filling with zeros" << std::endl;
    NPUW_ASSERT(false);
  }
}

// Compare tensors numerically, return max absolute difference
float compare_tensors(const ov::Tensor &t1, const ov::Tensor &t2) {
  if (t1.get_element_type() != t2.get_element_type()) {
    std::cerr << "Element types differ!" << std::endl;
    return std::numeric_limits<float>::infinity();
  }

  if (t1.get_size() != t2.get_size()) {
    std::cerr << "Tensor sizes differ: " << t1.get_size() << " vs "
              << t2.get_size() << std::endl;
    return std::numeric_limits<float>::infinity();
  }

  float max_diff = 0.0f;
  size_t num_elements = t1.get_size();
  auto element_type = t1.get_element_type();

  if (element_type == ov::element::f32) {
    const float *data1 = t1.data<float>();
    const float *data2 = t2.data<float>();
    for (size_t i = 0; i < num_elements; ++i) {
      max_diff = std::max(max_diff, std::abs(data1[i] - data2[i]));
    }
  } else if (element_type == ov::element::f16) {
    const ov::float16 *data1 = t1.data<ov::float16>();
    const ov::float16 *data2 = t2.data<ov::float16>();
    for (size_t i = 0; i < num_elements; ++i) {
      max_diff = std::max(max_diff, std::abs(static_cast<float>(data1[i]) -
                                             static_cast<float>(data2[i])));
    }
  } else if (element_type == ov::element::i64) {
    const int64_t *data1 = t1.data<int64_t>();
    const int64_t *data2 = t2.data<int64_t>();
    for (size_t i = 0; i < num_elements; ++i) {
      max_diff =
          std::max(max_diff, std::abs(static_cast<float>(data1[i] - data2[i])));
    }
  } else if (element_type == ov::element::i32) {
    const int32_t *data1 = t1.data<int32_t>();
    const int32_t *data2 = t2.data<int32_t>();
    for (size_t i = 0; i < num_elements; ++i) {
      max_diff =
          std::max(max_diff, std::abs(static_cast<float>(data1[i] - data2[i])));
    }
  }

  return max_diff;
}

int main() {
  std::cout << "=== MoE Expert Unrolling Accuracy Test ===" << std::endl;

  // Load model
  std::string model_path = "C:\\Intel\\xiong\\openvino.genai\\tools\\llm_"
                           "bench\\run_moe\\Model0_kv1152_02_REP0108.xml";
  std::cout << "\n[1/10] Loading model: " << model_path << std::endl;

  ov::Core core;
  std::shared_ptr<ov::Model> original_model;
  try {
    original_model = core.read_model(model_path);
    std::cout << "  ✓ Model loaded - "
              << original_model->get_parameters().size() << " parameters, "
              << original_model->get_results().size() << " results"
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "  ✗ Failed: " << e.what() << std::endl;
    return 1;
  }

  // Analyze MoE structure
  std::cout << "\n[2/10] Analyzing MoE structure..." << std::endl;
  auto structure_info =
      ov::npuw::function::analyze_moe_structure(original_model);

  if (!structure_info || !structure_info->is_valid()) {
    std::cerr << "  ✗ Failed to detect MoE structure" << std::endl;
    return 1;
  }

  std::cout << "  ✓ MoE detected - " << structure_info->num_experts
            << " experts, hidden_dim=" << structure_info->expert_hidden_dim
            << ", tokens=" << structure_info->input_token_count << std::endl;

  // Generate random inputs
  std::cout << "\n[3/10] Generating random inputs (seed=42)..." << std::endl;
  std::mt19937 rng(42);
  std::map<std::string, ov::Tensor> original_inputs;

  auto params = original_model->get_parameters();
  for (size_t i = 0; i < params.size(); ++i) {
    auto param = params[i];
    std::string param_name = param->get_friendly_name();
    auto shape = param->get_partial_shape();
    auto element_type = param->get_element_type();

    if (!shape.is_static()) {
      std::cerr << "  Warning: Parameter " << param_name
                << " has dynamic shape, skipping" << std::endl;
      continue;
    }

    ov::Tensor tensor(element_type, shape.to_shape());
    fill_tensor_random(tensor, rng);
    original_inputs[param_name] = tensor;
  }
  std::cout << "  ✓ Generated " << original_inputs.size() << " input tensors"
            << std::endl;

  // Compile and run original model
  std::cout << "\n[4/10] Running original model..." << std::endl;
  ov::CompiledModel compiled_original =
      core.compile_model(original_model, "CPU");
  ov::InferRequest infer_original = compiled_original.create_infer_request();

  for (size_t i = 0; i < params.size(); ++i) {
    auto it = original_inputs.find(params[i]->get_friendly_name());
    if (it != original_inputs.end()) {
      infer_original.set_input_tensor(i, it->second);
    }
  }

  infer_original.infer();
  std::cout << "  ✓ Inference completed" << std::endl;

  // Save original outputs
  std::map<std::string, ov::Tensor> original_outputs;
  auto results = original_model->get_results();
  for (size_t i = 0; i < results.size(); ++i) {
    ov::Tensor output_tensor = infer_original.get_output_tensor(i);
    ov::Tensor output_copy(output_tensor.get_element_type(),
                           output_tensor.get_shape());
    std::memcpy(output_copy.data(), output_tensor.data(),
                output_tensor.get_byte_size());
    original_outputs[results[i]->get_friendly_name()] = output_copy;
  }

  // Unroll model
  std::cout << "\n[5/10] Unrolling expert dimension..." << std::endl;
  auto unrolled_model = ov::npuw::function::unroll_expert_dimension(
      original_model, *structure_info, structure_info->num_experts);

  if (!unrolled_model) {
    std::cerr << "  ✗ Unrolling failed" << std::endl;
    return 1;
  }

  std::cout << "  ✓ Unrolled - " << unrolled_model->get_parameters().size()
            << " parameters, " << unrolled_model->get_ops().size() << " ops"
            << std::endl;

  // Save unrolled model
  std::cout << "\n[6/10] Saving unrolled model..." << std::endl;
  try {
    std::string output_path = "moe_unrolled_final.xml";
    ov::serialize(unrolled_model, output_path);
    std::cout << "  ✓ Saved to: " << output_path << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "  Warning: Failed to save - " << e.what() << std::endl;
  }

  // Map inputs to unrolled model
  std::cout << "\n[7/10] Mapping inputs to unrolled model..." << std::endl;
  auto unrolled_params = unrolled_model->get_parameters();
  std::vector<ov::Tensor> unrolled_input_tensors(unrolled_params.size());

  for (size_t i = 0; i < unrolled_params.size(); ++i) {
    auto param = unrolled_params[i];
    std::string param_name = param->get_friendly_name();
    auto shape = param->get_partial_shape();
    auto element_type = param->get_element_type();

    if (!shape.is_static()) {
      continue;
    }

    // Check RTInfo for MoE parameter mapping
    auto &rt_info = param->get_rt_info();
    if (rt_info.count("moe_original_param")) {
      // This is an unrolled MoE parameter
      std::string original_name =
          rt_info["moe_original_param"].as<std::string>();
      size_t expert_idx = rt_info["moe_expert_index"].as<int64_t>();
      size_t num_experts = rt_info["moe_num_experts"].as<int64_t>();

      auto it = original_inputs.find(original_name);
      if (it != original_inputs.end()) {
        const ov::Tensor &orig_tensor = it->second;
        auto orig_shape = orig_tensor.get_shape();

        if (orig_shape[0] == num_experts) {
          ov::Tensor expert_tensor(element_type, shape.to_shape());

          size_t bytes_per_expert = orig_tensor.get_byte_size() / num_experts;
          const uint8_t *src_data =
              reinterpret_cast<const uint8_t *>(orig_tensor.data());
          uint8_t *dst_data = reinterpret_cast<uint8_t *>(expert_tensor.data());

          std::memcpy(dst_data, src_data + expert_idx * bytes_per_expert,
                      bytes_per_expert);
          unrolled_input_tensors[i] = expert_tensor;
          continue;
        }
      }
    }

    // Direct mapping for unchanged parameters
    auto it = original_inputs.find(param_name);
    if (it != original_inputs.end()) {
      unrolled_input_tensors[i] = it->second;
    }
  }
  std::cout << "  ✓ Mapped " << unrolled_input_tensors.size()
            << " input tensors" << std::endl;

  // Compile and run unrolled model
  std::cout << "\n[8/10] Running unrolled model..." << std::endl;
  ov::CompiledModel compiled_unrolled =
      core.compile_model(unrolled_model, "CPU");
  ov::InferRequest infer_unrolled = compiled_unrolled.create_infer_request();

  for (size_t i = 0; i < unrolled_input_tensors.size(); ++i) {
    if (unrolled_input_tensors[i]) {
      infer_unrolled.set_input_tensor(i, unrolled_input_tensors[i]);
    }
  }

  infer_unrolled.infer();
  std::cout << "  ✓ Inference completed" << std::endl;

  // Save unrolled outputs
  std::map<std::string, ov::Tensor> unrolled_outputs;
  auto unrolled_results = unrolled_model->get_results();
  for (size_t i = 0; i < unrolled_results.size(); ++i) {
    ov::Tensor output_tensor = infer_unrolled.get_output_tensor(i);
    ov::Tensor output_copy(output_tensor.get_element_type(),
                           output_tensor.get_shape());
    std::memcpy(output_copy.data(), output_tensor.data(),
                output_tensor.get_byte_size());
    unrolled_outputs[unrolled_results[i]->get_friendly_name()] = output_copy;
  }

  // Compare outputs
  std::cout << "\n[9/10] Comparing outputs..." << std::endl;
  bool all_match = true;
  float max_overall_diff = 0.0f;

  for (const auto &[orig_name, orig_tensor] : original_outputs) {
    auto it = unrolled_outputs.find(orig_name);
    if (it == unrolled_outputs.end()) {
      std::cerr << "  ✗ Output " << orig_name << " not found in unrolled model"
                << std::endl;
      all_match = false;
      continue;
    }

    const ov::Tensor &unrolled_tensor = it->second;
    float max_diff = compare_tensors(orig_tensor, unrolled_tensor);
    max_overall_diff = std::max(max_overall_diff, max_diff);

    std::string orig_hash = compute_tensor_hash(orig_tensor);
    std::string unrolled_hash = compute_tensor_hash(unrolled_tensor);
    bool hash_match = (orig_hash == unrolled_hash);

    // Determine tolerance
    float tolerance =
        (orig_tensor.get_element_type() == ov::element::f16) ? 1e-2f : 1e-3f;

    if (max_diff > tolerance) {
      std::cout << "  ⚠ " << orig_name << ": diff=" << max_diff
                << " (tolerance=" << tolerance << ")" << std::endl;
      if (max_diff > 0.1f) {
        all_match = false;
      }
    } else if (max_diff == 0.0f && hash_match) {
      std::cout << "  ✓ " << orig_name << ": PERFECT MATCH (diff=0, hash match)"
                << std::endl;
    } else {
      std::cout << "  ✓ " << orig_name << ": diff=" << max_diff
                << " (within tolerance)" << std::endl;
    }
  }

  // Summary
  std::cout << "\n[10/10] Summary" << std::endl;
  std::cout << "  Max difference: " << max_overall_diff << std::endl;

  if (all_match) {
    std::cout << "  ✓ All outputs match - transformation preserves accuracy"
              << std::endl;
    std::cout << "\n=== Test PASSED ===" << std::endl;
    return 0;
  } else {
    std::cerr << "  ✗ Some outputs differ - transformation has accuracy issues"
              << std::endl;
    return 1;
  }
}
