
#include "openvino/op/rpe.hpp"

#include <sstream>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace v5 {
static constexpr float kTemperature_ort = 10000.0;
static constexpr float kScale_trt = 2.0 * M_PI;

static inline float rot_pos_theta_trt_with_freq(int id, int c_dim_div3, const float* rel_pos, const float* freq) {
    int pos_id = id / c_dim_div3;
    float theta = rel_pos[pos_id] * freq[id];
    return theta;
}

static inline float rot_vec_first_trt(float cos_theta, float sin_theta, const float* ptr) {
    float rot_v = cos_theta * (*ptr) - sin_theta * (*(ptr + 1));
    return rot_v;
}

static inline float rot_vec_second_trt(float cos_theta, float sin_theta, const float* ptr) {
    float rot_v = sin_theta * (*(ptr - 1)) + cos_theta * (*ptr);
    return rot_v;
}

void get_rel_pos_freq(float* pe_freq, int embed_dims, int pos_dim) {
    for (int idx = 0; idx < embed_dims; ++idx) {
        int num_feats = embed_dims / pos_dim;
        int dim = idx % num_feats;
        float f_dim = powf(kTemperature_ort, 2.0 * (dim / 2) / num_feats);
        pe_freq[idx] = kScale_trt / f_dim;
    }
}

void attention_value_computation(int bs,
                                 int total_query_num,
                                 int total_key_num,
                                 int key_index_size,
                                 int nhead,
                                 int hdim,
                                 const float* attn_weight,
                                 const float* value_features,
                                 const int* key_attn_index,
                                 const float* qk_relative_pos,
                                 const float* pe_freq,
                                 float* output) {
    // params   attn_weight_data        bs*nhead  x  total_query_num  x  key_index_size
    // params   value_features          total_query_num  x  bs*nhead  x  hdim
    // params   key_attn_index          bs  x  total_query_num  x  key_index_size
    // params   qk_relative_pos         bs  x  total_query_num  x  total_key_num  x  3
    // params   output:                 total_query_num  x  bs*nhead  x  hdim
    for (int query_idx = 0; query_idx < total_query_num; ++query_idx) {
        for (int bs_head_idx = 0; bs_head_idx < bs * nhead; ++bs_head_idx) {
            for (int hdim_idx = 0; hdim_idx < hdim; ++hdim_idx) {
                int bs_idx = bs_head_idx / nhead;
                int head_idx = bs_head_idx % nhead;
                int value_stride0 = bs * nhead * hdim;

                //
                float* out = output + query_idx * bs * nhead * hdim + bs_head_idx * hdim + hdim_idx;
                int embed_dim_div_3 = nhead * hdim / 3;
                int channel_id = hdim * head_idx + hdim_idx;
                int pos_id = channel_id / embed_dim_div_3;
                // int start_h_id = hdim * head_idx;
                // float two_div_nf = 2.0 / embed_dim_div_3;
                const float* in_rel_pos_base = qk_relative_pos + bs_idx * total_query_num * total_key_num * 3 +
                                               query_idx * total_key_num * 3 + pos_id;
                const float* in_value_base = value_features + bs_head_idx * hdim + hdim_idx;
                int flag = hdim_idx & 0x01;  // hdim_idx % 2;
                float attn_result = 0.0;
                float rot_v = 0.0;
                float (*rot_func)(float, float, const float*);
                if (flag == 0) {
                    rot_func = &rot_vec_first_trt;
                } else {
                    rot_func = &rot_vec_second_trt;
                }
                float theta_freq = pe_freq[channel_id];

                const int* shared_value_index =
                    key_attn_index + bs_idx * total_query_num * key_index_size + query_idx * key_index_size;
                for (int i = 0; i < key_index_size; i++) {
                    int value_idx = shared_value_index[i];
                    if (value_idx < 0 || value_idx >= total_key_num) {
                        continue;
                    }
                    float theta = (*(in_rel_pos_base + value_idx * 3)) * theta_freq;
                    float c_theta = cosf(theta);
                    float s_theta = sinf(theta);
                    const float* value_ptr = in_value_base + value_idx * value_stride0;
                    rot_v = rot_func(c_theta, s_theta, value_ptr);
                    attn_result +=
                        (attn_weight[bs_head_idx * total_query_num * key_index_size + query_idx * key_index_size + i]) *
                        rot_v;
                }

                out[0] = attn_result;
            }
        }
    }
}

void attention_weight_computation(int bs,
                                  int total_query_num,
                                  int total_key_num,
                                  int key_index_size,
                                  int nhead,
                                  int hdim,
                                  const float* query_features,
                                  const float* key_features,
                                  const int* key_attn_index,
                                  const float* qk_relative_pos,
                                  const float* pe_freq,
                                  float* output) {
    // params query_features: [total_query_num, bs * nhead, hdim]
    // params key_features: [total_key_num, bs * nhead, hdim]
    // params key_attn_index: [bs, total_query_num, key_index_size]
    // params qk_relative_pos: [bs, total_query_num, total_key_num, 3]
    // params output: [bs * nhead, total_query_num, key_index_size]

    for (int query_idx = 0; query_idx < total_query_num; ++query_idx) {
        for (int bs_head_idx = 0; bs_head_idx < bs * nhead; ++bs_head_idx) {
            for (int key_index_idx = 0; key_index_idx < key_index_size; ++key_index_idx) {
                int bs_idx = bs_head_idx / nhead;
                int head_idx = bs_head_idx % nhead;

                int bs_head = bs * nhead;
                int qk_stride = bs_head * hdim;
                int start_h_id = hdim * head_idx;

                int key_idx = key_attn_index[bs_idx * total_query_num * key_index_size + query_idx * key_index_size +
                                             key_index_idx];
                float* out = output + bs_head_idx * total_query_num * key_index_size + query_idx * key_index_size +
                             key_index_idx;

                if (key_idx < 0 || key_idx >= total_key_num) {
                    out[0] = -INFINITY;
                    continue;
                }
                // get key features.
                const float* in_key = key_features + key_idx * qk_stride + bs_head_idx * hdim;

                int embed_dim_div_3 = nhead * hdim / 3;

                // float two_div_nf = 2.0 / embed_dim_div_3;
                const float* in_rel_pos = qk_relative_pos + bs_idx * total_query_num * total_key_num * 3 +
                                          query_idx * total_key_num * 3 + key_idx * 3;
                float attn_weight = 0;
                for (int i = 0; i < hdim; i += 2) {
                    float theta = rot_pos_theta_trt_with_freq(start_h_id, embed_dim_div_3, in_rel_pos, pe_freq);
                    // float theta = rot_pos_theta_trt(start_h_id + i, embed_dim_div_3, two_div_nf, in_rel_pos);
                    float c_theta = cosf(theta);
                    float s_theta = sinf(theta);
                    float rot_key0 = c_theta * in_key[i] - s_theta * in_key[i + 1];
                    float rot_key1 = s_theta * in_key[i] + c_theta * in_key[i + 1];
                    attn_weight += (rot_key0 * query_features[query_idx * qk_stride + bs_head_idx * hdim + i] +
                                    rot_key1 * query_features[query_idx * qk_stride + bs_head_idx * hdim + i + 1]);
                    start_h_id += 2;
                }

                out[0] = attn_weight;
            }
        }
    }
}

RotRPEAttentionWeightWithIndexComputation::RotRPEAttentionWeightWithIndexComputation(const Output<Node>& input0,
                                                                                     const Output<Node>& input1,
                                                                                     const Output<Node>& input2,
                                                                                     const Output<Node>& input3)
    : Op({input0, input1, input2, input3}) {
    constructor_validate_and_infer_types();
}

void RotRPEAttentionWeightWithIndexComputation::validate_and_infer_types() {
    OV_OP_SCOPE(v5_RotRPEAttentionWeightWithIndexComputation_validate_and_infer_types);

    // Validate input count
    NODE_VALIDATION_CHECK(this, get_input_size() == 4, "Expected 4 inputs. Got: ", get_input_size());

    // Validate input shapes
    const auto& query_features_shape = get_input_partial_shape(0);
    const auto& key_features_shape = get_input_partial_shape(1);
    const auto& key_attn_index_shape = get_input_partial_shape(2);
    const auto& qk_relative_pos_shape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          query_features_shape.rank().is_static() && query_features_shape.rank() == 3,
                          "query_features must be 3-dimensional");
    NODE_VALIDATION_CHECK(this,
                          key_features_shape.rank().is_static() && key_features_shape.rank() == 3,
                          "key_features must be 3-dimensional");
    NODE_VALIDATION_CHECK(this,
                          key_attn_index_shape.rank().is_static() && key_attn_index_shape.rank() == 3,
                          "key_attn_index must be 3-dimensional");
    NODE_VALIDATION_CHECK(this,
                          qk_relative_pos_shape.rank().is_static() && qk_relative_pos_shape.rank() == 4,
                          "qk_relative_pos must be 4-dimensional");
    std::vector<ov::Dimension> output_dims = {
        query_features_shape[1],  // dimensions_0[1] (nhead * bs)
        query_features_shape[0],  // dimensions_0[0] (total_query_num)
        key_attn_index_shape[2]   // dimensions_2[2] (key_index_size)
    };

    set_output_type(0, get_input_element_type(0), ov::PartialShape(output_dims));
}

std::shared_ptr<Node> RotRPEAttentionWeightWithIndexComputation::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_RotRPEAttentionWeightWithIndexComputation_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<RotRPEAttentionWeightWithIndexComputation>(new_args.at(0),
                                                                       new_args.at(1),
                                                                       new_args.at(2),
                                                                       new_args.at(3));
}

bool RotRPEAttentionWeightWithIndexComputation::evaluate(ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs) const override {
    OV_OP_SCOPE(v5_RotRPEAttentionWeightWithIndexComputation_evaluate);
    // Extract input data pointers
    const float* query_features = inputs[0].data<const float>();
    const float* key_features = inputs[1].data<const float>();
    const int32_t* key_attn_index = inputs[2].data<const int32_t>();
    const float* qk_relative_pos = inputs[3].data<const float>();
    float* output = outputs[0].data<float>();

    // Get shapes
    const auto& query_features_shape = inputs[0].get_shape();
    const auto& key_features_shape = inputs[1].get_shape();
    const auto& key_attn_index_shape = inputs[2].get_shape();
    const auto& qk_relative_pos_shape = inputs[3].get_shape();

    // Extract dimensions
    const int32_t bs = qk_relative_pos_shape[0];
    const int32_t total_query_num = query_features_shape[1];
    const int32_t total_key_num = key_features_shape[0];
    const int32_t key_index_size = key_attn_index_shape[2];
    const int32_t nhead = query_features_shape[0] / bs;
    const int32_t hdim = key_features_shape[2];

    // Allocate frequency buffer
    std::vector<float> mFreq(nhead * hdim, 0);
    get_rel_pos_freq(mFreq.data(), nhead * hdim, qk_relative_pos_shape[3]);

    // Call computation kernel
    attention_weight_computation(bs,
                                 total_query_num,
                                 total_key_num,
                                 key_index_size,
                                 nhead,
                                 hdim,
                                 query_features,
                                 key_features,
                                 key_attn_index,
                                 qk_relative_pos,
                                 mFreq.data(),
                                 output);

    return true;
}

bool RotRPEAttentionWeightWithIndexComputation::has_evaluate() const override {
    return true;
}

// --- RotRPEProjectValueWithIndexComputation ---

RotRPEProjectValueWithIndexComputation::RotRPEProjectValueWithIndexComputation(const Output<Node>& input0,
                                                                               const Output<Node>& input1,
                                                                               const Output<Node>& input2,
                                                                               const Output<Node>& input3)
    : Op({input0, input1, input2, input3}) {
    constructor_validate_and_infer_types();
}

void RotRPEProjectValueWithIndexComputation::validate_and_infer_types() {
    OV_OP_SCOPE(v5_RotRPEProjectValueWithIndexComputation_validate_and_infer_types);

    // Validate input count
    NODE_VALIDATION_CHECK(this, get_input_size() == 4, "Expected 4 inputs. Got: ", get_input_size());

    // Validate input shapes
    const auto& attn_weight_shape = get_input_partial_shape(0);
    const auto& value_features_shape = get_input_partial_shape(1);
    const auto& key_attn_index_shape = get_input_partial_shape(2);
    const auto& qk_relative_pos_shape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          attn_weight_shape.rank().is_static() && attn_weight_shape.rank() == 3,
                          "attn_weight features must be 3-dimensional");
    NODE_VALIDATION_CHECK(this,
                          value_features_shape.rank().is_static() && value_features_shape.rank() == 3,
                          "value_features must be 3-dimensional");
    NODE_VALIDATION_CHECK(this,
                          key_attn_index_shape.rank().is_static() && key_attn_index_shape.rank() == 3,
                          "key_attn_index must be 3-dimensional");
    NODE_VALIDATION_CHECK(this,
                          qk_relative_pos_shape.rank().is_static() && pos_qk_relative_pos_shapeshape.rank() == 4,
                          "qk_relative_pos must be 4-dimensional");
    std::vector<ov::Dimension> output_dims = {attn_weight_shape[1], attn_weight_shape[0], key_attn_index_shape[2]};

    set_output_type(0, get_input_element_type(0), ov::PartialShape(output_dims));
}

std::shared_ptr<Node> RotRPEProjectValueWithIndexComputation::clone_with_new_inputs(
    const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_RotRPEProjectValueWithIndexComputation_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<RotRPEProjectValueWithIndexComputation>(new_args.at(0),
                                                                    new_args.at(1),
                                                                    new_args.at(2),
                                                                    new_args.at(3));
}

bool RotRPEProjectValueWithIndexComputation::evaluate(ov::TensorVector& outputs,
                                                      const ov::TensorVector& inputs) const override {
    OV_OP_SCOPE(v5_RotRPEProjectValueWithIndexComputation_evaluate);
    // Extract input data pointers
    const float* attn_weight = inputs[0].data<const float>();
    const float* key_features = inputs[1].data<const float>();
    const int32_t* key_attn_index = inputs[2].data<const int32_t>();
    const float* qk_relative_pos = inputs[3].data<const float>();
    float* output = outputs[0].data<float>();

    // Get shapes
    const auto& attn_weight_shape = inputs[0].get_shape();
    const auto& key_features_shape = inputs[1].get_shape();
    const auto& key_attn_index_shape = inputs[2].get_shape();
    const auto& qk_relative_pos_shape = inputs[3].get_shape();

    // Extract dimensions
    const int32_t bs = qk_relative_pos_shape[0];
    const int32_t total_query_num = attn_weight_shape[1];
    const int32_t total_key_num = key_features_shape[0];
    const int32_t key_index_size = key_attn_index_shape[2];
    const int32_t nhead = attn_weight_shape[0] / bs;
    const int32_t hdim = key_features_shape[2];

    // Allocate frequency buffer
    std::vector<float> mFreq(nhead * hdim, 0);
    get_rel_pos_freq(mFreq.data(), nhead * hdim, qk_relative_pos_shape[3]);

    // Call computation kernel
    attention_value_computation(bs,
                                total_query_num,
                                total_key_num,
                                key_index_size,
                                nhead,
                                hdim,
                                query_features,
                                key_features,
                                key_attn_index,
                                qk_relative_pos,
                                mFreq.data(),
                                output);

    return true;
}

bool RotRPEProjectValueWithIndexComputation::has_evaluate() const override {
    return true;
}

// // 注册扩展
// OPENVINO_CREATE_EXTENSIONS(std::vector<std::shared_ptr<ov::Extension>>(
//                                {std::make_shared<ov::OpExtension<RotRPEAttentionWeightWithIndexComputation>>()}))；

}  // namespace v5
}  // namespace op
}  // namespace ov
