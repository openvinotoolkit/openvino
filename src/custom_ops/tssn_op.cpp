#include "tssn_op.hpp"
#include "openvino/core/type/element_type.hpp"
#include "btl/btl_function_library.hpp"
#include <immintrin.h>
#include <cstring>
#include <vector>
#include <algorithm>

namespace ov {
namespace op {
namespace v0 {

TSSN::TSSN(const Output<Node>& x, 
           const Output<Node>& h_prev,
           const Output<Node>& A,
           const Output<Node>& B,
           const Output<Node>& C,
           uint32_t func_id_a,
           uint32_t func_id_b,
           uint32_t func_id_c,
           uint32_t func_id_agg,
           bool packed_weights) 
           : Op({x, h_prev, A, B, C}),
             m_func_id_a(func_id_a),
             m_func_id_b(func_id_b),
             m_func_id_c(func_id_c),
             m_func_id_agg(func_id_agg),
             m_packed_weights(packed_weights) {
    validate_and_infer_types();
}

void TSSN::validate_and_infer_types() {
    auto x_shape = get_input_partial_shape(0);
    auto h_shape = get_input_partial_shape(1);
    set_output_type(0, element::i8, x_shape);
    set_output_type(1, element::i8, h_shape);
}

std::shared_ptr<Node> TSSN::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<TSSN>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4),
                                  m_func_id_a, m_func_id_b, m_func_id_c, m_func_id_agg, m_packed_weights);
}

bool TSSN::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("func_id_a", m_func_id_a);
    visitor.on_attribute("func_id_b", m_func_id_b);
    visitor.on_attribute("func_id_c", m_func_id_c);
    visitor.on_attribute("func_id_agg", m_func_id_agg);
    visitor.on_attribute("packed_weights", m_packed_weights);
    return true;
}

bool TSSN::has_evaluate() const {
    return true;
}

void TSSN::unpack_weights(const uint8_t* packed, int8_t* unpacked, size_t count) const {
    for (size_t i = 0; i < count; ++i) {
        size_t byte_idx = i / 4;
        size_t bit_offset = (i % 4) * 2;
        uint8_t val = (packed[byte_idx] >> bit_offset) & 0x3;
        if (val == 0) unpacked[i] = 0;
        else if (val == 1) unpacked[i] = 1;
        else if (val == 2) unpacked[i] = -1;
        else unpacked[i] = 0;
    }
}

// Helper for AVX512 emulation if needed
#if defined(__AVX512BW__)
static inline __m512i mm512_sign_epi8_emulated(__m512i a, __m512i b) {
    // _mm512_sign_epi8 is available in AVX512BW
    return _mm512_sign_epi8(a, b);
}
#endif

bool TSSN::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    const auto& x = inputs[0];
    const auto& h_prev = inputs[1];
    const auto& A = inputs[2];
    const auto& B = inputs[3];
    const auto& C = inputs[4];
    
    auto& y = outputs[0];
    auto& h_next = outputs[1];

    // Get data pointers (assuming int8)
    const int8_t* x_data = x.data<int8_t>();
    const int8_t* h_prev_data = h_prev.data<int8_t>();
    
    // Handle Weights (Unpack if needed)
    std::vector<int8_t> A_buf, B_buf, C_buf;
    const int8_t *A_data, *B_data, *C_data;
    
    size_t hidden = x.get_shape()[2]; // Assuming [Batch, Seq, Hidden]

    if (m_packed_weights) {
        A_buf.resize(hidden); unpack_weights(A.data<uint8_t>(), A_buf.data(), hidden); A_data = A_buf.data();
        B_buf.resize(hidden); unpack_weights(B.data<uint8_t>(), B_buf.data(), hidden); B_data = B_buf.data();
        C_buf.resize(hidden); unpack_weights(C.data<uint8_t>(), C_buf.data(), hidden); C_data = C_buf.data();
    } else {
        A_data = A.data<int8_t>();
        B_data = B.data<int8_t>();
        C_data = C.data<int8_t>();
    }
    
    int8_t* y_data = y.data<int8_t>();
    int8_t* h_next_data = h_next.data<int8_t>();

    // Shapes
    auto x_shape = x.get_shape();
    size_t batch = x_shape[0];
    size_t seq_len = x_shape[1];
    // hidden is already set

    // Temporary state buffer for current timestep
    std::vector<int8_t> h_curr(hidden);

    for (size_t b = 0; b < batch; ++b) {
        // Initialize state for this batch with h_prev
        const int8_t* batch_h_prev = h_prev_data + b * hidden;
        std::memcpy(h_curr.data(), batch_h_prev, hidden * sizeof(int8_t));

        for (size_t t = 0; t < seq_len; ++t) {
            const int8_t* batch_x_t = x_data + b * seq_len * hidden + t * hidden;
            int8_t* batch_y_t = y_data + b * seq_len * hidden + t * hidden;

            size_t i = 0;
            
            // AVX-512 Loop
            #if defined(__AVX512BW__)
            for (; i + 64 <= hidden; i += 64) {
                __m512i x_vec = _mm512_loadu_si512(batch_x_t + i);
                __m512i h_vec = _mm512_loadu_si512(h_curr.data() + i);
                
                __mmask64 mask_x = _mm512_test_epi8_mask(x_vec, x_vec);
                __mmask64 mask_h = _mm512_test_epi8_mask(h_vec, h_vec);
                
                if (mask_x == 0 && mask_h == 0) {
                    _mm512_storeu_si512(h_curr.data() + i, _mm512_setzero_si512());
                    _mm512_storeu_si512(batch_y_t + i, _mm512_setzero_si512());
                    continue;
                }
                
                __m512i a_vec = _mm512_loadu_si512(A_data + i);
                __m512i b_vec = _mm512_loadu_si512(B_data + i);
                __m512i c_vec = _mm512_loadu_si512(C_data + i);
                
                __m512i term1 = _mm512_setzero_si512();
                if (mask_h != 0) term1 = mm512_sign_epi8_emulated(h_vec, a_vec);
                
                __m512i term2 = _mm512_setzero_si512();
                if (mask_x != 0) term2 = mm512_sign_epi8_emulated(x_vec, b_vec);
                
                __m512i h_new = _mm512_adds_epi8(term1, term2);
                _mm512_storeu_si512(h_curr.data() + i, h_new);
                
                __m512i y_new = mm512_sign_epi8_emulated(h_new, c_vec);
                _mm512_storeu_si512(batch_y_t + i, y_new);
            }
            #elif defined(__AVX2__)
            for (; i + 32 <= hidden; i += 32) {
                __m256i x_vec = _mm256_loadu_si256((const __m256i*)(batch_x_t + i));
                __m256i h_vec = _mm256_loadu_si256((const __m256i*)(h_curr.data() + i));
                
                int zero_x = _mm256_testz_si256(x_vec, x_vec);
                int zero_h = _mm256_testz_si256(h_vec, h_vec);
                
                if (zero_x && zero_h) {
                    _mm256_storeu_si256((__m256i*)(h_curr.data() + i), _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i*)(batch_y_t + i), _mm256_setzero_si256());
                    continue;
                }
                
                __m256i a_vec = _mm256_loadu_si256((const __m256i*)(A_data + i));
                __m256i b_vec = _mm256_loadu_si256((const __m256i*)(B_data + i));
                __m256i c_vec = _mm256_loadu_si256((const __m256i*)(C_data + i));
                
                __m256i term1 = _mm256_setzero_si256();
                if (!zero_h) term1 = _mm256_sign_epi8(h_vec, a_vec);
                
                __m256i term2 = _mm256_setzero_si256();
                if (!zero_x) term2 = _mm256_sign_epi8(x_vec, b_vec);
                
                __m256i h_new = _mm256_adds_epi8(term1, term2);
                _mm256_storeu_si256((__m256i*)(h_curr.data() + i), h_new);
                
                __m256i y_new = _mm256_sign_epi8(h_new, c_vec);
                _mm256_storeu_si256((__m256i*)(batch_y_t + i), y_new);
            }
            #endif

            // Scalar Tail
            for (; i < hidden; ++i) {
                int8_t a_val = A_data[i];
                int8_t b_val = B_data[i];
                int8_t c_val = C_data[i];
                
                int8_t x_val = batch_x_t[i];
                int8_t h_val = h_curr[i];

                int32_t h_new_val = 0;
                if (a_val != 0 && h_val != 0) h_new_val += (int32_t)a_val * (int32_t)h_val;
                if (b_val != 0 && x_val != 0) h_new_val += (int32_t)b_val * (int32_t)x_val;

                if (h_new_val > 127) h_new_val = 127;
                if (h_new_val < -128) h_new_val = -128;
                
                h_curr[i] = (int8_t)h_new_val;

                int32_t y_val = 0;
                if (c_val != 0 && h_curr[i] != 0) y_val = (int32_t)c_val * (int32_t)h_curr[i];
                
                if (y_val > 127) y_val = 127;
                if (y_val < -128) y_val = -128;

                batch_y_t[i] = (int8_t)y_val;
            }
        }
        
        // Save final state to h_next
        int8_t* batch_h_next = h_next_data + b * hidden;
        std::memcpy(batch_h_next, h_curr.data(), hidden * sizeof(int8_t));
    }

    return true;
}

} // namespace v0
} // namespace op
} // namespace ov
