#ifndef BTL_FUNCTION_LIBRARY_HPP
#define BTL_FUNCTION_LIBRARY_HPP

#include <array>
#include <vector>
#include <cstdint>
#include <string>
#include <map>

namespace btl {

// Ternary value type
using TernaryValue = int8_t;  // {-1, 0, +1}
using TruthTable = std::array<TernaryValue, 9>;  // 9 outputs for 3x3 inputs

// Algebraic properties (bit flags)
struct AlgebraicProperties {
    bool symmetric : 1;
    bool antisymmetric : 1;
    bool monotonic : 1;
    bool self_dual : 1;
    bool associative : 1;
    bool idempotent : 1;
    bool linear_gf3 : 1;
    bool is_projection : 1;
    bool is_constant : 1;
    bool preserves_zero : 1;
};

// Semantic profile
struct SemanticProfile {
    uint8_t excitatory_bias;  // 0-255 (fraction * 255)
    uint8_t inhibitory_bias;
    uint8_t neutral_bias;
};

// BTL Function structure
struct BTLFunction {
    uint16_t id;                    // Unique Function ID (0-19682)
    uint16_t npn_class_id;          // NPN equivalence class (0-1443)
    uint16_t canonical_function_id; // Canonical representative function ID
    TruthTable truth_table;
    uint16_t orbit_size;            // Number of equivalent functions
    AlgebraicProperties properties;
    SemanticProfile semantic_profile;
    
    // Fast application via lookup table
    inline TernaryValue apply(TernaryValue a, TernaryValue b) const {
        // Map {-1,0,+1} × {-1,0,+1} -> index 0-8
        int idx = (a + 1) * 3 + (b + 1);
        return truth_table[idx];
    }
};

    // BTL Function Library
class BTLLibrary {
public:
    BTLLibrary();  // Constructor loads from file or embedded fallback
    
    // Core queries
    const BTLFunction& get_by_npn_class(uint16_t npn_class_id) const;
    const BTLFunction& get_by_function_id(uint32_t function_id) const;
    
    // Property-based search
    std::vector<const BTLFunction*> search_by_properties(
        bool require_monotonic = false,
        bool require_preserves_zero = false,
        float min_neutral_bias = 0.0f,
        float max_neutral_bias = 1.0f
    ) const;
    
    // Exotic function accessors
    const BTLFunction& get_exotic(const std::string& name) const;
    
    // Statistics
    size_t num_npn_classes() const { return npn_classes_.size(); }
    size_t total_functions() const { return 19683; }
    
private:
    std::vector<BTLFunction> npn_classes_;
    std::map<uint32_t, uint16_t> function_to_class_;  // Maps func_id -> npn_class_id
    std::map<std::string, BTLFunction> exotic_functions_;
    
    void load_embedded_data();
    void load_from_file(const std::string& path);
};// Exotic function name constants
namespace exotic {
    constexpr const char* T_WAVE = "T_WAVE";
    constexpr const char* T_DEFAULT = "T_DEFAULT";
    constexpr const char* T_POLARITY = "T_POLARITY";
    constexpr const char* T_SCRY = "T_SCRY";
    constexpr const char* TAND = "TAND";
    constexpr const char* TOR = "TOR";
}

}  // namespace btl

#endif  // BTL_FUNCTION_LIBRARY_HPP
