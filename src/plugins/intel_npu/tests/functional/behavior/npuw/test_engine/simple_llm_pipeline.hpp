#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/shared_object.hpp"

class SimpleLLMPipeline {
public:
    void initialize(const std::string& model_path, ov::Core& core, const ov::AnyMap& config);
    std::vector<int64_t> generate(const std::vector<int64_t>& input_ids_vec);
private:
    uint32_t m_batch_dim{std::numeric_limits<uint32_t>::max()};
    uint32_t m_seq_len_dim{std::numeric_limits<uint32_t>::max()};
    uint32_t m_max_prompt_len{std::numeric_limits<uint32_t>::max()};
    uint32_t m_min_response_len{std::numeric_limits<uint32_t>::max()};
    uint32_t m_kvcache_total{std::numeric_limits<uint32_t>::max()};
    std::shared_ptr<ov::CompiledModel> m_compiled_model;
    std::shared_ptr<ov::InferRequest> m_request;
};
