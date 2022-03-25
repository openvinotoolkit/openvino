// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/ops.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "transformations/hash.hpp"

using namespace ngraph;

namespace {  // helpers

template <typename T>
uint64_t hash_combine(uint64_t seed, const T& a) {
    // Hash combine formula from boost
    return seed ^ (std::hash<T>()(a) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename Container>
void join_hash(uint64_t& hash, const Container& c) {
    for (const auto& v : c) {
        hash = hash_combine(hash, v);
    }
}

uint64_t hash_combine(const void* v, size_t size) {
    constexpr auto cel_size = sizeof(uint64_t);
    auto seed = static_cast<uint64_t>(size);
    const auto data = static_cast<const uint64_t*>(v);
    const auto steps = size / cel_size;
    const auto d_end = std::next(data, static_cast<int64_t>(steps));
    // Slower version: this is > ~2 times slower than calculation of separate 4 sums
    //    for (auto d = data; d < d_end; ++d) {
    //        sum ^= *d + 0x9e3779b9 + (sum << 6) + (sum >> 2);
    //    }

    // ---- Faster version ---
    // Calculate 4 separate independent hash sums within one cycle
    constexpr auto fast_step = 4;
    uint64_t sum[fast_step] = {0};
    const auto fast_step_cnt = steps / fast_step;
    const auto d_end_fast = std::next(data, static_cast<int64_t>(fast_step_cnt * fast_step));
    for (auto d = data; d < d_end_fast; d += fast_step) {
        // This appears to be almost 2 times faster to calculate separate sums than do it with step=1
        for (auto i = 0; i < fast_step; i++) {
            sum[i] ^= d[i] + 0x9e3779b9 + (sum[i] << 6) + (sum[i] >> 2);
        }
    }
    for (auto i = 0; i < steps % fast_step; i++) {
        seed = hash_combine(seed, d_end_fast[i]);
    }
    for (auto i = 0; i < fast_step; i++) {
        seed = hash_combine(seed, sum[i]);
    }
    // ---- End Faster version ----
    uint64_t last_bytes{0};
    std::memcpy(&last_bytes, d_end, size % cel_size);
    seed ^= last_bytes + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

void model_2_hash(uint64_t& hash, const ov::Model& f);

namespace rt_info {

class RTInfoHasher : public ov::AttributeVisitor {
    uint64_t& m_hash;

public:
    RTInfoHasher(uint64_t& hash) : m_hash(hash) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        check_attribute_name(name);
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            join_hash(m_hash, a->get());
        } else {
            // Other RT attributes are not taken into account for hashing
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        check_attribute_name(name);
        m_hash = hash_combine(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        check_attribute_name(name);
        m_hash = hash_combine(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        check_attribute_name(name);
        m_hash = hash_combine(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        check_attribute_name(name);
        m_hash = hash_combine(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        check_attribute_name(name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        check_attribute_name(name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        check_attribute_name(name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        check_attribute_name(name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        check_attribute_name(name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        throw ngraph_error("Model type is unsupported for rt info hashing");
    }

    void check_attribute_name(const std::string& name) {
        OPENVINO_ASSERT(name != "name" && name != "version",
                        "Attribute key with name: " + name + " is not allowed. Please use another name");
        m_hash = hash_combine(m_hash, name);
    }
};
}  // namespace rt_info

std::unordered_map<ov::Node*, int> create_layer_ids(const std::vector<std::shared_ptr<ov::Node>>& ops) {
    std::unordered_map<ov::Node*, int> layer_ids;
    int id = 0;
    for (const auto& node : ops) {
        layer_ids[node.get()] = id++;
    }
    return layer_ids;
}

class HashSerializer : public ov::AttributeVisitor {
    uint64_t& m_hash;

    void input_descriptions_on_adapter(
        const std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>>& input_descriptions) {
        for (const auto& input_description : input_descriptions) {
            m_hash = hash_combine(m_hash, input_description->m_input_index);
            m_hash = hash_combine(m_hash, input_description->m_body_parameter_index);
            if (auto slice_input =
                    ov::as_type_ptr<ov::op::util::SubGraphOp::SliceInputDescription>(input_description)) {
                m_hash = hash_combine(m_hash, slice_input->m_axis);
                m_hash = hash_combine(m_hash, slice_input->m_start);
                m_hash = hash_combine(m_hash, slice_input->m_end);
                m_hash = hash_combine(m_hash, slice_input->m_stride);
                m_hash = hash_combine(m_hash, slice_input->m_part_size);
            } else if (auto merged_input =
                           ov::as_type_ptr<ov::op::util::SubGraphOp::MergedInputDescription>(input_description)) {
                m_hash = hash_combine(m_hash, merged_input->m_body_value_index);
            }
        }
    }

    void output_descriptions_on_adapter(
        const std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>>& output_descriptions) {
        for (const auto& output_description : output_descriptions) {
            m_hash = hash_combine(m_hash, output_description->m_output_index);
            m_hash = hash_combine(m_hash, output_description->m_body_value_index);

            if (auto concat_output =
                    ov::as_type_ptr<ov::op::util::SubGraphOp::ConcatOutputDescription>(output_description)) {
                m_hash = hash_combine(m_hash, concat_output->m_axis);
                m_hash = hash_combine(m_hash, concat_output->m_start);
                m_hash = hash_combine(m_hash, concat_output->m_end);
                m_hash = hash_combine(m_hash, concat_output->m_stride);
                m_hash = hash_combine(m_hash, concat_output->m_part_size);
            }
        }
    }

public:
    explicit HashSerializer(uint64_t& hash) : m_hash(hash) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        m_hash = hash_combine(m_hash, name);

        if (const auto& a_id = ov::as_type<
                ov::AttributeAdapter<std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>>>>(
                &adapter)) {
            input_descriptions_on_adapter(a_id->get());
        } else if (const auto& a_od = ov::as_type<ov::AttributeAdapter<
                       std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>>>>(&adapter)) {
            output_descriptions_on_adapter(a_od->get());
        } else if (const auto& a_ports =
                       ov::as_type<ov::AttributeAdapter<ov::op::v5::Loop::SpecialBodyPorts>>(&adapter)) {
            m_hash = hash_combine(m_hash, a_ports->get().current_iteration_input_idx);
            m_hash = hash_combine(m_hash, a_ports->get().body_condition_output_idx);
        } else if (const auto& a_v =
                       ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>>(&adapter)) {
            m_hash = hash_combine(m_hash, a_v->get()->get_info().variable_id);
            std::stringstream stream;
            stream << a_v->get()->get_info().data_shape << a_v->get()->get_info().data_type;
            m_hash = hash_combine(m_hash, stream.str());
        } else if (const auto& a_ab =
                       ov::as_type<ov::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(&adapter)) {
            m_hash = hash_combine(m_hash, hash_combine(a_ab->get()->get_ptr(), a_ab->get()->size()));
        } else if (const auto& a_fn = ov::as_type<ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>>(&adapter)) {
            const auto& attrs = a_fn->get();

            m_hash = hash_combine(m_hash, attrs.get_type_name());

            if (!attrs.get_opset_name().empty()) {
                m_hash = hash_combine(m_hash, attrs.get_opset_name());
            }

            // Update node attributes in data field
            for (const auto& attr : attrs) {
                m_hash = hash_combine(m_hash, attr.first);
                m_hash = hash_combine(m_hash, attr.second);
            }
        } else if (const auto& a_tv = ov::as_type<ov::AttributeAdapter<ov::element::TypeVector>>(&adapter)) {
            const auto& attrs = a_tv->get();
            m_hash = hash_combine(m_hash, join(attrs));
        } else if (const auto& a_ps = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& attrs = a_ps->get();
            std::stringstream shape_str_stream;
            shape_str_stream << attrs;
            m_hash = hash_combine(m_hash, shape_str_stream.str());
        } else if (const auto& a_dim = ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
            const auto& attrs = a_dim->get();
            std::stringstream dim_str_stream;
            dim_str_stream << attrs;
            m_hash = hash_combine(m_hash, dim_str_stream.str());
            auto dim_str = dim_str_stream.str();
        } else {
            // Unsupported attribute type will not be taken into account for hash calculation
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<Function>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        model_2_hash(m_hash, *adapter.get());
    }
};

template <typename T>
bool is_name_auto_generated(const T& n) {
    return n.get_friendly_name() == n.get_name();
}

void hash_runtime_info(uint64_t& hash, const RTMap& attributes) {
    for (const auto& item : attributes) {
        hash = hash_combine(hash, item.first);
        if (item.second.is<ov::RuntimeAttribute>()) {
            auto& rt_attribute = item.second.as<ov::RuntimeAttribute>();
            const auto& type_info = rt_attribute.get_type_info();
            hash = hash_combine(hash, type_info.name);
            hash = hash_combine(hash, type_info.get_version());
            rt_info::RTInfoHasher rt_hasher(hash);
            rt_attribute.visit_attributes(rt_hasher);
        } else if (item.second.is<std::string>()) {
            // It is much faster than default serialization via 'stringstream'
            hash = hash_combine(hash, item.second.as<std::string>());
        } else if (item.second.is<int64_t>()) {
            hash = hash_combine(hash, item.second.as<int64_t>());
        } else if (item.second.is<bool>()) {
            hash = hash_combine(hash, item.second.as<bool>());
        } else if (item.second.is<uint64_t>()) {
            hash = hash_combine(hash, item.second.as<uint64_t>());
        } else if (item.second.is<float>()) {
            hash = hash_combine(hash, item.second.as<float>());
        } else if (item.second.is<double>()) {
            hash = hash_combine(hash, item.second.as<double>());
        } else if (item.second.is<std::vector<std::string>>()) {
            join_hash(hash, item.second.as<std::vector<std::string>>());
        } else if (item.second.is<std::vector<int64_t>>()) {
            join_hash(hash, item.second.as<std::vector<int64_t>>());
        } else if (item.second.is<std::vector<int>>()) {
            join_hash(hash, item.second.as<std::vector<int>>());
        } else if (item.second.is<std::vector<float>>()) {
            join_hash(hash, item.second.as<std::vector<float>>());
        } else if (item.second.is<std::vector<uint64_t>>()) {
            join_hash(hash, item.second.as<std::vector<uint64_t>>());
        } else {
            std::stringstream stream;
            item.second.print(stream);
            hash = hash_combine(hash, stream.str());
        }
    }
}

void model_2_hash(uint64_t& hash, const ov::Model& f) {
    auto& rt_info = f.get_rt_info();
    if (rt_info.count("version")) {
        hash = hash_combine(hash, rt_info.at("version").as<int64_t>());
    }
    // Don't include auto-generated names into hash
    if (!is_name_auto_generated(f)) {
        hash = hash_combine(hash, f.get_friendly_name());
    }
    std::unordered_set<std::string> unique_names;
    auto sorted_ops = f.get_ordered_ops();
    auto layer_ids = create_layer_ids(sorted_ops);
    std::vector<std::shared_ptr<ov::Node>> result;
    for (const auto& n : sorted_ops) {
        ov::Node* node = n.get();
        std::string node_type_name{node->get_type_name()};
        hash = hash_combine(hash, node_type_name);
        // Don't include auto-generated names into hash
        if (!is_name_auto_generated(*node)) {
            hash = hash_combine(hash, node->get_friendly_name());
        }

        for (auto& i : node->inputs()) {
            // This code is taken from Serialize pass
            // WA for LSTMCellv0, peephole input shall not be serialized
            if (i.get_index() == 6 && dynamic_cast<opset1::LSTMCell*>(node)) {
                continue;
            }

            hash = hash_combine(hash, i.get_element_type().hash());
            for (const auto& d : i.get_partial_shape()) {
                hash = hash_combine(hash, d.get_min_length());
                hash = hash_combine(hash, d.get_max_length());
            }
            hash_runtime_info(hash, i.get_rt_info());

            // hash edges to this node (using layer_ids)
            auto source_output = i.get_source_output();
            auto source_node = source_output.get_node();
            auto current_node = i.get_node();
            auto found_source_node = layer_ids.find(source_node);
            auto found_current_node = layer_ids.find(current_node);

            NGRAPH_CHECK(found_source_node != layer_ids.end(), "Internal error, model is invalid");
            NGRAPH_CHECK(found_current_node != layer_ids.end(), "Internal error, model is invalid");

            hash = hash_combine(hash, found_source_node->second);
            hash = hash_combine(hash, source_output.get_index());
            hash = hash_combine(hash, found_current_node->second);
            hash = hash_combine(hash, i.get_index());
        }
        if ((node->get_output_size() > 0) && !ov::op::util::is_output(node)) {
            for (auto& o : node->outputs()) {
                hash = hash_combine(hash, o.get_element_type().hash());

                // Sort tensor names
                const auto& tensor_names = o.get_tensor().get_names();
                std::vector<std::string> vector_names(tensor_names.begin(), tensor_names.end());
                sort(vector_names.begin(), vector_names.end());
                join_hash(hash, vector_names);

                for (const auto& d : o.get_partial_shape()) {
                    hash = hash_combine(hash, d.get_min_length());
                    hash = hash_combine(hash, d.get_max_length());
                }
                hash_runtime_info(hash, o.get_rt_info());
            }
        }

        // hash general attributes
        HashSerializer visitor(hash);
        NGRAPH_CHECK(node->visit_attributes(visitor), "Visitor API is not supported in ", node);
        hash_runtime_info(hash, node->get_rt_info());
    }
}

}  // namespace

namespace ov {
pass::FasterHash::FasterHash(uint64_t& hash) : m_hash(hash) {}

bool pass::FasterHash::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_hash = 0;
    model_2_hash(m_hash, *model);

    // Return false because we didn't change ov::Model
    return false;
}

}  // namespace ov
