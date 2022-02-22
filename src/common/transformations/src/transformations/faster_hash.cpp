// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <ngraph/variant.hpp>
#include <unordered_map>
#include <unordered_set>

#include "ngraph/ops.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/hash.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

using namespace ngraph;

namespace {  // helpers

template <typename T>
static uint64_t hash_combine(uint64_t seed, const T& a) {
    // Hash combine formula from boost
    return seed ^ (std::hash<T>()(a) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename Container>
void join_hash(uint64_t& hash, const Container& c) {
    for (const auto& v : c) {
        hash = hash_combine(hash, v);
    }
}

size_t hash_combine(const void* v, int64_t size) {
    constexpr auto cel_size = sizeof(size_t);
    auto seed = static_cast<size_t>(size);
    const auto data = static_cast<const size_t*>(v);
    const auto steps = size / cel_size;
    const auto d_end = std::next(data, steps);
    // Slower version: this is > ~2 times slower than calculation of separate 4 sums
    //    for (auto d = data; d < d_end; ++d) {
    //        sum ^= *d + 0x9e3779b9 + (sum << 6) + (sum >> 2);
    //    }

    // ---- Faster version ---
    // Calculate 4 separate independent hash sums within one cycle
    constexpr auto fast_step = 4;
    size_t sum[fast_step] = {0};
    const auto fast_step_cnt = steps / fast_step;
    const auto d_end_fast = std::next(data, fast_step_cnt * fast_step);
    for (auto d = data; d < d_end_fast; d += fast_step) {
        // This appears to be almost 2 times faster to calculate separate sums than do it with step=1
        for (auto i = 0; i < fast_step; i++) {
            sum[i] = d[i] + 0x9e3779b9 + (sum[i] << 6) + (sum[i] >> 2);
        }
    }
    for (auto i = 0; i < steps % fast_step; i++) {
        seed = hash_combine(seed, d_end_fast[i]);
    }
    for (auto i = 0; i < fast_step; i++) {
        seed = hash_combine(seed, sum[i]);
    }
    // ---- End Faster version ----
    size_t last_bytes{0};
    std::memcpy(&last_bytes, d_end, size % cel_size);
    seed ^= last_bytes + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

void model_2_hash(uint64_t& hash, const ov::Model& f);

namespace rt_info {

class RTInfoHasher : public ngraph::AttributeVisitor {
    uint64_t& m_hash;

public:
    RTInfoHasher(uint64_t& hash) : m_hash(hash) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            join_hash(m_hash, a->get());
        } else {
            // Other RT attributes are not taking into account for hashing
        }
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::shared_ptr<Function>>& adapter) override {
        throw ngraph_error("Model type is unsupported for rt info hashing");
    }

    static void check_attribute_name(const std::string& name) {
        if (name == "name" || name == "version") {
            throw ngraph_error("Attribute key with name: " + name + " is not allowed. Please use another name");
        }
    }
};
}  // namespace rt_info

std::unordered_map<ngraph::Node*, int> create_layer_ids(const std::vector<std::shared_ptr<ov::Node>>& ops) {
    std::unordered_map<ngraph::Node*, int> layer_ids;
    int id = 0;
    for (const auto& node : ops) {
        layer_ids[node.get()] = id++;
    }
    return layer_ids;
}

class HashSerializer : public ngraph::AttributeVisitor {
    size_t& m_hash;

    void input_descriptions_on_adapter(
            const std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>& input_descriptions) {
        for (const auto& input_description : input_descriptions) {
            m_hash = hash_combine(m_hash, input_description->m_input_index);
            m_hash = hash_combine(m_hash, input_description->m_body_parameter_index);
            if (auto slice_input =
                    ov::as_type_ptr<ngraph::op::util::SubGraphOp::SliceInputDescription>(input_description)) {
                m_hash = hash_combine(m_hash, slice_input->m_axis);
                m_hash = hash_combine(m_hash, slice_input->m_start);
                m_hash = hash_combine(m_hash, slice_input->m_end);
                m_hash = hash_combine(m_hash, slice_input->m_stride);
                m_hash = hash_combine(m_hash, slice_input->m_part_size);
            } else if (auto merged_input =
                    ov::as_type_ptr<ngraph::op::util::SubGraphOp::MergedInputDescription>(input_description)) {
                m_hash = hash_combine(m_hash, merged_input->m_body_value_index);
            }
        }
    }

    void output_descriptions_on_adapter(
            const std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>& output_descriptions) {
        for (const auto& output_description : output_descriptions) {
            m_hash = hash_combine(m_hash, output_description->m_output_index);
            m_hash = hash_combine(m_hash, output_description->m_body_value_index);

            if (auto concat_output =
                    ov::as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(output_description)) {
                m_hash = hash_combine(m_hash, concat_output->m_axis);
                m_hash = hash_combine(m_hash, concat_output->m_start);
                m_hash = hash_combine(m_hash, concat_output->m_end);
                m_hash = hash_combine(m_hash, concat_output->m_stride);
                m_hash = hash_combine(m_hash, concat_output->m_part_size);
            }
        }
    }

    void special_body_ports_on_adapter(const ngraph::op::v5::Loop::SpecialBodyPorts& special_body_ports) {
        if (special_body_ports.current_iteration_input_idx != -1) {
            m_hash = hash_combine(m_hash, special_body_ports.current_iteration_input_idx);
        }

        if (special_body_ports.body_condition_output_idx != -1) {
            m_hash = hash_combine(m_hash, special_body_ports.body_condition_output_idx);
        }
    }

public:
    HashSerializer(size_t& hash)
            : m_hash(hash) {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        using BodyTargetNames = std::tuple<std::string, std::string, std::vector<std::string>>;
        m_hash = hash_combine(m_hash, name);

//        const std::vector<BodyTargetNames> body_names = {
//                BodyTargetNames{"body", "port_map", {"input_descriptions", "output_descriptions", "special_body_ports"}},
//                BodyTargetNames{"then_body", "then_port_map", {"then_inputs", "then_outputs"}},
//                BodyTargetNames{"else_body", "else_port_map", {"else_inputs", "else_outputs"}}};
//        BodyTargetNames bnames;
//        bool is_body_target = false;
//        for (const auto& _body_target : body_names) {
//            if (m_xml_node.parent().child(std::get<0>(_body_target).c_str())) {
//                auto vec_names = std::get<2>(_body_target);
//
//                if (std::find(vec_names.begin(), vec_names.end(), name) != vec_names.end()) {
//                    is_body_target = true;
//                    bnames = _body_target;
//                    break;
//                }
//            }
//        }
//        if (is_body_target) {
//            auto body_name = std::get<0>(bnames);
//            auto portmap_name = std::get<1>(bnames);
            // TI, Loop do not have attributtes as regular ops, it is necessary to append "port_map" and
            // "back_edges" to layer above (m_xml_node.parent()) as in ngfunction_2_ir() layer (here "m_xml_node")
            // with empty attributes is removed.
//            if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<
//                    std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>>>(&adapter)) {
//                input_descriptions_on_adapter(a->get());
//            } else if (const auto& a = ngraph::as_type<ngraph::AttributeAdapter<
//                    std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>>>(
//                    &adapter)) {
//                uint32_t op_input_count = 0;
////                for (auto c = m_xml_node.parent().child("input").first_child(); !c.empty(); c = c.next_sibling()) {
////                    op_input_count++;
////                }
//                output_descriptions_on_adapter(a->get());
//            } else if (const auto& a =
//                    ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::v5::Loop::SpecialBodyPorts>>(
//                            &adapter)) {
//                special_body_ports_on_adapter(a->get());
//            }
        if (const auto& a_id = ngraph::as_type<ngraph::AttributeAdapter<
                std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::InputDescription>>>>(&adapter)) {
            input_descriptions_on_adapter(a_id->get());
        } else if (const auto& a_od = ngraph::as_type<ngraph::AttributeAdapter<
                std::vector<std::shared_ptr<ngraph::op::util::MultiSubGraphOp::OutputDescription>>>>(
                &adapter)) {
            output_descriptions_on_adapter(a_od->get());
        } else if (const auto& a_ports =
                ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::v5::Loop::SpecialBodyPorts>>(
                        &adapter)) {
            special_body_ports_on_adapter(a_ports->get());
        } else if (const auto& a_v =
                ngraph::as_type<ngraph::AttributeAdapter<std::shared_ptr<ngraph::Variable>>>(&adapter)) {
            m_hash = hash_combine(m_hash, a_v->get()->get_info().variable_id);
        } else if (const auto& a_ab =
                ngraph::as_type<ngraph::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
                        &adapter)) {
            m_hash = hash_combine(m_hash, hash_combine(a_ab->get()->get_ptr(), a_ab->get()->size()));
        } else if (const auto& a_fn =
                ngraph::as_type<ngraph::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>>(&adapter)) {
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
        } else if (const auto& a_tv = ngraph::as_type<ngraph::AttributeAdapter<ngraph::element::TypeVector>>(&adapter)) {
            const auto& attrs = a_tv->get();
            m_hash = hash_combine(m_hash, join(attrs));
        } else if (const auto& a_ps = ngraph::as_type<ngraph::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& attrs = a_ps->get();
            std::stringstream shape_str_stream;
            shape_str_stream << attrs;
            m_hash = hash_combine(m_hash, shape_str_stream.str());
        } else if (const auto& a_dim = ngraph::as_type<ngraph::AttributeAdapter<ov::Dimension>>(&adapter)) {
            const auto& attrs = a_dim->get();
            std::stringstream dim_str_stream;
            dim_str_stream << attrs;
            m_hash = hash_combine(m_hash, dim_str_stream.str());
            auto dim_str = dim_str_stream.str();
        } else {
            throw ngraph_error("Unsupported attribute type for serialization: " + name);
        }
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        m_hash = hash_combine(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_hash = hash_combine(m_hash, name);
        join_hash(m_hash, adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::shared_ptr<Function>>& adapter) override {
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
//        hash = hash_combine(hash, item.second.hash());
        if (item.second.is<ov::RuntimeAttribute>()) {
            auto &rt_attribute = item.second.as<ov::RuntimeAttribute>();
            const auto &type_info = rt_attribute.get_type_info();
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

void model_2_hash(uint64_t& hash,
                     const ov::Model& f) {
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
        ngraph::Node* node = n.get();
        const std::string& node_type_name{node->get_type_name()};
        // Don't include auto-generated names into hash
        if (!is_name_auto_generated(*node)) {
            hash = hash_combine(hash, node->get_friendly_name());
        }

        int port_id = 0;
        for (auto& i : node->inputs()) {
            // WA for LSTMCellv0, peephole input shall not be serialized
            if (i.get_index() == 6 && dynamic_cast<opset1::LSTMCell*>(node)) {
                port_id++;
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
        if ((node->get_output_size() > 0) && !ngraph::op::is_output(node)) {
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
pass::FasterHash::FasterHash(uint64_t& hash): m_hash(hash) {}

bool pass::FasterHash::run_on_model(const std::shared_ptr<ov::Model>& model) {
    model_2_hash(m_hash, *model);

    // Return false because we didn't change ov::Model
    return false;
}

}  // namespace ov
