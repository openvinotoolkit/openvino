// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "ov_ops/moe_expert.hpp"
#include <vector>

namespace cldnn {
using MOEExpert = ov::op::internal::MOEExpert;

/// @brief moe_expert primitive
/// @details Performs moe expert
struct moe_expert : public primitive_base<moe_expert> {
    CLDNN_DECLARE_PRIMITIVE(moe_expert)

    moe_expert() : primitive_base("", {}) {}

    /// @brief branch has compiled program, input_map and output_map
    ///
    struct branch {
        std::map<primitive_id, primitive_id> input_map;
        std::map<size_t, primitive_id> output_map;
        program::ptr inner_program;

        std::string str() {
            std::stringstream ss;
            ss << "branch: {input_map : [(outer_id,inner_id),";
            for (auto& in_iter : input_map) {
                ss << "(" << in_iter.first << "," << in_iter.second << "),";
            }
            ss << "],";

            ss << " output_map : [(outer_idx,inner_id),";
            for (auto& out_iter : output_map) {
                ss << "(" << out_iter.first << ","<< out_iter.second << "),";
            }
            ss << "]}";
            return ss.str();
        }

        void save(BinaryOutputBuffer& ob) const {
            ob << input_map.size();
            for (auto& input_pair : input_map) {
                ob << input_pair.first;
                ob << input_pair.second;
            }
            ob << output_map.size();
            for (auto& output_pair : output_map) {
                ob << output_pair.first;
                ob << output_pair.second;
            }
            inner_program->save(ob);
        }

        void load(BinaryInputBuffer& ib) {
            size_t map_size;
            ib >> map_size;
            input_map.clear();
            for (size_t i = 0; i < map_size; ++i) {
                primitive_id input_first, input_second;
                ib >> input_first;
                ib >> input_second;
                input_map.insert({input_first, input_second});
            }
            ib >> map_size;
            output_map.clear();
            for (size_t i = 0; i < map_size; ++i) {
                size_t output_index;
                primitive_id output_second;
                ib >> output_index;
                ib >> output_second;
                output_map.insert({output_index, output_second});
            }
            inner_program = std::make_shared<cldnn::program>(ib.get_engine());
            inner_program->load(ib);
        }
    };

    /// @brief Constructs moe_expert primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    /// @param branch             Branch containg primitives. body in ngraph
    moe_expert(const primitive_id& id,
            const std::vector<input_info>& inputs,
            const MOEExpert::Config& config,
            const branch& branch)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
        _config(config),
        _branch(branch) {}

    MOEExpert::Config _config;
    branch _branch;

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_expert>::save(ob);
        ob << _branch;
        ob << _config.expert_no;
        ob << _config.expert_num;
        ob << _config.hidden_size;
        ob << _config.topk;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_expert>::load(ib);
        ib >> _branch;
        ib >> _config.expert_no;
        ib >> _config.expert_num;
        ib >> _config.hidden_size;
        ib >> _config.topk;
    }

protected:
    std::vector<input_info> get_dependencies() const override { return {}; }
};

static inline std::ostream& operator<< (std::ostream& os, moe_expert::branch& info) {
    os << info.str();
    return os;
}
}  // namespace cldnn
