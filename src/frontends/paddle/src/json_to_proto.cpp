// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_to_proto.hpp"

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "framework.pb.h"
#include "nlohmann/json.hpp"
#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
namespace paddle {

using json = nlohmann::json;
using namespace ::paddle::framework::proto;

static std::string var_name(int id) {
    return "v_" + std::to_string(id);
}

static std::string strip_op_name(const std::string& raw) {
    if (raw == "p")
        return "p";
    auto dot = raw.find('.');
    if (dot != std::string::npos) {
        std::string name = raw.substr(dot + 1);
        if (!name.empty() && name.back() == '_')
            name.pop_back();
        return name;
    }
    return raw;
}

static VarType_Type dtype_to_proto(const std::string& dtype) {
    static const std::map<std::string, VarType_Type> m{
        {"float32", VarType_Type_FP32},
        {"float64", VarType_Type_FP64},
        {"float16", VarType_Type_FP16},
        {"int32", VarType_Type_INT32},
        {"int64", VarType_Type_INT64},
        {"int16", VarType_Type_INT16},
        {"int8", VarType_Type_INT8},
        {"uint8", VarType_Type_UINT8},
        {"bool", VarType_Type_BOOL},
        {"bfloat16", VarType_Type_BF16},
    };
    auto it = m.find(dtype);
    FRONT_END_GENERAL_CHECK(it != m.end(), "Unknown PIR dtype: ", dtype);
    return it->second;
}

static VarType_Type type_tag_to_proto(const std::string& tag) {
    static const std::map<std::string, VarType_Type> m{
        {"0.t_f32", VarType_Type_FP32},
        {"0.t_f64", VarType_Type_FP64},
        {"0.t_f16", VarType_Type_FP16},
        {"0.t_i32", VarType_Type_INT32},
        {"0.t_i64", VarType_Type_INT64},
        {"0.t_i16", VarType_Type_INT16},
        {"0.t_i8", VarType_Type_INT8},
        {"0.t_ui8", VarType_Type_UINT8},
        {"0.t_bool", VarType_Type_BOOL},
        {"0.t_bf16", VarType_Type_BF16},
    };
    auto it = m.find(tag);
    if (it != m.end())
        return it->second;
    // Unknown type tag — fall back to FP32; this is safe for most cases
    return VarType_Type_FP32;
}

static const std::map<std::string, std::vector<std::string>>& op_input_names() {
    static const std::map<std::string, std::vector<std::string>> m{
        {"conv2d", {"Input", "Filter"}},
        {"depthwise_conv2d", {"Input", "Filter"}},
        {"conv2d_transpose", {"Input", "Filter"}},
        {"batch_norm", {"X", "Mean", "Variance", "Scale", "Bias"}},
        {"pool2d", {"X"}},
        {"relu", {"X"}},
        {"hardswish", {"X"}},
        {"hardsigmoid", {"X"}},
        {"sigmoid", {"X"}},
        {"add", {"X", "Y"}},
        {"subtract", {"X", "Y"}},
        {"multiply", {"X", "Y"}},
        {"divide", {"X", "Y"}},
        {"elementwise_add", {"X", "Y"}},
        {"elementwise_mul", {"X", "Y"}},
        {"reshape", {"X", "ShapeTensor"}},
        {"reshape2", {"X", "ShapeTensor"}},
        {"concat", {"X"}},
        {"flatten_contiguous_range", {"X"}},
        {"matmul_v2", {"X", "Y"}},
        {"softmax", {"X"}},
        {"scale", {"X"}},
        {"flatten2", {"X"}},
        {"transpose2", {"X"}},
        {"unsqueeze2", {"X"}},
        {"squeeze2", {"X"}},
        {"layer_norm", {"X", "Scale", "Bias"}},
        {"leaky_relu", {"X"}},
        {"gelu", {"X"}},
        {"slice", {"Input"}},
    };
    return m;
}

static const std::map<std::string, std::vector<std::string>>& op_output_names() {
    static const std::map<std::string, std::vector<std::string>> m{
        {"batch_norm", {"Y", "MeanOut", "VarianceOut", "SavedMean", "SavedVariance", "ReserveSpace"}},
        {"pool2d", {"Out"}},
        {"conv2d", {"Output"}},
        {"depthwise_conv2d", {"Output"}},
        {"conv2d_transpose", {"Output"}},
        {"add", {"Out"}},
        {"subtract", {"Out"}},
        {"multiply", {"Out"}},
        {"divide", {"Out"}},
        {"relu", {"Out"}},
        {"hardswish", {"Out"}},
        {"hardsigmoid", {"Out"}},
        {"sigmoid", {"Out"}},
        {"reshape", {"Out"}},
        {"reshape2", {"Out", "XShape"}},
        {"concat", {"Out"}},
        {"softmax", {"Out"}},
        {"scale", {"Out"}},
        {"elementwise_add", {"Out"}},
        {"elementwise_mul", {"Out"}},
        {"flatten_contiguous_range", {"Out"}},
        {"flatten2", {"Out", "XShape"}},
        {"matmul_v2", {"Out"}},
        {"transpose2", {"Out", "XShape"}},
        {"unsqueeze2", {"Out", "XShape"}},
        {"squeeze2", {"Out", "XShape"}},
        {"layer_norm", {"Y", "Mean", "Variance"}},
        {"leaky_relu", {"Out"}},
        {"gelu", {"Out"}},
        {"slice", {"Out"}},
    };
    return m;
}

static std::vector<std::string> get_input_port_names(const std::string& op_name, size_t count) {
    auto it = op_input_names().find(op_name);
    if (it != op_input_names().end() && it->second.size() >= count) {
        return std::vector<std::string>(it->second.begin(), it->second.begin() + count);
    }
    std::vector<std::string> names;
    for (size_t i = 0; i < count; ++i)
        names.push_back("I_" + std::to_string(i));
    return names;
}

static std::vector<std::string> get_output_port_names(const std::string& op_name, size_t count) {
    auto it = op_output_names().find(op_name);
    if (it != op_output_names().end() && it->second.size() >= count) {
        return std::vector<std::string>(it->second.begin(), it->second.begin() + count);
    }
    std::vector<std::string> names;
    for (size_t i = 0; i < count; ++i)
        names.push_back("O_" + std::to_string(i));
    return names;
}

static void set_attr(OpDesc::Attr* attr, const json& a_obj) {
    const auto& at_node = a_obj.at("AT");
    const std::string at_type = at_node.at("#").get<std::string>();

    if (at_type == "0.a_str") {
        attr->set_type(AttrType::STRING);
        attr->set_s(at_node.at("D").get<std::string>());
    } else if (at_type == "0.a_i32") {
        attr->set_type(AttrType::INT);
        attr->set_i(at_node.at("D").get<int32_t>());
    } else if (at_type == "0.a_i64") {
        attr->set_type(AttrType::LONG);
        attr->set_l(at_node.at("D").get<int64_t>());
    } else if (at_type == "0.a_f32") {
        attr->set_type(AttrType::FLOAT);
        attr->set_f(at_node.at("D").get<float>());
    } else if (at_type == "0.a_bool") {
        attr->set_type(AttrType::BOOLEAN);
        attr->set_b(at_node.at("D").get<bool>());
    } else if (at_type == "0.a_array") {
        const auto& arr = at_node.at("D");
        if (arr.empty()) {
            attr->set_type(AttrType::INTS);
            return;
        }
        const std::string elem_type = arr[0].at("#").get<std::string>();
        if (elem_type == "0.a_i32") {
            attr->set_type(AttrType::INTS);
            for (auto& e : arr)
                attr->add_ints(e.at("D").get<int32_t>());
        } else if (elem_type == "0.a_i64") {
            attr->set_type(AttrType::LONGS);
            for (auto& e : arr)
                attr->add_longs(e.at("D").get<int64_t>());
        } else if (elem_type == "0.a_f32") {
            attr->set_type(AttrType::FLOATS);
            for (auto& e : arr)
                attr->add_floats(e.at("D").get<float>());
        } else if (elem_type == "0.a_bool") {
            attr->set_type(AttrType::BOOLEANS);
            for (auto& e : arr)
                attr->add_bools(e.at("D").get<bool>());
        } else if (elem_type == "0.a_str") {
            attr->set_type(AttrType::STRINGS);
            for (auto& e : arr)
                attr->add_strings(e.at("D").get<std::string>());
        }
    } else if (at_type == "1.a_intarray") {
        attr->set_type(AttrType::INTS);
        const auto& d = at_node.at("D");
        if (d.is_array()) {
            for (auto& e : d)
                attr->add_ints(e.get<int32_t>());
        }
    } else if (at_type == "1.a_dtype") {
        attr->set_type(AttrType::INT);
        attr->set_i(static_cast<int>(dtype_to_proto(at_node.at("D").get<std::string>())));
    } else if (at_type == "1.a_place") {
        attr->set_type(AttrType::INT);
        attr->set_i(0);
    }
}

static void fill_var_desc_from_output(VarDesc* var, const json& output_node) {
    var->set_name(var_name(output_node.at("%").get<int>()));
    auto* var_type = var->mutable_type();
    var_type->set_type(VarType_Type_LOD_TENSOR);
    auto* lod = var_type->mutable_lod_tensor();
    auto* tensor = lod->mutable_tensor();

    if (output_node.contains("TT")) {
        const auto& tt = output_node.at("TT");
        const auto& d = tt.at("D");
        if (d.is_array() && d.size() >= 2) {
            if (d[0].is_object() && d[0].contains("#")) {
                tensor->set_data_type(type_tag_to_proto(d[0].at("#").get<std::string>()));
            }
            // d[1] = shape array  [N, C, H, W] or [-1, 3, -1, -1]
            if (d[1].is_array()) {
                for (auto& dim : d[1]) {
                    tensor->add_dims(dim.get<int64_t>());
                }
            }
        }
    }
}

struct FullIntArrayConst {
    std::vector<int64_t> values;
};

static std::map<int, FullIntArrayConst> scan_full_int_arrays(const json& ops) {
    std::map<int, FullIntArrayConst> m;
    for (auto& op : ops) {
        const std::string raw_name = op.at("#").get<std::string>();
        if (strip_op_name(raw_name) != "full_int_array")
            continue;
        // Get output value ID
        const auto& outputs = op.at("O");
        int out_id = -1;
        if (outputs.is_array() && !outputs.empty()) {
            out_id = outputs[0].at("%").get<int>();
        } else if (outputs.is_object()) {
            out_id = outputs.at("%").get<int>();
        }
        if (out_id < 0)
            continue;
        FullIntArrayConst c;
        if (op.contains("A") && op.at("A").is_array()) {
            for (auto& attr : op.at("A")) {
                if (attr.contains("N") && attr.at("N").get<std::string>() == "value") {
                    const auto& at_node = attr.at("AT");
                    if (at_node.at("#").get<std::string>() == "0.a_array") {
                        for (auto& e : at_node.at("D")) {
                            if (e.contains("D"))
                                c.values.push_back(e.at("D").get<int64_t>());
                        }
                    }
                }
            }
        }
        m[out_id] = c;
    }
    return m;
}

std::shared_ptr<ProgramDesc> json_to_program_desc(std::istream& json_stream) {
    json root = json::parse(json_stream);

    FRONT_END_GENERAL_CHECK(root.contains("program"), "PIR JSON has no 'program' key");
    const auto& program = root.at("program");
    FRONT_END_GENERAL_CHECK(program.contains("regions") && program.at("regions").is_array(),
                            "PIR JSON has no 'regions' array");
    const auto& regions = program.at("regions");

    auto desc = std::make_shared<ProgramDesc>();

    for (auto& region : regions) {
        if (!region.contains("blocks") || !region.at("blocks").is_array())
            continue;
        for (auto& block_json : region.at("blocks")) {
            if (!block_json.contains("ops") || !block_json.at("ops").is_array())
                continue;

            auto* block = desc->add_blocks();
            block->set_idx(desc->blocks_size() - 1);
            block->set_parent_idx(0);

            const auto& ops_json = block_json.at("ops");

            auto fia_map = scan_full_int_arrays(ops_json);

            std::map<int, std::string> param_vars;
            for (auto& op_json : ops_json) {
                std::string raw = op_json.at("#").get<std::string>();
                if (raw != "p")
                    continue;

                std::string weight_name;
                const auto& a_field = op_json.at("A");
                if (a_field.is_array() && a_field.size() >= 4 && a_field[3].is_string()) {
                    weight_name = a_field[3].get<std::string>();
                }

                const auto& o = op_json.at("O");
                int out_id = o.at("%").get<int>();

                param_vars[out_id] = weight_name;

                auto* var = block->add_vars();
                var->set_name(weight_name);
                var->set_persistable(true);

                auto* var_type = var->mutable_type();
                var_type->set_type(VarType_Type_LOD_TENSOR);
                auto* lod = var_type->mutable_lod_tensor();
                auto* tensor = lod->mutable_tensor();

                if (o.contains("TT")) {
                    const auto& tt = o.at("TT");
                    const auto& d = tt.at("D");
                    if (d.is_array() && d.size() >= 2) {
                        if (d[0].is_object() && d[0].contains("#"))
                            tensor->set_data_type(type_tag_to_proto(d[0].at("#").get<std::string>()));
                        if (d[1].is_array()) {
                            for (auto& dim : d[1])
                                tensor->add_dims(dim.get<int64_t>());
                        }
                    }
                }
            }

            std::map<int, std::string> value_to_var;
            for (auto& [id, wname] : param_vars) {
                value_to_var[id] = wname;
            }

            for (auto& op_json : ops_json) {
                std::string raw = op_json.at("#").get<std::string>();
                std::string name = strip_op_name(raw);

                if (raw == "p")
                    continue;

                if (name == "full_int_array") {
                    const auto& outputs = op_json.at("O");
                    if (outputs.is_array()) {
                        for (auto& o : outputs) {
                            int oid = o.at("%").get<int>();
                            std::string vname = var_name(oid);
                            value_to_var[oid] = vname;
                            auto* var = block->add_vars();
                            var->set_name(vname);
                            var->set_persistable(false);
                            fill_var_desc_from_output(var, o);
                        }
                    } else if (outputs.is_object()) {
                        int oid = outputs.at("%").get<int>();
                        std::string vname = var_name(oid);
                        value_to_var[oid] = vname;
                        auto* var = block->add_vars();
                        var->set_name(vname);
                        var->set_persistable(false);
                        fill_var_desc_from_output(var, outputs);
                    }
                    continue;
                }

                std::string proto_op_type = name;
                if (name == "data") {
                    proto_op_type = "feed";
                } else if (name == "fetch") {
                    proto_op_type = "fetch";
                }

                auto* op_desc = block->add_ops();
                op_desc->set_type(proto_op_type);

                std::vector<int> input_ids;
                if (op_json.contains("I") && op_json.at("I").is_array()) {
                    for (auto& i_item : op_json.at("I")) {
                        if (i_item.is_object() && i_item.contains("%")) {
                            input_ids.push_back(i_item.at("%").get<int>());
                        }
                    }
                }

                if (proto_op_type == "pool2d" && input_ids.size() == 2) {
                    int fia_id = input_ids[1];
                    auto fia_it = fia_map.find(fia_id);
                    if (fia_it != fia_map.end()) {
                        auto* ksize_attr = op_desc->add_attrs();
                        ksize_attr->set_name("ksize");
                        ksize_attr->set_type(AttrType::INTS);
                        for (auto v : fia_it->second.values)
                            ksize_attr->add_ints(static_cast<int32_t>(v));
                    }
                    input_ids.resize(1);
                }

                auto in_names = get_input_port_names(proto_op_type, input_ids.size());
                if (proto_op_type == "feed") {
                } else if (proto_op_type == "fetch") {
                    if (!input_ids.empty()) {
                        auto* inp = op_desc->add_inputs();
                        inp->set_parameter("X");
                        std::string vname;
                        auto vit = value_to_var.find(input_ids[0]);
                        if (vit != value_to_var.end())
                            vname = vit->second;
                        else
                            vname = var_name(input_ids[0]);
                        inp->add_arguments(vname);
                    }
                } else {
                    for (size_t i = 0; i < input_ids.size(); ++i) {
                        auto* inp = op_desc->add_inputs();
                        inp->set_parameter(in_names[i]);
                        std::string vname;
                        auto vit = value_to_var.find(input_ids[i]);
                        if (vit != value_to_var.end())
                            vname = vit->second;
                        else
                            vname = var_name(input_ids[i]);
                        inp->add_arguments(vname);
                    }
                }

                std::vector<int> output_ids;
                const auto& outputs = op_json.contains("O") ? op_json.at("O") : json();
                if (outputs.is_array()) {
                    for (auto& o : outputs)
                        output_ids.push_back(o.at("%").get<int>());
                } else if (outputs.is_object() && outputs.contains("%")) {
                    output_ids.push_back(outputs.at("%").get<int>());
                }

                if (proto_op_type == "feed") {
                    if (!output_ids.empty()) {
                        int oid = output_ids[0];
                        std::string out_vname = var_name(oid);
                        value_to_var[oid] = out_vname;

                        auto* outp = op_desc->add_outputs();
                        outp->set_parameter("Out");
                        outp->add_arguments(out_vname);

                        auto* var = block->add_vars();
                        var->set_name(out_vname);
                        var->set_persistable(false);
                        if (outputs.is_array() && !outputs.empty())
                            fill_var_desc_from_output(var, outputs[0]);
                        else if (outputs.is_object())
                            fill_var_desc_from_output(var, outputs);
                    }
                } else if (proto_op_type == "fetch") {
                } else {
                    auto out_names = get_output_port_names(proto_op_type, output_ids.size());
                    for (size_t i = 0; i < output_ids.size(); ++i) {
                        int oid = output_ids[i];
                        std::string out_vname = var_name(oid);
                        value_to_var[oid] = out_vname;

                        auto* outp = op_desc->add_outputs();
                        outp->set_parameter(out_names[i]);
                        outp->add_arguments(out_vname);

                        auto* var = block->add_vars();
                        var->set_name(out_vname);
                        var->set_persistable(false);
                        if (outputs.is_array() && i < outputs.size())
                            fill_var_desc_from_output(var, outputs[i]);
                    }
                }

                if (op_json.contains("A") && op_json.at("A").is_array()) {
                    for (auto& a : op_json.at("A")) {
                        if (!a.is_object() || !a.contains("N") || !a.contains("AT"))
                            continue;
                        auto* attr = op_desc->add_attrs();
                        attr->set_name(a.at("N").get<std::string>());
                        set_attr(attr, a);
                    }
                }

                // OA (output attributes) are informational in PIR JSON; not needed for conversion

                if (proto_op_type == "feed") {
                    if (op_json.contains("A") && op_json.at("A").is_array()) {
                        for (auto& a : op_json.at("A")) {
                            if (!a.is_object())
                                continue;
                            if (a.contains("N") && a.at("N").get<std::string>() == "name") {
                                if (!output_ids.empty()) {
                                    const auto& at_node = a.at("AT");
                                    std::string feed_name = at_node.at("D").get<std::string>();
                                    int oid = output_ids[0];
                                    for (int vi = 0; vi < block->vars_size(); ++vi) {
                                        if (block->vars(vi).name() == var_name(oid)) {
                                            block->mutable_vars(vi)->set_name(feed_name);
                                        }
                                    }
                                    if (op_desc->outputs_size() > 0) {
                                        op_desc->mutable_outputs(0)->set_arguments(0, feed_name);
                                    }
                                    value_to_var[oid] = feed_name;
                                }
                            }
                        }
                    }
                }

            }
        }
    }

    return desc;
}
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
