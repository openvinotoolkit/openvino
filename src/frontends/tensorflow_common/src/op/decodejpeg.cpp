// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/decodeimg.hpp"
#include "openvino/op/random_uniform.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_decodejpeg_op(const NodeContext& node) {
    default_op_checks(node, 1,
                      {"DecodeBmp", "DecodeJpeg", "DecodePng", "DecodeGif",
                       "Decodebmp", "Decodejpeg", "Decodepng", "Decodegif",
                       "decodebmp", "decodejpeg", "decodepng", "decodegif"});
    auto input = node.get_input(0);
    auto dct_method_str = node.get_attribute<std::string>("dct_method", "");
    auto fancy_upscaling_str = node.get_attribute<std::string>("fancy_upscaling", "");
    auto ratio_str = node.get_attribute<std::string>("ratio", "");

    std::for_each(dct_method_str.begin(), dct_method_str.end(), [](char& c) {
        c = ::toupper(c);
    });
    std::for_each(fancy_upscaling_str.begin(), fancy_upscaling_str.end(), [](char& c) {
        c = ::toupper(c);
    });

    uint8_t dct_method = 1;
    uint8_t fancy_upscaling = 1;
    uint8_t ratio = 1;

    if (dct_method_str.compare("INTEGER_ACCURATE") == 0)
        dct_method = 0;
    if ((fancy_upscaling_str.compare("NO") == 0) ||
        (fancy_upscaling_str.compare("FALSE") == 0))
        fancy_upscaling = 0;
    if (!ratio_str.empty())
        ratio = std::atoi(ratio_str.c_str());

    std::cout << "$$$ translate_decodeimg_op : name=" << node.get_name() << ", op_type=" << node.get_op_type()
              << ", input size=" << node.get_input_size() << ", input0=" << input
              << ", dct_method_str=" << dct_method_str << "(" << std::to_string(dct_method) << ")"
              << ", fancy_upscaling=" << fancy_upscaling_str << "(" << std::to_string(fancy_upscaling) << ")"
              << ", ratio=" << ratio_str << "(" << std::to_string(ratio) << ")" << std::endl;

    auto res = make_shared<v0::DecodeImg>(input, dct_method, fancy_upscaling, ratio);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
