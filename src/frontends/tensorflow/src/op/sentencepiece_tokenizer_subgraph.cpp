// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/op/sentencepiece_tokenizer.hpp"
#include "tf_framework_node.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

std::shared_ptr<DecoderBase> extract_decoder (std::shared_ptr<Node> node) {
    auto fw_node = std::dynamic_pointer_cast<tensorflow::FrameworkNode>(node);
    std::cerr << "Node: " << node << "\n";
    if(!fw_node)
        std::exit(1);
    FRONT_END_GENERAL_CHECK(
        fw_node,
        "translate_sentensepiece_tokenizer_subgraph cannot recognize FW node in the pattern");
    return fw_node->get_decoder();
}

OutputVector translate_sentencepiece_tokenizer_subgraph(const NodeContext& node) {
    // This is a temporary extension that is designed to convert
    // entire small subgraph consisting of Sentencepiece Tokenizer.
    // Note that this translator `node` points to RaggedTensorToSparse node

    std::cerr << "[ TF FE ] Started translate_sentensepiece_tokenizer_subgraph\n";
    std::cerr << node.get_input(0) << "\n";

    // Take pointers to all node decoders involved in the conversion

    auto ragged_tensor_to_sparse = node.get_decoder();

    auto sentencepiece_tokenize_op = extract_decoder(node.get_input(0).get_node_shared_ptr());

    auto sentencepiece_op = extract_decoder(node.get_input(0).
                            get_node_shared_ptr()->    // SentencepieceTokenizeOp
                            get_input_node_shared_ptr(0));  // SentencepieceOp

    auto data_input = node.get_input(0).
                            get_node_shared_ptr()->
                            get_input_node_shared_ptr(1);

    return make_shared<SentencepieceTokenizerExtensionOp>(
        OutputVector{data_input})->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
