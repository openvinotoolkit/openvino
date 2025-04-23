// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <openvino/openvino.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/codec_xor.hpp"

std::shared_ptr<ov::Node> make_convolution(const ov::Output<ov::Node>& in,
                                           const ov::element::Type& type,
                                           const std::vector<size_t>& filter_size,
                                           const std::vector<size_t>& strides,
                                           const std::vector<ptrdiff_t>& pads_begin,
                                           const std::vector<ptrdiff_t>& pads_end,
                                           const std::vector<size_t>& dilations,
                                           const ov::op::PadType& auto_pad,
                                           size_t num_out_channels) {
    auto shape = in.get_partial_shape();
    ov::Shape filter_weights_shape = {num_out_channels, static_cast<size_t>(shape[1].get_length())};
    filter_weights_shape.insert(filter_weights_shape.end(), filter_size.begin(), filter_size.end());

    std::shared_ptr<ov::op::v0::Constant> filter_weights_node;
    auto tensor = ov::Tensor(type, filter_weights_shape);
    auto size = shape_size(filter_weights_shape);
    double default_value = 0.5;

    for (std::size_t i = 0; i < size; i++) {
        switch (type) {
        case ov::element::i8:
            tensor.data<ov::fundamental_type_for<ov::element::i8>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::i8>>(default_value);
            break;
        case ov::element::i16:
            tensor.data<ov::fundamental_type_for<ov::element::i16>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::i16>>(default_value);
            break;
        case ov::element::i32:
            tensor.data<ov::fundamental_type_for<ov::element::i32>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::i32>>(default_value);
            break;
        case ov::element::i64:
            tensor.data<ov::fundamental_type_for<ov::element::i64>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::i64>>(default_value);
            break;
        case ov::element::u8:
            tensor.data<ov::fundamental_type_for<ov::element::u8>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::u8>>(default_value);
            break;
        case ov::element::u16:
            tensor.data<ov::fundamental_type_for<ov::element::u16>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::u16>>(default_value);
            break;
        case ov::element::u32:
            tensor.data<ov::fundamental_type_for<ov::element::u32>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::u32>>(default_value);
            break;
        case ov::element::u64:
            tensor.data<ov::fundamental_type_for<ov::element::u64>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::u64>>(default_value);
            break;
        case ov::element::bf16:
            tensor.data<ov::fundamental_type_for<ov::element::bf16>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::bf16>>(default_value);
            break;
        case ov::element::f16:
            tensor.data<ov::fundamental_type_for<ov::element::f16>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::f16>>(default_value);
            break;
        case ov::element::f32:
            tensor.data<ov::fundamental_type_for<ov::element::f32>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::f32>>(default_value);
            break;
        case ov::element::f64:
            tensor.data<ov::fundamental_type_for<ov::element::f64>>()[i] =
                static_cast<ov::fundamental_type_for<ov::element::f64>>(default_value);
            break;
        default:
            ov::Exception::create(__FILE__,
                                  __LINE__,
                                  std::string("Not supported elment type: ") + type.get_type_name());
        }
    }

    filter_weights_node = std::make_shared<ov::op::v0::Constant>(tensor);

    return std::make_shared<ov::op::v1::Convolution>(in,
                                                     filter_weights_node,
                                                     strides,
                                                     pads_begin,
                                                     pads_end,
                                                     dilations,
                                                     auto_pad);
}

std::shared_ptr<ov::Model> make_conv_pool_relu() {
    ov::Shape input_shape = {1, 1, 32, 32};
    ov::element::Type type = ov::element::f32;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});

    ov::Shape const_shape = {input_shape[0], input_shape[2], input_shape[1], input_shape[3]};
    auto const1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, const_shape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});

    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(params.front(), const1, false);
    reshape1->set_friendly_name("Reshape_1");
    reshape1->output(0).get_tensor().set_names({"reshape1"});

    auto conv1 = make_convolution(reshape1, type, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, ov::op::PadType::EXPLICIT, 4);
    conv1->set_friendly_name("Conv_1");
    conv1->output(0).get_tensor().set_names({"conv"});

    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ov::op::v1::MaxPool>(conv1,
                                                       stride,
                                                       padB,
                                                       padE,
                                                       kernel,
                                                       ov::op::RoundingType::FLOOR,
                                                       ov::op::PadType::EXPLICIT);
    pool1->output(0).get_tensor().set_names({"pool"});
    pool1->set_friendly_name("Pool_1");

    auto relu1 = std::make_shared<ov::op::v0::Relu>(pool1);
    relu1->set_friendly_name("Relu_1");
    relu1->output(0).get_tensor().set_names({"relu"});

    ov::Shape reluShape = relu1->outputs()[0].get_tensor().get_shape();
    std::vector<size_t> constShape2 = {1, ov::shape_size(reluShape)};
    auto const2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, constShape2);
    const2->output(0).get_tensor().set_names({"const2"});
    const2->set_friendly_name("Const_2");

    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(relu1, const2, false);
    reshape2->output(0).get_tensor().set_names({"reshape2"});
    reshape2->set_friendly_name("Reshape_2");

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reshape2)};
    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("dummy_model");
    return model;
}

int main(int argc, char* argv[]) {
    ov::Core core;
    auto tensor = ov::read_tensor_data("C:\\Work\\VPUX\\vpux-large-models\\Networks\\Llama2\\openvino2024_3\\llama2_7B_chat_4SymW16A_kvcache_staticshape_eager\\openvino_model.blob");  // if we don't test this, uncomment lines 171 and 196
    ov::SharedStreamBuffer streambuf(reinterpret_cast<char*>(tensor.data()), tensor.get_byte_size());
    {
        // auto compiledModel = core.compile_model(make_conv_pool_relu(), "NPU", {});

        std::stringstream sstream;
        dynamic_cast<std::iostream*>(&sstream)->rdbuf(&streambuf);

        // below code needed by sstream.str(), std::stringstream implementation has its own std::stringbuf, other than
        // std::streambuf which is found in std::ios
        /* ov::SharedStreamBuffer copyStreamBuf(streambuf);
        std::stringbuf strbuf;
        copyStreamBuf.swap(*dynamic_cast<std::streambuf*>(&strbuf));
        sstream.rdbuf()->swap(strbuf); */

        // below code needed by operator>> str whitespaces fix
        auto& f = std::use_facet<std::iostream::_Ctype>(sstream.getloc());
        for (size_t i = 0; i < f.table_size; ++i) {
            static_cast<std::iostream::_Ctype::mask>(f.table()[i]) = ~f.space;
        }

        auto compiledModel = core.import_model(sstream,
                                               "NPU",
                                               {ov::intel_npu::disable_version_check(true),
                                                ov::intel_npu::defer_weights_load(true),
                                                ov::hint::compiled_blob(tensor)});

        // std::ofstream outputCryptedBlob("crypted_blob.blob", std::ios::out | std::ios::binary);
        // compiledModel.export_model(sstream);
        auto cryptedBlob = ov::util::codec_xor(sstream.str());
        // outputCryptedBlob.write(cryptedBlob.c_str(), cryptedBlob.size());
    }

    {
        std::ifstream inputCryptedBlob("crypted_blob.blob", std::ios::in | std::ios::binary);
        inputCryptedBlob.seekg(0, std::ios::end);
        std::string decryptBuffer(inputCryptedBlob.tellg(), '\0');
        inputCryptedBlob.seekg(0, std::ios::beg);

        inputCryptedBlob.read(decryptBuffer.data(), decryptBuffer.size());
        auto decryptedBlobSO = std::make_shared<std::string>(ov::util::codec_xor(decryptBuffer));
        decryptBuffer.resize(0);
        decryptBuffer.shrink_to_fit();

        tensor = ov::Tensor(ov::element::u8, ov::Shape{decryptedBlobSO->size()}, decryptedBlobSO->data());
        auto impl = ov::get_tensor_impl(tensor);
        impl._so = decryptedBlobSO;
        tensor = ov::make_tensor(impl);

        std::stringstream sstream;
        streambuf = ov::SharedStreamBuffer(reinterpret_cast<char*>(tensor.data()), tensor.get_byte_size());
        dynamic_cast<std::iostream*>(&sstream)->rdbuf(&streambuf);

        // below code needed by sstream.str(), std::stringstream implementation has its own std::stringbuf, other than
        // std::streambuf which is found in std::ios
        /* ov::SharedStreamBuffer copyStreamBuf(streambuf);
        std::stringbuf strbuf;
        copyStreamBuf.swap(*dynamic_cast<std::streambuf*>(&strbuf));
        sstream.rdbuf()->swap(strbuf); */

        // needed by operator>> str whitespaces fix
        auto& f = std::use_facet<std::iostream::_Ctype>(sstream.getloc());
        for (size_t i = 0; i < f.table_size; ++i) {
            static_cast<std::iostream::_Ctype::mask>(f.table()[i]) = ~f.space;
        }

        auto compiledModel = core.import_model(sstream,
                                               "NPU",
                                               ov::AnyMap{ov::intel_npu::defer_weights_load(true),
                                                          ov::intel_npu::disable_version_check(true),
                                                          ov::hint::compiled_blob(tensor)});

        // Test #1, operator>> is used
        std::string str;
        // the below statement will malfunction on Windows due to:
        /*
            C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\include\xstring#L3400-3401
            } else if (_Ctype_fac.is(_Ctype::space, _Traits::to_char_type(_Meta))) {
                break; // whitespace, quit

        */
        // potential fix, custom locale for stringstream: https://en.cppreference.com/w/cpp/locale/ctype_char
        sstream >> str;
        // because whitespaces were eliminated, operator>> will set sstream state to 1 (eof), need to clear this flag
        sstream.clear(sstream.rdstate() & ~std::ios::eofbit);
        sstream.seekg(-sstream.tellg(), std::ios::cur);
        size_t size = str.size();
        str.resize(0);
        str.shrink_to_fit();

        // Test #2 sstream.read() is used
        str = std::string(size, '\0');
        sstream.read(str.data(), str.size());
        sstream.seekg(-sstream.tellg(), std::ios::cur);
        str.resize(0);
        str.shrink_to_fit();

        // Test #3 sstream.str() is used
        str = sstream.str();                        // current seekg won't be changed
        std::cout << sstream.tellg() << std::endl;  // expect 0
        str.resize(0);
        str.shrink_to_fit();

        // Test #4 write to stringstream works
        char c = 'Y';
        // write won't work unless `overflow` method is overriden
        sstream.write(&c, 1);  // for large blobs > 2GB this won't work due to INT_MAX = 2GB in `overflow` function
        // can be fixed in a custom streambuf as `overflow` function is virtual
        sstream.seekg(-1, std::ios::end);
        sstream.seekp(0, std::ios::end);
        std::cout << sstream.tellp() << std::endl;  // expect 78776
        std::cout << sstream.tellg() << std::endl;  // expect 0

        sstream.read(&c, 1);

        // TODO: // Test #... check the tests above for istringstream
    }

    return 0;
}
