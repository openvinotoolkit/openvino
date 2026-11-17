// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <streambuf>
#include <string>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace ov {
namespace hetero {
namespace tests {

// State of the mock device MOCKHSM, which exports a nested HSM container: how it was imported last.
struct MockHsmImport {
    enum class Source { none, stream, tensor };
    Source source = Source::none;
    uint64_t weights_offset = 0;          //!< Offset of the weights section in the nested container.
    size_t tensor_imports = 0;            //!< import_model(Tensor) calls, including failed ones.
    size_t stream_imports = 0;            //!< import_model(istream) calls, including failed ones.
    bool export_not_implemented = false;  //!< export_model throws ov::NotImplemented, like the Auto batch plugin.
};
MockHsmImport& mock_hsm_import();

// Output buffer that collects the written bytes and fails every seek.
class NonSeekableOutputBuffer : public std::streambuf {
public:
    const std::string& str() const {
        return _data;
    }

protected:
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        if (count <= 0) {
            return 0;
        }
        _data.append(data, static_cast<std::string::size_type>(count));
        return count;
    }

    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }
        _data.push_back(traits_type::to_char_type(ch));
        return ch;
    }

    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override {
        return pos_type(off_type(-1));
    }

    pos_type seekpos(pos_type, std::ios_base::openmode) override {
        return pos_type(off_type(-1));
    }

private:
    std::string _data;
};

// Input buffer that serves the given bytes and fails every seek, so tellg() reports -1.
class NonSeekableInputBuffer : public std::streambuf {
public:
    explicit NonSeekableInputBuffer(std::string data) : _data(std::move(data)) {
        setg(_data.data(), _data.data(), _data.data() + _data.size());
    }

private:
    std::string _data;
};

class HeteroTests : public ::testing::Test {
public:
    ov::Core core;

    void SetUp() override;
    void TearDown() override;

    std::shared_ptr<ov::Model> create_model_with_subtract(bool dynamic = false);
    std::shared_ptr<ov::Model> create_model_with_subtract_reshape(bool dynamic = false);
    std::shared_ptr<ov::Model> create_model_with_subtract_reshape_relu(bool dynamic = false);
    std::shared_ptr<ov::Model> create_model_with_reshape(bool dynamic = false);
    std::shared_ptr<ov::Model> create_model_with_subtract_shapeof_reshape(bool dynamic = false);
    std::shared_ptr<ov::Model> create_model_with_independent_parameter(bool dynamic = false);
    std::shared_ptr<ov::Model> create_model_with_multi_add();
    ov::Tensor create_and_fill_tensor(const ov::element::Type& type, const ov::Shape& shape);

private:
    template <class T>
    ov::Tensor create_tensor(const ov::element::Type& type, const ov::Shape& shape) {
        ov::Tensor tensor(type, shape);
        T* data = tensor.data<T>();
        for (size_t i = 0; i < tensor.get_size(); i++) {
            data[i] = static_cast<T>(i);
        }
        return tensor;
    }
    std::vector<std::shared_ptr<ov::IPlugin>> m_mock_plugins;
    std::shared_ptr<void> m_so;

    template <typename T>
    void reg_plugin_type(const std::string& device_name);

    void reg_plugin(std::shared_ptr<ov::IPlugin>& plugin);

    void clearMockPlugin();
};

}  // namespace tests
}  // namespace hetero
}  // namespace ov
