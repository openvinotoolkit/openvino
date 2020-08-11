//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ngraph/ngraph_visibility.hpp>

// The CPIO file format can be found at
// https://www.mkssoftware.com/docs/man4/cpio.4.asp

namespace ngraph
{
    namespace cpio
    {
        class Header;
        class FileInfo;
        class Writer;
        class Reader;

        bool is_cpio(const std::string&);
        bool is_cpio(std::istream&);
    }
}

class ngraph::cpio::Header
{
public:
    uint16_t magic;
    uint16_t dev;
    uint16_t ino;
    uint16_t mode;
    uint16_t uid;
    uint16_t gid;
    uint16_t nlink;
    uint16_t rdev;
    uint32_t mtime;
    uint16_t namesize;
    uint32_t filesize;

    static Header read(std::istream&);
    static void write(std::ostream&, const std::string& name, uint32_t size);

private:
};

class NGRAPH_API ngraph::cpio::FileInfo
{
public:
    FileInfo(const std::string& name, size_t size, size_t offset)
        : m_name(name)
        , m_size(size)
        , m_offset(offset)
    {
    }
    const std::string& get_name() const;
    size_t get_size() const;
    size_t get_offset() const;

private:
    std::string m_name;
    size_t m_size;
    size_t m_offset;
};

class NGRAPH_API ngraph::cpio::Writer
{
public:
    Writer();
    Writer(std::ostream& out);
    Writer(const std::string& filename);
    ~Writer();

    void open(std::ostream& out);
    void open(const std::string& filename);
    void write(const std::string& file_name, const void* data, uint32_t size_in_bytes);

private:
    std::ostream* m_stream;
    std::ofstream m_my_stream;
};

class NGRAPH_API ngraph::cpio::Reader
{
public:
    Reader();
    Reader(std::istream& in);
    Reader(const std::string& filename);
    ~Reader();

    void open(std::istream& in);
    void open(const std::string& filename);
    void close();
    const std::vector<FileInfo>& get_file_info();
    bool read(const std::string& file_name, void* data, size_t size_in_bytes);
    std::vector<char> read(const FileInfo& info);

private:
    std::istream* m_stream;
    std::ifstream m_my_stream;
    std::vector<cpio::FileInfo> m_file_info;
};
