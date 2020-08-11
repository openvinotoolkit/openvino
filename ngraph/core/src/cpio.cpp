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

#include "ngraph/cpio.hpp"
#include "ngraph/log.hpp"

using namespace ngraph;
using namespace std;

static uint16_t read_u16(istream& stream, bool big_endian = false)
{
    uint8_t ch[2];
    uint16_t rc;

    stream.read(reinterpret_cast<char*>(&ch[0]), 2);
    if (big_endian)
    {
        rc = static_cast<uint16_t>((ch[0] << 8) + ch[1]);
    }
    else
    {
        rc = static_cast<uint16_t>((ch[1] << 8) + ch[0]);
    }

    return rc;
}

static uint32_t read_u32(istream& stream, bool big_endian = false)
{
    uint32_t rc;

    uint16_t sh[2];
    sh[0] = read_u16(stream, big_endian);
    sh[1] = read_u16(stream, big_endian);
    rc = (sh[0] << 16) + sh[1];

    return rc;
}

static void write_u16(ostream& stream, uint16_t value)
{
    const char* p = reinterpret_cast<const char*>(&value);
    stream.write(p, 2);
}

static void write_u32(ostream& stream, uint32_t value)
{
    uint16_t* v = reinterpret_cast<uint16_t*>(&value);
    write_u16(stream, v[1]);
    write_u16(stream, v[0]);
}

cpio::Header cpio::Header::read(istream& stream)
{
    uint8_t ch;
    stream.read(reinterpret_cast<char*>(&ch), 1);
    Header rc;
    switch (ch)
    {
    case 0x71: // Big Endian
        stream.read(reinterpret_cast<char*>(&ch), 1);
        if (ch != 0xC7)
        {
            throw runtime_error("CPIO magic error");
        }
        // magic value defined in CPIO spec
        rc.magic = 0x71C7;
        rc.dev = read_u16(stream, true);
        rc.ino = read_u16(stream, true);
        rc.mode = read_u16(stream, true);
        rc.uid = read_u16(stream, true);
        rc.gid = read_u16(stream, true);
        rc.nlink = read_u16(stream, true);
        rc.rdev = read_u16(stream, true);
        rc.mtime = read_u32(stream, true);
        rc.namesize = read_u16(stream, true);
        rc.filesize = read_u32(stream, true);
        break;
    case 0xC7: // Little Endian
        stream.read(reinterpret_cast<char*>(&ch), 1);
        if (ch != 0x71)
        {
            throw runtime_error("CPIO magic error");
        }
        // magic value defined in CPIO spec
        rc.magic = 0x71C7;
        rc.dev = read_u16(stream);
        rc.ino = read_u16(stream);
        rc.mode = read_u16(stream);
        rc.uid = read_u16(stream);
        rc.gid = read_u16(stream);
        rc.nlink = read_u16(stream);
        rc.rdev = read_u16(stream);
        rc.mtime = read_u32(stream);
        rc.namesize = read_u16(stream);
        rc.filesize = read_u32(stream);
        break;
    case '0': throw runtime_error("CPIO ASCII unsupported");
    default: throw runtime_error("CPIO invalid file");
    }

    return rc;
}

void cpio::Header::write(ostream& stream, const string& name, uint32_t size)
{
    // namesize includes the null string terminator so + 1
    uint16_t namesize = static_cast<uint16_t>(name.size()) + 1;
    write_u16(stream, 0x71C7);   // magic
    write_u16(stream, 0);        // dev
    write_u16(stream, 0);        // ino
    write_u16(stream, 0);        // mode
    write_u16(stream, 0);        // uid
    write_u16(stream, 0);        // gid
    write_u16(stream, 0);        // nlink
    write_u16(stream, 0);        // rdev
    write_u32(stream, 0);        // mtime
    write_u16(stream, namesize); // namesize
    write_u32(stream, size);     // filesize
    stream.write(name.c_str(), namesize + (namesize % 2));
}

cpio::Writer::Writer()
    : m_stream(nullptr)
{
}

cpio::Writer::Writer(ostream& out)
    : Writer()
{
    open(out);
}

cpio::Writer::Writer(const string& filename)
    : Writer()
{
    open(filename);
}

cpio::Writer::~Writer()
{
    write("TRAILER!!!", nullptr, 0);
    if (m_my_stream.is_open())
    {
        m_my_stream.close();
    }
}

void cpio::Writer::open(ostream& out)
{
    m_stream = &out;
}

void cpio::Writer::open(const string& filename)
{
    m_stream = &m_my_stream;
    m_my_stream.open(filename, ios_base::binary | ios_base::out);
}

void cpio::Writer::write(const string& record_name, const void* data, uint32_t size_in_bytes)
{
    if (m_stream)
    {
        Header::write(*m_stream, record_name, size_in_bytes);
        m_stream->write(static_cast<const char*>(data), size_in_bytes);
        if (size_in_bytes % 2)
        {
            char ch = 0;
            m_stream->write(&ch, 1);
        }
    }
    else
    {
        throw runtime_error("cpio writer output not set");
    }
}

cpio::Reader::Reader()
    : m_stream(nullptr)
{
}

cpio::Reader::Reader(istream& in)
    : Reader()
{
    open(in);
}

cpio::Reader::Reader(const string& filename)
    : Reader()
{
    open(filename);
}

cpio::Reader::~Reader()
{
}

void cpio::Reader::open(istream& in)
{
    m_stream = &in;
    m_stream->seekg(0, ios_base::beg);
}

void cpio::Reader::open(const string& filename)
{
    m_stream = &m_my_stream;
    m_my_stream.open(filename, ios_base::binary | ios_base::in);
}

void cpio::Reader::close()
{
    if (m_my_stream.is_open())
    {
        m_my_stream.close();
    }
}

const vector<cpio::FileInfo>& cpio::Reader::get_file_info()
{
    if (m_file_info.empty())
    {
        while (*m_stream)
        {
            Header header = Header::read(*m_stream);

            auto buffer = new char[header.namesize];
            m_stream->read(buffer, header.namesize);
            // namesize includes the null string terminator so -1
            string file_name = string(buffer, header.namesize - 1);
            delete[] buffer;
            // skip any pad characters
            if (header.namesize % 2)
            {
                m_stream->seekg(1, ios_base::cur);
            }

            if (file_name == "TRAILER!!!")
            {
                break;
            }

            size_t offset = m_stream->tellg();
            m_file_info.emplace_back(file_name, header.filesize, offset);

            m_stream->seekg((header.filesize % 2) + header.filesize, ios_base::cur);
        }
    }

    return m_file_info;
}

bool cpio::Reader::read(const string& file_name, void* data, size_t size_in_bytes)
{
    bool rc = false;
    for (const FileInfo& info : get_file_info())
    {
        if (info.get_name() == file_name)
        {
            if (size_in_bytes != info.get_size())
            {
                throw runtime_error("Buffer size does not match file size");
            }
            m_stream->seekg(info.get_offset(), ios_base::beg);
            m_stream->read(reinterpret_cast<char*>(data), size_in_bytes);
            rc = true;
            break;
        }
    }
    return rc;
}

vector<char> cpio::Reader::read(const FileInfo& info)
{
    vector<char> buffer(info.get_size());
    read(info.get_name(), buffer.data(), info.get_size());
    return buffer;
}

bool cpio::is_cpio(const string& path)
{
    ifstream in(path, ios_base::binary | ios_base::in);
    return is_cpio(in);
}

bool cpio::is_cpio(istream& in)
{
    size_t offset = in.tellg();
    in.seekg(0, ios_base::beg);
    bool rc = false;
    uint8_t ch;
    in.read(reinterpret_cast<char*>(&ch), 1);
    switch (ch)
    {
    case 0x71: // Big Endian
        in.read(reinterpret_cast<char*>(&ch), 1);
        if (ch == 0xC7)
        {
            rc = true;
        }
        break;
    case 0xC7: // Little Endian
        in.read(reinterpret_cast<char*>(&ch), 1);
        if (ch == 0x71)
        {
            rc = true;
        }
        break;
    default: break;
    }
    in.seekg(offset, ios_base::beg);
    return rc;
}

const string& cpio::FileInfo::get_name() const
{
    return m_name;
}

size_t cpio::FileInfo::get_size() const
{
    return m_size;
}

size_t cpio::FileInfo::get_offset() const
{
    return m_offset;
}
