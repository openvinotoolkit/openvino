// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <streambuf>
#include <vector>

namespace intel_npu::driverCompilerAdapter {

/**
 * @brief Custom stream buffer for reading and writing data of data larger than 2G on Windows
 */
class CustomStreamBuf : public std::streambuf {
public:
    CustomStreamBuf(size_t bufferSize = 1024) : _buffer(bufferSize), _readPos(0), _writePos(0) {
        // Update read and write pointers
        char* base = _buffer.data();
        this->setg(base, base, base);
        this->setp(base, base + _buffer.size());
    }

    std::streampos tellp() {
        return std::streampos(_writePos);
    }

    std::streampos tellg() {
        return std::streampos(_readPos);
    }

protected:
    virtual int overflow(int ch) override {
        if (ch != traits_type::eof()) {
            if (_writePos == _buffer.size()) {
                _buffer.push_back(ch);
            } else {
                _buffer[_writePos] = ch;
            }
            ++_writePos;
            this->setp(_buffer.data(), _buffer.data() + _buffer.size());
            this->pbump(static_cast<int>(_writePos));
            return ch;
        }
        return traits_type::eof();
    }

    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (_writePos + n > _buffer.size()) {
            _buffer.resize(_writePos + n);
        }
        std::copy(s, s + n, _buffer.data() + _writePos);
        _writePos += n;
        this->setp(_buffer.data(), _buffer.data() + _buffer.size());
        this->pbump(static_cast<int>(_writePos));
        return n;
    }

    virtual std::streamsize xsgetn(char* s, std::streamsize n) override {
        std::streamsize num = std::min(n, static_cast<std::streamsize>(_buffer.size() - _readPos));
        std::copy(_buffer.data() + _readPos, _buffer.data() + _readPos + num, s);
        _readPos += num;
        this->setg(_buffer.data(), _buffer.data() + _readPos, _buffer.data() + _buffer.size());
        return num;
    }

    virtual std::streampos seekoff(std::streamoff off,
                                   std::ios_base::seekdir way,
                                   std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override {
        if (which & std::ios_base::in) {
            switch (way) {
            case std::ios_base::beg:
                _readPos = off;
                break;
            case std::ios_base::cur:
                _readPos += off;
                break;
            case std::ios_base::end:
                _readPos = _buffer.size() + off;
            default:
                break;
            }

            if (_readPos > _buffer.size() || _readPos < 0) {
                return std::streampos(std::streamoff(-1));
            }
            this->setg(_buffer.data(), _buffer.data() + _readPos, _buffer.data() + _buffer.size());
            return std::streampos(_readPos);
        }

        if (which & std::ios_base::out) {
            switch (way) {
            case std::ios_base::beg:
                _writePos = off;
                break;
            case std::ios_base::cur:
                _writePos += off;
                break;
            case std::ios_base::end:
                _writePos = _buffer.size() + off;
            default:
                break;
            }

            if (_writePos > _buffer.size() || _writePos < 0) {
                return std::streampos(std::streamoff(-1));
            }
            this->setp(_buffer.data(), _buffer.data() + _buffer.size());
            this->pbump(static_cast<int>(_writePos));
            return std::streampos(_writePos);
        }
        return std::streampos(std::streamoff(-1));
    }

    virtual std::streampos seekpos(std::streampos pos,
                                   std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override {
        return seekoff(static_cast<std::streamoff>(pos), std::ios_base::beg, which);
    }

private:
    std::vector<char> _buffer;
    size_t _readPos;
    size_t _writePos;
};

}  // namespace intel_npu::driverCompilerAdapter
