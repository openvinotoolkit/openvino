/*
   Copyright 2017 Leon Merten Lohse

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NPY_H
#define NPY_H

#include <complex>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>


namespace npy {

/* Compile-time test for byte order.
   If your compiler does not define these per default, you may want to define
   one of these constants manually. 
   Defaults to little endian order. */
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || \
    defined(__BIG_ENDIAN__) || \
    defined(__ARMEB__) || \
    defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || \
    defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
const bool big_endian = true;
#else
const bool big_endian = false;
#endif


const char magic_string[] = "\x93NUMPY";
const size_t magic_string_length = 6;

const char little_endian_char = '<';
const char big_endian_char = '>';
const char no_endian_char = '|';

constexpr char host_endian_char = ( big_endian ? 
    big_endian_char : 
    little_endian_char );

/* npy array length */
typedef unsigned long int ndarray_len_t;

inline void write_magic(std::ostream& ostream, unsigned char v_major=1, unsigned char v_minor=0) {
  ostream.write(magic_string, magic_string_length);
  ostream.put(v_major);
  ostream.put(v_minor);
}

inline void read_magic(std::istream& istream, unsigned char& v_major, unsigned char& v_minor) {
  char buf[magic_string_length+2];
  istream.read(buf, magic_string_length+2);

  if(!istream) {
    throw std::runtime_error("io error: failed reading file");
  }

  if (0 != std::memcmp(buf, magic_string, magic_string_length))
    throw std::runtime_error("this file does not have a valid npy format.");

  v_major = buf[magic_string_length];
  v_minor = buf[magic_string_length+1];
}

// typestring magic
struct Typestring {
  private:
    char c_endian;
    char c_type;
    int  len;

  public:
    inline std::string str() {
      const size_t max_buflen = 16;
      char buf[max_buflen];
      std::sprintf(buf, "%c%c%u", c_endian, c_type, len);
      return std::string(buf);
    }

    Typestring(const std::vector<float>& v) 
      :c_endian {host_endian_char}, c_type {'f'}, len {sizeof(float)} {}
    Typestring(const std::vector<double>& v) 
      :c_endian {host_endian_char}, c_type {'f'}, len {sizeof(double)} {}
    Typestring(const std::vector<long double>& v) 
      :c_endian {host_endian_char}, c_type {'f'}, len {sizeof(long double)} {}

    Typestring(const std::vector<char>& v) 
      :c_endian {no_endian_char}, c_type {'i'}, len {sizeof(char)} {}
    Typestring(const std::vector<short>& v) 
      :c_endian {host_endian_char}, c_type {'i'}, len {sizeof(short)} {}
    Typestring(const std::vector<int>& v) 
      :c_endian {host_endian_char}, c_type {'i'}, len {sizeof(int)} {}
    Typestring(const std::vector<long>& v)
      :c_endian {host_endian_char}, c_type {'i'}, len {sizeof(long)} {}
    Typestring(const std::vector<long long>& v) :c_endian {host_endian_char}, c_type {'i'}, len {sizeof(long long)} {}

    Typestring(const std::vector<unsigned char>& v)
      :c_endian {no_endian_char}, c_type {'u'}, len {sizeof(unsigned char)} {}
    Typestring(const std::vector<unsigned short>& v)
      :c_endian {host_endian_char}, c_type {'u'}, len {sizeof(unsigned short)} {}
    Typestring(const std::vector<unsigned int>& v)
      :c_endian {host_endian_char}, c_type {'u'}, len {sizeof(unsigned int)} {}
    Typestring(const std::vector<unsigned long>& v)
      :c_endian {host_endian_char}, c_type {'u'}, len {sizeof(unsigned long)} {}
    Typestring(const std::vector<unsigned long long>& v)
      :c_endian {host_endian_char}, c_type {'u'}, len {sizeof(unsigned long long)} {}

    Typestring(const std::vector<std::complex<float>>& v)
      :c_endian {host_endian_char}, c_type {'c'}, len {sizeof(std::complex<float>)} {}
    Typestring(const std::vector<std::complex<double>>& v)
      :c_endian {host_endian_char}, c_type {'c'}, len {sizeof(std::complex<double>)} {}
    Typestring(const std::vector<std::complex<long double>>& v)
      :c_endian {host_endian_char}, c_type {'c'}, len {sizeof(std::complex<long double>)} {}
};

inline void parse_typestring( std::string typestring){
//  std::regex re ("'([<>|])([ifuc])(\\d+)'");
//  std::smatch sm;
//
//  std::regex_match(typestring, sm, re );
//
//  if ( sm.size() != 4 ) {
//    throw std::runtime_error("invalid typestring");
//  }
}

namespace pyparse {

/**
  Removes leading and trailing whitespaces
  */
inline std::string trim(const std::string& str) {
  const std::string whitespace = " \t";
  auto begin = str.find_first_not_of(whitespace);

  if (begin == std::string::npos)
    return "";

  auto end = str.find_last_not_of(whitespace);

  return str.substr(begin, end-begin+1);
}


inline std::string get_value_from_map(const std::string& mapstr) {
  size_t sep_pos = mapstr.find_first_of(":");
  if (sep_pos == std::string::npos)
    return "";

  std::string tmp = mapstr.substr(sep_pos+1);
  return trim(tmp);
}

/**
   Parses the string representation of a Python dict

   The keys need to be known and may not appear anywhere else in the data.
 */
inline std::unordered_map<std::string, std::string> parse_dict(std::string in, std::vector<std::string>& keys) {

  std::unordered_map<std::string, std::string> map;

  if (keys.size() == 0)
    return map;

  in = trim(in);

  // unwrap dictionary
  if ((in.front() == '{') && (in.back() == '}'))
    in = in.substr(1, in.length()-2);
  else
    throw std::runtime_error("Not a Python dictionary.");

  std::vector<std::pair<size_t, std::string>> positions;

  for (auto const& value : keys) {
    size_t pos = in.find( "'" + value + "'" );

    if (pos == std::string::npos)
      throw std::runtime_error("Missing '"+value+"' key.");

    std::pair<size_t, std::string> position_pair { pos, value };
    positions.push_back(position_pair);
  }

  // sort by position in dict
  std::sort(positions.begin(), positions.end() );

  for(size_t i = 0; i < positions.size(); ++i) {
    std::string raw_value;
    size_t begin { positions[i].first };
    size_t end { std::string::npos };

    std::string key = positions[i].second;

    if ( i+1 < positions.size() )
      end = positions[i+1].first;

    raw_value = in.substr(begin, end-begin);

    raw_value = trim(raw_value);

    if (raw_value.back() == ',')
      raw_value.pop_back();

    map[key] = get_value_from_map(raw_value);
  }

  return map;
}

/**
  Parses the string representation of a Python boolean
  */
inline bool parse_bool(const std::string& in) {
  if (in == "True")
    return true;
  if (in == "False")
    return false;

  throw std::runtime_error("Invalid python boolan.");
}

/**
  Parses the string representation of a Python str
  */
inline std::string parse_str(const std::string& in) {
  if ((in.front() == '\'') && (in.back() == '\''))
    return in.substr(1, in.length()-2);

  throw std::runtime_error("Invalid python string.");
}

/**
  Parses the string represenatation of a Python tuple into a vector of its items
 */
inline std::vector<std::string> parse_tuple(std::string in) {
  std::vector<std::string> v;
  const char seperator = ',';

  in = trim(in);

  if ((in.front() == '(') && (in.back() == ')'))
    in = in.substr(1, in.length()-2);
  else
    throw std::runtime_error("Invalid Python tuple.");

  std::istringstream iss(in);

  for (std::string token; std::getline(iss, token, seperator);) {
      v.push_back(token);
  }

  return v;
}

template <typename T>
inline std::string write_tuple(const std::vector<T>& v) {
  if (v.size() == 0)
    return "";

  std::ostringstream ss;

  if (v.size() == 1) {
    ss << "(" << v.front() << ",)";
  } else {
    const std::string delimiter = ", ";
    // v.size() > 1
    ss << "(";
    std::copy(v.begin(), v.end()-1, std::ostream_iterator<T>(ss, delimiter.c_str()));
    ss << v.back();
    ss << ")";
  }

  return ss.str();
}

inline std::string write_boolean(bool b) {
  if(b)
    return "True";
  else
    return "False";
}

} // namespace pyparse


inline void parse_header(std::string header, std::string& descr, bool& fortran_order, std::vector<ndarray_len_t>& shape) {
  /*
     The first 6 bytes are a magic string: exactly "x93NUMPY".
     The next 1 byte is an unsigned byte: the major version number of the file format, e.g. x01.
     The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. x00. Note: the version of the file format is not tied to the version of the numpy package.
     The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN.
     The next HEADER_LEN bytes form the header data describing the array's format. It is an ASCII string which contains a Python literal expression of a dictionary. It is terminated by a newline ('n') and padded with spaces ('x20') to make the total length of the magic string + 4 + HEADER_LEN be evenly divisible by 16 for alignment purposes.
     The dictionary contains three keys:

     "descr" : dtype.descr
     An object that can be passed as an argument to the numpy.dtype() constructor to create the array's dtype.
     "fortran_order" : bool
     Whether the array data is Fortran-contiguous or not. Since Fortran-contiguous arrays are a common form of non-C-contiguity, we allow them to be written directly to disk for efficiency.
     "shape" : tuple of int
     The shape of the array.
     For repeatability and readability, this dictionary is formatted using pprint.pformat() so the keys are in alphabetic order.
   */

  // remove trailing newline
  if (header.back() != '\n')
    throw std::runtime_error("invalid header");
  header.pop_back();

  // parse the dictionary
  std::vector<std::string> keys { "descr", "fortran_order", "shape" };
  auto dict_map = npy::pyparse::parse_dict(header, keys);

  if (dict_map.size() == 0)
    throw std::runtime_error("invalid dictionary in header");

  std::string descr_s = dict_map["descr"];
  std::string fortran_s = dict_map["fortran_order"];
  std::string shape_s = dict_map["shape"];

  // TODO: extract info from typestring
  parse_typestring(descr_s);
  // remove 
  descr = npy::pyparse::parse_str(descr_s);

  // convert literal Python bool to C++ bool
  fortran_order = npy::pyparse::parse_bool(fortran_s);

  // parse the shape tuple
  auto shape_v = npy::pyparse::parse_tuple(shape_s);
  if (shape_v.size() == 0)
    throw std::runtime_error("invalid shape tuple in header");

  for ( auto item : shape_v ) {
    ndarray_len_t dim = static_cast<ndarray_len_t>(std::stoul(item));
    shape.push_back(dim);
  }
}


inline std::string write_header_dict(const std::string& descr, bool fortran_order, const std::vector<ndarray_len_t>& shape) {
    std::string s_fortran_order = npy::pyparse::write_boolean(fortran_order);
    std::string shape_s = npy::pyparse::write_tuple(shape);

    return "{'descr': '" + descr + "', 'fortran_order': " + s_fortran_order + ", 'shape': " + shape_s + ", }";
}

inline void write_header(std::ostream& out, const std::string& descr, bool fortran_order, const std::vector<ndarray_len_t>& shape_v)
{
    std::string header_dict = write_header_dict(descr, fortran_order, shape_v);

    size_t length = magic_string_length + 2 + 2 + header_dict.length() + 1;

    unsigned char version[2] = {1, 0};
    if (length >= 255*255) {
      length = magic_string_length + 2 + 4 + header_dict.length() + 1;
      version[0] = 2;
      version[1] = 0;
    }
    size_t padding_len = 16 - length % 16;
    std::string padding (padding_len, ' ');

    // write magic
    write_magic(out, version[0], version[1]);

    // write header length
    if (version[0] == 1 && version[1] == 0) {
      char header_len_le16[2];
      uint16_t header_len = header_dict.length() + padding.length() + 1;

      header_len_le16[0] = (header_len >> 0) & 0xff;
      header_len_le16[1] = (header_len >> 8) & 0xff;
      out.write(reinterpret_cast<char *>(header_len_le16), 2);
    }else{
      char header_len_le32[4];
      uint32_t header_len = header_dict.length() + padding.length() + 1;

      header_len_le32[0] = (header_len >> 0) & 0xff;
      header_len_le32[1] = (header_len >> 8) & 0xff;
      header_len_le32[2] = (header_len >> 16) & 0xff;
      header_len_le32[3] = (header_len >> 24) & 0xff;
      out.write(reinterpret_cast<char *>(header_len_le32), 4);
    }

    out << header_dict << padding << '\n';
}

inline std::string read_header(std::istream& istream) {
    // check magic bytes an version number
    unsigned char v_major, v_minor;
    read_magic(istream, v_major, v_minor);

    uint32_t header_length;
    if(v_major == 1 && v_minor == 0){

      char header_len_le16[2];
      istream.read(header_len_le16, 2);
      header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);

      if((magic_string_length + 2 + 2 + header_length) % 16 != 0) {
          // TODO: display warning
      }
    }else if(v_major == 2 && v_minor == 0) {
      char header_len_le32[4];
      istream.read(header_len_le32, 4);

      header_length = (header_len_le32[0] <<  0) | (header_len_le32[1] <<  8)
                    | (header_len_le32[2] << 16) | (header_len_le32[3] <<  24);

      if((magic_string_length + 2 + 4 + header_length) % 16 != 0) {
        // TODO: display warning
      }
    }else{
       throw std::runtime_error("unsupported file format version");
    }

    auto buf_v = std::vector<char>();
    buf_v.reserve(header_length);
    istream.read(buf_v.data(), header_length);
    std::string header(buf_v.data(), header_length);

    return header;
}

inline ndarray_len_t comp_size(const std::vector<ndarray_len_t>& shape) {
    ndarray_len_t size = 1;
    for (ndarray_len_t i : shape )
      size *= i;

    return size;
}

template<typename Scalar>
inline void SaveArrayAsNumpy( const std::string& filename, bool fortran_order, unsigned int n_dims, const unsigned long shape[], const std::vector<Scalar>& data)
{
    Typestring typestring_o(data);
    std::string typestring = typestring_o.str();

    std::ofstream stream( filename, std::ofstream::binary);
    if(!stream) {
        throw std::runtime_error("io error: failed to open a file.");
    }

    std::vector<ndarray_len_t> shape_v(shape, shape+n_dims);
    write_header(stream, typestring, fortran_order, shape_v);

    auto size = static_cast<size_t>(comp_size(shape_v));

    stream.write(reinterpret_cast<const char*>(data.data()), sizeof(Scalar) * size);
}


template<typename Scalar>
inline void LoadArrayFromNumpy(const std::string& filename, std::vector<unsigned long>& shape, std::vector<Scalar>& data)
{
    std::ifstream stream(filename, std::ifstream::binary);
    if(!stream) {
        throw std::runtime_error("io error: failed to open a file.");
    }

    std::string header = read_header(stream);

    // parse header
    bool fortran_order;
    std::string typestr;

    parse_header(header, typestr, fortran_order, shape);

    // check if the typestring matches the given one
    Typestring typestring_o {data};
    std::string expect_typestr = typestring_o.str();
    if (typestr != expect_typestr) {
      throw std::runtime_error("formatting error: typestrings not matching");
    }


    // compute the data size based on the shape
    auto size = static_cast<size_t>(comp_size(shape));
    data.resize(size);

    // read the data
    stream.read(reinterpret_cast<char*>(data.data()), sizeof(Scalar)*size);
}

} // namespace npy

#endif // NPY_H
