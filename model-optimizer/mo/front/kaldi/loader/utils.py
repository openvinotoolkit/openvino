"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import io

import numpy as np
import os
import struct

from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

end_of_nnet_tag = '</Nnet>'
end_of_component_tag = '<!EndOfComponent>'

supported_components = [
    'addshift',
    'affinecomponent',
    'affinecomponentpreconditionedonline',
    'affinetransform',
    'backproptruncationcomponent',
    'batchnormcomponent',
    'clipgradientcomponent',
    'convolutional1dcomponent',
    'convolutionalcomponent',
    'copy',
    'elementwiseproductcomponent',
    'fixedaffinecomponent',
    'linearcomponent',
    'logsoftmaxcomponent',
    'lstmnonlinearitycomponent',
    'lstmprojected',
    'lstmprojectedstreams',
    'maxpoolingcomponent',
    'naturalgradientaffinecomponent',
    'naturalgradientperelementscalecomponent',
    'noopcomponent',
    'normalizecomponent',
    'parallelcomponent',
    'pnormcomponent',
    'rectifiedlinearcomponent',
    'rescale',
    'sigmoid',
    'sigmoidcomponent',
    'softmax',
    'softmaxcomponent',
    'splicecomponent',
    'sumgroupcomponent',
    'tanhcomponent',
]


def get_bool(s: bytes) -> bool:
    """
    Get bool value from bytes
    :param s: bytes array contains bool value
    :return: bool value from bytes array
    """
    if str(s) == "b\'F\'":
        return False
    elif str(s) == "b\'T\'":
        return True
    else:
        return struct.unpack('?', s)[0]


def get_uint16(s: bytes) -> int:
    """
    Get unsigned int16 value from bytes
    :param s: bytes array contains unsigned int16 value
    :return: unsigned int16 value from bytes array
    """
    return struct.unpack('H', s)[0]


def get_uint32(s: bytes) -> int:
    """
    Get unsigned int32 value from bytes
    :param s: bytes array contains unsigned int32 value
    :return: unsigned int32 value from bytes array
    """
    return struct.unpack('I', s)[0]


def get_uint64(s: bytes) -> int:
    """
    Get unsigned int64 value from bytes
    :param s: bytes array contains unsigned int64 value
    :return: unsigned int64 value from bytes array
    """
    return struct.unpack('q', s)[0]


def read_binary_bool_token(file_desc: io.BufferedReader) -> bool:
    """
    Get next bool value from file
    The carriage moves forward to 1 position.
    :param file_desc: file descriptor
    :return: next boolean value in file
    """
    return get_bool(file_desc.read(1))


def read_binary_integer32_token(file_desc: io.BufferedReader) -> int:
    """
    Get next int32 value from file
    The carriage moves forward to 5 position.
    :param file_desc: file descriptor
    :return: next uint32 value in file
    """
    buffer_size = file_desc.read(1)
    return get_uint32(file_desc.read(buffer_size[0]))


def read_binary_integer64_token(file_desc: io.BufferedReader) -> int:
    """
    Get next int64 value from file
    The carriage moves forward to 9 position.
    :param file_desc: file descriptor
    :return: next uint64 value in file
    """
    buffer_size = file_desc.read(1)
    return get_uint64(file_desc.read(buffer_size[0]))


def read_binary_float_token(file_desc: io.BufferedReader) -> float:
    """
    Get next float32 value from file
    The carriage moves forward to 5 position.
    :param file_desc: file descriptor
    :return: next float32 value in file
    """
    buffer_size = file_desc.read(1)
    s = file_desc.read(buffer_size[0])
    return np.fromstring(s, dtype=np.float32)[0]


def read_string(file_desc: io.BufferedReader) -> int:
    return collect_until_whitespace(file_desc)


def find_next_tag(file_desc: io.BufferedReader) -> str:
    """
    Get next tag in the file
    :param file_desc:file descriptor
    :return: string like '<sometag>'
    """
    tag = b''
    while True:
        symbol = file_desc.read(1)
        if symbol == b'':
            raise Error('Unexpected end of Kaldi model')
        if tag == b'' and symbol != b'<':
            continue
        elif symbol == b'<':
            tag = b''
        tag += symbol
        if symbol != b'>':
            continue
        try:
            return tag.decode('ascii')
        except UnicodeDecodeError:
            # Tag in Kaldi model always in ascii encoding
            tag = b''


def read_placeholder(file_desc: io.BufferedReader, size=3) -> bytes:
    """
    Read size bytes from file
    :param file_desc:file descriptor
    :param size:number of reading bytes
    :return: bytes
    """
    return file_desc.read(size)


def find_next_component(file_desc: io.BufferedReader) -> str:
    """
    Read next component in the file.
    All components are contained in supported_components
    :param file_desc:file descriptor
    :return: string like '<component>'
    """
    is_start = True
    while True:
        tag = find_next_tag(file_desc)
        # Tag is <NameOfTheLayer>. But we want get without '<' and '>'
        component_name = tag[1:-1].lower()
        if component_name in supported_components or tag == end_of_nnet_tag:
            # There is whitespace after component's name
            read_placeholder(file_desc, 1)
            return component_name
        elif tag == '<ComponentName>':
            raise Error('Component has unsupported or not specified type')
        elif not (is_start and tag == end_of_component_tag) and tag.find('Component') != -1:
            raise Error('Component has unsupported type {}'.format(tag))
        is_start = False


def get_name_from_path(path: str) -> str:
    """
    Get name from path to the file
    :param path: path to the file
    :return: name of the file
    """
    return os.path.splitext(os.path.basename(path))[0]


def find_end_of_component(file_desc: io.BufferedReader, component: str, end_tags: tuple = ()):
    """
    Find an index and a tag of the ent of the component
    :param file_desc: file descriptor
    :param component: component from supported_components
    :param end_tags: specific end tags
    :return: the index and the tag of the end of the component
    """
    end_tags_of_component = ['</{}>'.format(component),
                             end_of_component_tag.lower(),
                             end_of_nnet_tag.lower(),
                             *end_tags,
                             *['<{}>'.format(component) for component in supported_components]]
    next_tag = find_next_tag(file_desc)
    while next_tag.lower() not in end_tags_of_component:
        next_tag = find_next_tag(file_desc)
    return next_tag, file_desc.tell()


def get_parameters(file_desc: io.BufferedReader, start_index: int, end_index: int):
    """
    Get part of file
    :param file_desc: file descriptor
    :param start_index: Index of the start reading
    :param end_index:  Index of the end reading
    :return: part of the file
    """
    file_desc.seek(start_index)
    buffer = file_desc.read(end_index - start_index)
    return io.BytesIO(buffer)


def read_token_value(file_desc: io.BufferedReader, token: bytes = b'', value_type: type = np.uint32):
    """
    Get value of the token.
    Read next token (until whitespace) and check if next teg equals token
    :param file_desc: file descriptor
    :param token: token
    :param value_type:  type of the reading value
    :return: value of the token
    """
    getters = {
        np.uint32: read_binary_integer32_token,
        np.uint64: read_binary_integer64_token,
        np.bool: read_binary_bool_token
    }
    current_token = collect_until_whitespace(file_desc)
    if token != b'' and token != current_token:
        raise Error('Can not load token {} from Kaldi model'.format(token) +
                    refer_to_faq_msg(94))
    return getters[value_type](file_desc)


def collect_until_whitespace(file_desc: io.BufferedReader):
    """
    Read from file until whitespace
    :param file_desc: file descriptor
    :return:
    """
    res = b''
    while True:
        new_sym = file_desc.read(1)
        if new_sym == b' ' or new_sym == b'':
            break
        res += new_sym
    return res


def collect_until_token(file_desc: io.BufferedReader, token):
    """
    Read from file until the token
    :param file_desc: file descriptor
    :param token: token that we find
    :return:
    """
    while True:
        # usually there is the following structure <CellDim> DIM<ClipGradient> VALUEFM
        res = collect_until_whitespace(file_desc)
        if res == token or res[-len(token):] == token:
            return
        size = 0
        if isinstance(file_desc, io.BytesIO):
            size = len(file_desc.getbuffer())
        elif isinstance(file_desc, io.BufferedReader):
            size = os.fstat(file_desc.fileno()).st_size
        if file_desc.tell() == size:
            raise Error('End of the file. Token {} not found. {}'.format(token, file_desc.tell()))


def collect_until_token_and_read(file_desc: io.BufferedReader, token, value_type: type = np.uint32):
    """
    Read from file until the token
    :param file_desc: file descriptor
    :param token: token to find and read
    :param value_type: type of value to read
    :return:
    """
    getters = {
        np.uint32: read_binary_integer32_token,
        np.uint64: read_binary_integer64_token,
        np.bool: read_binary_bool_token,
        np.string_: read_string
    }
    collect_until_token(file_desc, token)
    return getters[value_type](file_desc)


def create_edge_attrs(prev_layer_id: str, next_layer_id: str, in_port=0, out_port=0) -> dict:
    """
    Create common edge's attributes
    :param prev_layer_id: id of previous layer
    :param next_layer_id: id of next layer
    :param in_port: 'in' port
    :param out_port: 'out' port
    :return: dictionary contains common attributes for edge
    """
    return {
        'out': out_port,
        'in': in_port,
        'name': next_layer_id,
        'fw_tensor_debug_info': [(prev_layer_id, next_layer_id)],
        'in_attrs': ['in', 'name'],
        'out_attrs': ['out', 'name'],
        'data_attrs': ['fw_tensor_debug_info']
    }


def read_blob(file_desc: io.BufferedReader, size: int, dtype=np.float32):
    """
    Read blob from the file
    :param file_desc: file descriptor
    :param size: size of the blob
    :param dtype: type of values of the blob
    :return: np array contains blob
    """
    dsizes = {
        np.float32: 4,
        np.int32: 4
    }
    data = file_desc.read(size * dsizes[dtype])
    return np.fromstring(data, dtype=dtype)


def get_args_for_specifier(string):
    """
    Parse arguments in brackets and return list of arguments
    :param string: string in format (<arg1>, <arg2>, .., <argn>)
    :return: list with arguments
    """
    open_bracket = 1
    pos = 1
    args = []
    prev_arg_pos = 1
    while pos < len(string):
        pos_open = string.find(b'(', pos)
        pos_close = string.find(b')', pos)
        pos_sep = string.find(b',', pos)

        if pos_open == -1:
            if open_bracket == 1:
                args = args + string[prev_arg_pos:pos_close].replace(b' ', b'').split(b',')
                pos = len(string)
            else:
                open_bracket = open_bracket - 1
                while open_bracket > 1:
                    pos_close = string.find(b')', pos_close+1)
                    if pos_close != -1:
                        open_bracket = open_bracket - 1
                    else:
                        raise Error("Syntax error in model: incorrect number of brackets")
                args.append(string[prev_arg_pos:pos_close+1].strip())
                prev_arg_pos = string.find(b',', pos_close+1) + 1
                if prev_arg_pos != 0 and string[prev_arg_pos:-2].replace(b' ', b'').split(b',') != [b'']:
                    args = args + string[prev_arg_pos:-2].replace(b' ', b'').split(b',')
                pos = len(string)
        else:
            if pos_sep < pos_open and open_bracket == 1:
                pos_sep = string[pos_sep:pos_open].rfind(b',') + pos_sep
                args = args + string[prev_arg_pos:pos_sep].replace(b' ', b'').split(b',')
                prev_arg_pos = pos_sep + 1

            if pos_open < pos_close:
                open_bracket = open_bracket + 1
                pos = pos_open + 1
            else:
                open_bracket = open_bracket - 1
                if open_bracket == 1:
                    args.append(string[prev_arg_pos:pos_close + 1].strip())
                    prev_arg_pos = string.find(b',', pos_close+1) + 1
                    pos = prev_arg_pos
                else:
                    pos = pos_close + 1

    return args
