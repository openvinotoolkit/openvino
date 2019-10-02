# !/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

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
import argparse
import os
import shutil
import subprocess
import sys


def shell(cmd, env=None, cwd=None):
    kwargs = dict(cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    print('Running: "{}"'.format(' '.join(cmd)))
    p = subprocess.Popen(cmd, **kwargs)
    (stdout, stderr) = p.communicate()
    return p.returncode, stdout, stderr


def get_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_proto', required=True, help='Path to caffe.proto')
    parser.add_argument('--output', help='Directory where output file are generated',
                        default=os.path.dirname(os.path.realpath(__file__)))
    return parser


def build_proto(proto_file_path, python_path):
    retcode, out, err = shell(['protoc', '-h'])
    if retcode:
        print(err)
        return 1
    if not (os.path.exists(proto_file_path) and os.path.isfile(proto_file_path)):
        print('File {} does not exist'.format(proto_file_path))
        return 1
    proto_path = os.path.split(proto_file_path)[0]
    if not proto_path:
        proto_path = os.getcwd()

    proto_file = os.path.split(proto_file_path)[1]
    command = ['protoc', proto_file, '--python_out={}'.format(python_path)]

    retcode, out, err = shell(command, cwd=proto_path)

    if retcode:
        print('protoc exit with code {}'.format(retcode))
        print('protoc out: {}'.format(out.decode().strip('\n')))
        print('protoc error: {}'.format(err.decode()))
    else:
        python_file = '{}_pb2.py'.format(proto_file.split('.')[0])
        shutil.move(os.path.join(python_path, python_file), os.path.join(python_path, 'caffe_pb2.py'))
        print('File {} was generated in: {}'.format('caffe_pb2.py', python_path))
    return retcode


if __name__ == "__main__":
    if sys.version_info < (3, 0):
        print('Python version should be of version 3.5 or newer')
        sys.exit(1)
    argv = get_cli_parser().parse_args()
    proto_file_path = argv.input_proto
    python_path = argv.output
    if not os.path.exists(python_path):
        print("Output directory {} does not exist".format(python_path))
        sys.exit(1)
    status = build_proto(proto_file_path, python_path)
    exit(status)
