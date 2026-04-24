# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest 
from common.layer_test_class import check_ir_version 
from common.onnx_layer_test_class import OnnxRuntimeLayerTest
from common.onnx_layer_test_class import onnx_make_model 
import numpy as np
import onnx
from onnx import helper 
from onnx import TensorProto

rng = np.random.default_rng()

class TestTfIdfVectorizer(OnnxRuntimeLayerTest) : 
    def _prepare_input(self, inputs_info) : 
        assert 'input' in inputs_info 
        input_shape = inputs_info['input'] 
        inputs_data = {} 
        sample_data = rng.choice(self.strings_dictionary, input_shape) 
        inputs_data['input'] = sample_data 
        return inputs_data
    
    def create_net(self, shape, ir_version, strings_dictionary) : 

        self.strings_dictionary = strings_dictionary
        input = helper.make_tensor_value_info('input', TensorProto.STRING, shape) 
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 50])

        #IDF weights for each vocabulary entry(one float per vocab item)
        #In TF - IDF : output[i] = TF(i) * IDF[i]
        weights =[1.0] * 50
        #Pool mode:
        #"TF"      → raw term frequency
        #"IDF"     → 1 if term present, 0 otherwise(binary)
        #"TFIDF"   → TF * IDF weight
        pool_strings =["10", "11", "2025", "4846", "928d", "appdata", 
                       "bin", "cache", "consim", "content", "data", "db", 
                       "detectionscripts", "dll", "documents", "e0e95c6c", 
                       "e5538022226d", "exe", "extension", "f194", "files", 
                       "gameshow", "hasrequestresponse", "ibd", "innodb_redo", 
                       "intune", "json", "local", "log", "management", "microsoft", 
                       "minigames", "mysql", "nri", "png", "program", "programdata", 
                       "pyc", "qqgametempest", "roaming", "script", "system32", "temp", 
                       "tmp", "txt", "user", "users", "windows", "x86", "xml"] 
        pool_mode = "TF"

        #min_gram_length / max_gram_length : size of n - grams
        #For unigrams : both = 1
        min_gram_length = 1 
        max_gram_length = 1

        #ngram_counts : starting index of each gram level in the pool
        #For unigrams only : [0]
        ngram_counts =[0]

        #ngram_indexes : maps each n - gram to its output vector dimension
        #For unigrams : each token i maps to output index i
        ngram_indexes = list(range(50))
        node_def = onnx.helper.make_node(op_type = "TfIdfVectorizer", inputs =["input"], outputs =["output"], domain = "",)
        # ── required attributes ──────────────────────────────────────
        max_skip_count = 0, 
        #Create the graph(GraphProto)
        graph_def = helper.make_graph([node_def], 'test_model', [input], [output], )

        #Create the model(ModelProto)
        #Standard ONNX opset
        onnx_net = onnx_make_model(graph_def, producer_name = 'test_model')
        onnx_net.ir_version = 8
        
        ref_net = None 
        return onnx_net, ref_net

    @pytest.mark.parametrize("input_shape", [[2, 3]]) 
    @pytest.mark.parametrize("strings_dictionary", 
                             [['programdata', 'hasrequestresponse', ' detectionscripts', 'json']]) 
    @pytest.mark.nightly 
    
    def test_tfidfvectorizer(self, 
                            input_shape, 
                            strings_dictionary, 
                            ie_device, precision, 
                            ir_version, 
                            temp_dir) : 
        self._test(* self.create_net(shape = input_shape, 
                                     strings_dictionary = strings_dictionary, 
                                     ir_version = ir_version), 
                                     ie_device, precision, 
                                     ir_version, 
                                     temp_dir = temp_dir)
