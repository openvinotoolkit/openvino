import logging as log

import re
from xml.etree.ElementTree import Element, SubElement

import numpy as np


class Layer:
    def __init__(self, xml_element=None, layer_name=None, layer_type=None, layer_id=0,
                 port_id=0, precision='FP32', inputs=None, attrs=None, get_out_shape_def=None,
                 framework_representation_def=None, ir_representation_def=None):
        self.weights_size = 0
        self.biases_size = 0

        self.weights_offset = 0
        self.biases_offset = 0

        self.weights = np.zeros(self.weights_size)
        self.biases = np.zeros(self.biases_size)

        self.framework_representation_def = framework_representation_def
        self.ir_representation_def = ir_representation_def

        if xml_element:
            Layer.load_from_xml(self, xml_element)
        else:
            self.type = layer_type
            self.id = layer_id
            self.precision = precision
            if layer_name:
                self.name = layer_name
            else:
                self.name = '{}{}'.format(layer_type, self.id)
            self.inputs = {}
            for k in inputs.keys():
                self.inputs[k] = []
                for i in inputs[k]:
                    self.inputs[k].append((port_id, i))
                    port_id += 1
            self.attrs = attrs
            self.outputs = {}
            if get_out_shape_def:
                out_shape = get_out_shape_def(self)
                self.outputs[self.name] = []
                if isinstance(out_shape, list):
                    for o in out_shape:
                        self.outputs[self.name].append((port_id, o))
                        port_id += 1
                else:
                    self.outputs[self.name].append((port_id, get_out_shape_def(self)))
            else:
                mes = 'Function get_out_shape_def for layer type {} is not define.'.format(self.type)
                log.error(mes)
                raise NotImplementedError(mes)

    # ------------------------------HELPS------------------------------------

    def __lt__(self, other):
        return self.id < other.id

    def get_type(self):
        return self.type

    def get_name(self):
        return self.name

    def get_id(self):
        return self.id

    def get_inputs_names(self):
        return list(sorted(self.inputs.keys()))

    def get_outputs_names(self):
        return list(self.outputs.keys())

    def get_out_name(self):
        return list(self.outputs.keys())[0]

    def get_inputs_shape(self, name):
        return self.inputs[name][0][1]

    def get_out_shapes(self):
        if isinstance(self.outputs[self.name], list):
            return [out[1] for out in self.outputs[self.name]]
        else:
            return [self.outputs[self.name][1]]

    def get_out_port_ids(self):
        if isinstance(self.outputs[self.get_out_name()], list):
            return [out[0] for out in self.outputs[self.get_out_name()]]
        else:
            return [self.outputs[self.get_out_name()][0]]

    def get_in_port_id(self, name_input):
        return self.inputs[name_input][0]

    def get_weights_size(self):
        return self.weights_size

    def get_biases_size(self):
        return self.biases_size

    def set_input(self, _input):
        self.inputs[_input.get_name()] = [(_input.id, _input.outputs[_input.name][0][1])]

    def set_out_shape(self, port_id, calc_shape):
        self.outputs = {self.name: [(port_id, calc_shape(self))]}

    # ----------------------------End HELPS----------------------------------

    # ---------------Functions for load layer's params from xml--------------

    def load_from_xml(self, xml_element: Element):
        self.name = xml_element.get('name')
        self.type = xml_element.get('type')
        self.id = int(xml_element.get('id'))

        data = Layer.get_data_from_xml(xml_element)
        self.attrs = {}
        if data is not None:
            for attr in data.items():
                try:
                    val = float(attr[1])
                except ValueError:
                    val = attr[1]
                self.attrs[attr[0].replace('-', '_')] = val

        blobs = Layer.get_element_from_xml(xml_element, 'blobs') if xml_element.find('blobs') else xml_element

        wdata = Layer.get_element_from_xml(blobs, 'weights')
        if wdata is not None:
            self.weights_offset = int(wdata.get('offset'))
            self.weights_size = int(wdata.get('size'))

        bdata = Layer.get_element_from_xml(blobs, 'biases')
        if bdata is not None:
            self.biases_offset = int(bdata.get('offset'))
            self.biases_size = int(bdata.get('size'))

        self.outputs = self.load_outputs_from_xml(xml_element)
        self.inputs = {}

    def load_outputs_from_xml(self, xml_element: Element):
        output = xml_element.find('output')
        if output:
            return self.load_ports_from_xml(output)

    def load_ports_from_xml(self, xml_elements: Element):
        results = {}
        if xml_elements:
            ports = list(xml_elements)
            for ind, port in enumerate(ports):
                if ind == 0:  # ugly hack to get precision from the first output port
                    self.precision = port.get('precision')
                dims = tuple([int(i.text) for i in list(port)])
                results[self.name] = [(int(port.get('id')), dims)]
        return results

    @staticmethod
    def get_data_from_xml(xml_element: Element):
        # get all elements with names '*data'
        data = [el for el in xml_element.findall('*') if el.tag[-4:] == 'data']
        if data:
            # hope that data contains only one element
            return data[0]
        return None

    @staticmethod
    def get_element_from_xml(xml_element: Element, tag: str):
        return xml_element.find(tag)

    # -------------End functions for load layer's params from xml------------

    # ----------------Functions for load weights from ndarray----------------
    def load_bin(self, bins: np.ndarray):
        # The 'precision' attribute is set based on output port precision.
        # The Result op does not have output ports so the 'precision' for this op is not set.
        if self.type == 'Result': return

        method_to_call = 'float32' if self.precision == 'FP32' else 'float16'

        w_end = int((self.weights_offset + self.weights_size) / np.dtype(getattr(np, method_to_call)).itemsize)
        b_end = int((self.biases_offset + self.biases_size) / np.dtype(getattr(np, method_to_call)).itemsize)
        if w_end > len(bins):
            log.error('Can not load weights for layer {}'.format(self.name))
            exit(2)
        if b_end > len(bins):
            log.error('Can not load biases for layer {}'.format(self.name))
            exit(2)
        self.weights = bins[int(self.weights_offset / np.dtype(getattr(np, method_to_call)).itemsize):w_end]
        self.biases = bins[int(self.biases_offset / np.dtype(getattr(np, method_to_call)).itemsize):b_end]

    # --------------End functions for load weights from ndarray--------------

    # -----------------------Functions for comparision-----------------------
    def compare(self, other, break_on_first_diff, ignore_attributes):
        eps = 10e-5 if self.precision == 'FP32' else 4e-2
        return self.params_compare(other, break_on_first_diff, ignore_attributes) and self.bin_compare(other, eps)

    def params_compare(self, other, break_on_first_diff, ignore_attributes):
        status = True
        log.info(' === Comparing {} vs. {}'.format(self.name, other.name))
        if len(self.attrs) != len(other.attrs):
            log.info(' {}: different lengths of attributes'.format(self.type))
            status = False

            value = [k for k in set(other.attrs) - set(self.attrs)]
            if len(value) != 0:
                if sorted(value) == sorted(ignore_attributes.get(self.type, [])):
                    status = True
                else:
                    log.info(" {}: '{}' hasn't attributes: {}".format(self.type, self.name, ', '.join(value)))

            value = [k for k in set(self.attrs) - set(other.attrs)]
            if len(value) != 0:
                if sorted(value) == sorted(ignore_attributes.get(other.type, [])):
                    status = True
                else:
                    log.info(" {}: '{}' hasn't attributes: {}".format(other.type, other.name, ', '.join(value)))

        for attr in self.attrs.keys():
            if self.type in ignore_attributes and attr in ignore_attributes[self.type]:
                continue

            eps = 1.0e-5
            if type(self.attrs.get(attr)) == float:
                status = abs(self.attrs.get(attr, 0) - other.attrs.get(attr, 0)) < eps
            elif type(self.attrs.get(attr)) == str and type(other.attrs.get(attr)) == float:
                status = abs(float(self.attrs.get(attr, 0)) - other.attrs.get(attr, 0)) < eps
            else:
                status = self.attrs.get(attr) == other.attrs.get(attr)
                # Compare dims like '1,3,3136,16' and '[1, 3, 3136, 16]'
                if not status and isinstance(other.attrs.get(attr), str):
                    if re.findall('\d+', str(self.attrs.get(attr))):
                        status = re.findall(
                            '\d+', str(self.attrs.get(attr))) == re.findall(
                            '\d+', other.attrs.get(attr))
            if not status:
                log.info(" {}: '{}' params are different: {} != {} ".format(self.type, attr, self.attrs.get(attr),
                                                                            other.attrs.get(attr)))
                if break_on_first_diff:
                    break

        for item in {k: other.attrs[k] for k in set(other.attrs) - set(self.attrs)}.items():
            log.info(" {}: '{}' param for layer {} not found".format(self.type, item[0], self.get_name()))

        if not self.ports_comparing(self.inputs, other.inputs):
            log.info(' {}: input shapes are different: {} != {} '.format(self.type, self.inputs, other.inputs))
            if not (self.type in ignore_attributes and 'input' in ignore_attributes[self.type]):
                status = False

        if not self.ports_comparing(self.outputs, other.outputs):
            log.info(' {}: out shapes are different: {} != {} '.format(self.type, self.outputs, other.outputs))
            status = False
        return status

    @staticmethod
    def ports_comparing(self_ports: dict, other_ports: dict):
        if len(self_ports) != len(other_ports):
            return False
        s_ports = self_ports.copy()
        o_ports = other_ports.copy()
        for s_port in s_ports.items():
            for s_port_id in s_port[1]:
                for o_port in o_ports.items():
                    for o_port_id in o_port[1]:
                        try:
                            if not np.abs(np.array(s_port_id[1]) - np.array(o_port_id[1])).any():
                                o_ports.pop(o_port[0], None)
                                break
                        except ValueError:
                            pass
                    else:  # continue if there is no breakages
                        continue
                    # the inner loop was breaking so break this too
                    break

        if not len(o_ports):
            return True

    @staticmethod
    def blob_compare(first, second, eps=0.00001, blob_type='weights'):
        status = True
        mess = []
        if len(first) == len(second):
            diff = np.abs(first - second)
            if diff.max() > eps:
                mess.append('Not sorted {} are different'.format(blob_type))
                mess.append('Max diff is {} for index {}'.format(diff.max(), diff.argmax()))
                mess.append('Sum {} diff is {}'.format(blob_type, diff.sum()))
                status = False
            if not status:
                diff = np.abs(np.sort(first) - np.sort(first))
                if diff.max() > eps:
                    mess.append('Sorted {} are different'.format(blob_type))
                    mess.append('Max diff is {} for index {}'.format(diff.max(), diff.argmax()))
                    mess.append('Sum {} diff is {}'.format(blob_type, diff.sum()))
                    status = False
                else:
                    log.info('Sorted {} are equals'.format(blob_type))
        else:
            mess.append('Different length of {}: {} vs {}'.format(blob_type, len(first), len(second)))
            status = False
        return status, mess

    def bin_compare(self, other, eps=10e-5):
        status = True
        if len(self.weights):
            status_weights, messages = Layer.blob_compare(self.weights, other.weights, eps=eps, blob_type='weights')
            if status_weights:
                log.debug('[ OK ] Compare weights: {} vs. {}'.format(self.get_name(), other.get_name()))
            else:
                log.error('[ FAILED ] Compare weights: {} vs. {}'.format(self.get_name(), other.get_name()))
                for message in messages:
                    log.error(message)
            status = status and status_weights
        if len(self.biases):
            status_biases, messages = Layer.blob_compare(self.biases, other.biases, eps=eps, blob_type='biases')
            if status_biases:
                log.debug('[ OK ] Compare biases: {} vs. {}'.format(self.get_name(), other.get_name()))
            else:
                log.warning('[ FAILED ] Compare biases: {} vs. {}'.format(self.get_name(), other.get_name()))
                for message in messages:
                    log.warning(message)
            status = status and status_biases
        return status

    # --------------------End functions for comparision---------------------

    # ----------Functions for dumping to Framework representation-----------
    # ----------------------------------IR----------------------------------
    def to_xml(self, precision, offset):
        layer_element = Element('layer')
        layer_element.set('name', self.name)
        layer_element.set('type', self.type)
        layer_element.set('precision', precision)
        layer_element.set('id', str(self.id))
        self.add_data(layer_element)
        self.add_xml_input_port(layer_element)
        self.add_xml_output_port(layer_element)
        self.add_xml_weights(layer_element, offset)
        return layer_element

    def add_data(self, layer_element):
        if len(self.attrs.keys()):
            data = SubElement(layer_element, 'data')
            for attr in self.attrs.keys():
                data.set(attr, str(self.attrs[attr]))

    def add_xml_output_port(self, layer_element):
        output = SubElement(layer_element, 'output')
        port = SubElement(output, 'port', id=str(self.get_out_port_ids()[0]))
        Layer.add_port_xml(port, self.get_out_shapes()[0])

    def add_xml_input_port(self, layer_element):
        if self.inputs:
            input_port = SubElement(layer_element, 'input')
            for input_layer in self.inputs:
                input_name = input_layer
                port = SubElement(input_port, 'port', id=str(self.inputs[input_name][0][0]))
                Layer.add_port_xml(port, self.inputs[input_name][0][1])

    def add_xml_weights(self, layer_element, offset):
        method_to_call = 'float32' if self.attrs.get('precision', 'FP32') == 'FP32' else 'float16'

        if len(self.weights):
            weights = SubElement(layer_element, 'weights')
            weights.set('offset', str(offset))
            weights.set('size', str(len(self.weights) * np.dtype(getattr(np, method_to_call)).itemsize))
            self.weights_offset = offset
            offset += np.dtype(getattr(np, method_to_call)).itemsize * len(self.weights)
        if len(self.biases):
            weights = SubElement(layer_element, 'biases')
            weights.set('offset', str(offset))
            weights.set('size', str(len(self.biases) * np.dtype(getattr(np, method_to_call)).itemsize))
            self.biases_offset = offset
            offset += np.dtype(getattr(np, method_to_call)).itemsize * len(self.biases)

    def get_bin(self, _bin):
        if self.weights_offset < self.biases_offset:
            _bin[self.weights_offset:self.weights_offset + len(self.weights)] = self.weights
            _bin[self.biases_offset:self.biases_offset + len(self.biases)] = self.biases
        else:
            _bin[self.biases_offset:self.biases_offset + len(self.biases)] = self.biases
            _bin[self.weights_offset:self.weights_offset + len(self.weights)] = self.weights
        return _bin

    def get_size_weights(self):
        return len(self.weights) + len(self.biases)

    @staticmethod
    def add_port_xml(port, shape):
        for s in shape:
            SubElement(port, 'dim').text = str(s)

# --------------------------------End IR--------------------------------
