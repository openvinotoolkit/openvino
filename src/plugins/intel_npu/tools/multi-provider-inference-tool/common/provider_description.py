import common.enums
import copy
import hashlib
import json
import os

from pathlib import Path

from common.converters import layout_to_str, shape_to_list

class Config:
    config_description = '''Expects information in JSON format implementing the schema:
"{
    \\"attr_name_0\\": \\"value_0\\",
    \\"attr_name_1\\": \\"value_1\\",
    ...
    \\"attr_name_N\\": \\"value_N\\",
}"
'''
    def __init__(self, cmd_argument: str):
        self.cfg_dict = {}
        if cmd_argument and len(cmd_argument) != 0:
            if not if_file(cmd_argument):
                self.cfg_dict = json.loads(cmd_argument)
            else:
                with open(cmd_argument, "r") as file:
                    isPlainText = False
                    try:
                        self.cfg_dict = json.load(file)
                    except json.JSONDecodeError as ex:
                        # not json file, probably just a plain file
                        isPlainText = True
                        pass

                    if isPlainText:
                        file.seek(0)
                        lines = file.readlines()
                        for line in lines:
                            line = line.strip()
                            if len(line) == 0  or line[0] == '#':
                                continue

                            kv_pairs = line.split()
                            if len(kv_pairs) != 2:
                                raise RuntimeError(f"Cannot parse config file: {cmd_argument}. It must be either JSON or a list of 'KEY\\tVALUE' pairs, encountered the failed line: \"{line}\"")
                            self.cfg_dict[kv_pairs[0]] = kv_pairs[1]


class ModelInfo:
    model_description = '''Expects information in JSON format implementing the schema:
"{
    \\"input_0\\": {
        \\"layout\\":\\"NCHW\\",
        \\"element_type\\":\\"float32\\",
        \\"shape\\": [1,2,3,4]
    },
    \\"input_1\\": {
        \\"shape\\": [2,3,4]
    }
}"
'''
    @staticmethod
    def __validate__(preproc_per_io):
        for i, d in preproc_per_io.items():
            if "shape" in d.keys():
                d["shape"] = shape_to_list(d["shape"])
            if "layout" in d.keys():
                d["layout"] = layout_to_str(d["layout"])

    def __init__(self, command_line_ppm=""):
        self.preproc_per_io = {}
        self.model_name = None
        if command_line_ppm and len(command_line_ppm) != 0:
            if os.path.isfile(command_line_ppm):
                with open(command_line_ppm) as file:
                    self.preproc_per_io = json.load(file)
            else:
                try:
                    self.preproc_per_io = json.loads(command_line_ppm)
                except Exception as ex:
                    raise RuntimeError(
                        ModelInfo.model_description + f"\nGot:\n{command_line_ppm}"
                    )
        ModelInfo.__validate__(self.preproc_per_io)

    def set_model_name(self, model_name : str):
        self.model_name = model_name

    def get_model_name(self):
        return self.model_name

    def insert_info(self, io_name: str, info: {}):
        self.preproc_per_io[io_name] = info
        ModelInfo.__validate__(self.preproc_per_io)

    def update_info(self, io_name: str, additional_info: {}):
        for k,v in additional_info.items():
            self.preproc_per_io[io_name][k] = v
        ModelInfo.__validate__(self.preproc_per_io)

    def get_model_io_names(self):
        return self.preproc_per_io.keys()

    def get_model_io_info(self, io_name: str):
        if io_name not in self.get_model_io_names():
            raise RuntimeError(f"Cannot find input/output: {io_name} for a model among: {self.get_model_io_names()}")
        return self.preproc_per_io[io_name]

class ModelInfoPrinter:
    def __init__(self):
        pass

    def serialize_model_info(self, provider_name : str, model_path : Path, orig_model_info : ModelInfo):
        model_info = copy.deepcopy(orig_model_info)

        base_directory = Path(*provider_name.split("/"))
        base_directory.mkdir(parents=True, exist_ok=True)

        utter_model_info = {}
        for minput_name in model_info.get_model_io_names():
            utter_model_info[minput_name] = model_info.get_model_io_info(minput_name)

        if not model_info.model_name or len(model_info.model_name) == 0:
            model_info.model_name = model_path.stem

        model_info_json_path = base_directory / (model_info.model_name + "_info.json")
        with model_info_json_path.open("w") as outfile:
            json.dump(utter_model_info, outfile)

        # add meta information
        for node_info in utter_model_info.values():
            if "shape" in node_info.keys():
                node_info["shape"] = shape_to_list(node_info["shape"])
        model_meta_info = {}
        model_meta_info["model_path"] = str(model_path)
        model_meta_info["model_info_path"] = str(model_info_json_path)
        sha256 = hashlib.sha256()
        sha256.update(str(Path(__file__).absolute()).encode('utf-8'))
        utter_model_info["_meta_" + sha256.hexdigest()] = model_meta_info
        return json.dumps(utter_model_info,indent=4)


class TensorInfo:
    necessary_attrs = {"data", "bytes_size", "model"}
    ext_attrs = ["element_type", "shape"]

    types={"input", "output"}
    def __init__(self):
        self.info = {}

    def set_type(self, ttype:str):
        if ttype not in TensorInfo.types:
            raise RuntimeError(f"Cannot specify type: {ttype} for TensorInfo, available types: {TensorInfo.types}")
        self.info["type"] = ttype

    def get_type(self) -> str:
        return self.info["type"]

    def validate(self):
        if not TensorInfo.necessary_attrs.issubset(self.info.keys()):
            raise RuntimeError(f"Fields: {TensorInfo.necessary_attrs} are required in {self.info}")

        if "shape" in self.info.keys():
            self.info["shape"] = shape_to_list(self.info["shape"])
        if "layout" in self.info.keys():
            self.info["layout"] = layout_to_str(self.info["layout"])

class TensorsInfoPrinter:
    canonization_table = {
        "<": "%%_lt_%%",
        ">": "%%_gt_%%",
        ":": "%%_colon_%%",
        "\"": "%%_dquote_%%",
        "/": "%%_fslash_%%",
        "\\": "%%_bslash_%%",
        "|": "%%_pipe_%%",
        "?": "%%_quest_%%",
        "*": "%%_asterix_%%"
    }
    decanonization_table = {v: k for k, v in canonization_table.items()}

    def __init__(self):
        pass

    @staticmethod
    def canonize_io_name(io_name):
        return "".join([TensorsInfoPrinter.canonization_table.get(c, c) for c in io_name])

    @staticmethod
    def decanonize_io_name(io_name):
        return "".join([TensorsInfoPrinter.decanonization_table.get(c, c) for c in io_name])

    @staticmethod
    def get_file_name_to_dump_model_source(source: str):
        return source + "s_dump_data.json"

    @staticmethod
    def canonize_file_name(file_name : str):
        file_name = "".join(str(file_name).split())   # remove spaces
        file_name = "_".join(str(file_name).split(","))   # remove spaces
        return file_name

    @staticmethod
    def get_printable_tensor_name(info):
        file_name = ""
        if info["type"] == "input":
            file_name = "idata_"
            for attr in [attr for attr in info.keys() if attr in TensorInfo.ext_attrs]:
                file_name += TensorsInfoPrinter.canonize_file_name(info[attr])
                file_name += "_"
        elif info["type"] == "output":
            file_name = "odata_"
            for attr in [attr for attr in info.keys() if attr in TensorInfo.ext_attrs]:
                file_name += TensorsInfoPrinter.canonize_file_name(info[attr])
                file_name += "_"
        if len(file_name) == 0:
            raise RuntimeError(f"Cannot compose tensor name from the info: {info}")
        file_name = file_name[0:-1] + ".blob"
        return file_name

    def get_printable_input_tensor_info(self, input_tensors_dict:list):
        printable_tensor_info = {}
        for info in input_tensors_dict:
            if info["type"] != "input":
                continue
            input_files = info['input_files']['files']
            tensor_source = {}
            tensor_source["files"] = info['input_files']['files']
            tensor_source["type"] = info['input_files']['type']
            if tensor_source["type"] == common.enums.InputSourceFileType.image.name:
                tensor_source["convert"] = {}
                tensor_source["convert"]["shape"] = info['shape']
                tensor_source["convert"]["layout"] = layout_to_str(info['layout'])
                tensor_source["convert"]["element_type"] = info['element_type']
            else:
                tensor_source["shape"] = info['shape']
                tensor_source["layout"] = layout_to_str(info['layout'])
                tensor_source["element_type"] = info['element_type']

            printable_tensor_info[info['source']] = tensor_source
        return printable_tensor_info

    def get_printable_output_tensor_info(self, input_tensors_dict:list):
        printable_tensor_info = {}
        for info in input_tensors_dict:
            if info["type"] != "output":
                continue

            tensor_source = {}
            if "input_files" in info.keys():
                tensor_source["files"] = info['input_files']['files']
                tensor_source["type"] = info['input_files']['type']

            if "layout" in info.keys():
                tensor_source["layout"] = layout_to_str(info['layout'])

            tensor_source["shape"] = info['shape']
            tensor_source["element_type"] = info['element_type']
            tensor_source["type"] = 'bin'
            printable_tensor_info[info['source']] = tensor_source
        return printable_tensor_info

    def get_printable_tensor_info(self, input_tensors_dict:list, ttype):
        return  self.get_printable_input_tensor_info(input_tensors_dict) if  ttype == "input" else self.get_printable_output_tensor_info(input_tensors_dict)


    def serialize_tensors_by_type(self, root_path : Path, input_tensors_dict : list, ttype):
        aggregated_input_meta = self.get_printable_tensor_info(input_tensors_dict, ttype)
        aggregated_tensor_meta = copy.deepcopy(aggregated_input_meta)

        serialized_blob_paths = []
        model_serialization_path = Path()
        try:
            # ensure the root directory exists
            root_path.mkdir(parents=True, exist_ok=True)
            for info in input_tensors_dict:
                if ttype and len(ttype) !=0 and info["type"] != ttype:
                    continue

                # create a model directory inside the root directory.
                # A nodel directory is a storage for model input and output directories
                # and meta information as well
                model_serialization_path = root_path / info["model"]

                # create model input/output directories which encompass meta information and blobs
                main_model_source_dir = model_serialization_path / info["type"]
                main_model_source_dir.mkdir(parents=True, exist_ok=True)

                # well known filesystems forbid special symbols in string paths,
                # so that let's apply path canonization to model input/ouput names
                canonized_fs_input_name = TensorsInfoPrinter.canonize_io_name(info["source"])

                # dump input meta in JSON
                if ttype == "input":
                    if aggregated_input_meta[info["source"]]["type"] != common.enums.InputSourceFileType.bin.name:
                        model_input_meta_info_file_path = main_model_source_dir / (canonized_fs_input_name + "_img.json")
                        with model_input_meta_info_file_path.open("w") as outfile:
                            json.dump({info["source"] : aggregated_input_meta[info["source"]]}, outfile)

                # create input/ouput directory which stores blobs only
                blob_storage_dir = main_model_source_dir / canonized_fs_input_name
                blob_storage_dir.mkdir(parents=True, exist_ok=True)

                blob_file_path = blob_storage_dir / TensorsInfoPrinter.get_printable_tensor_name(info)
                with blob_file_path.open("wb") as input_tensor_file:
                    input_tensor_file.write(info["data"])
                serialized_blob_paths.append(blob_file_path)

                # tensor meta is input meta having all conversions applied which fit a model
                aggregated_tensor_meta[info["source"]]["files"] = [str(blob_file_path)]   # print as list
                aggregated_tensor_meta[info["source"]]["type"] = common.enums.InputSourceFileType.bin.name
                if "convert" in aggregated_tensor_meta[info["source"]].keys():
                    aggregated_tensor_meta[info["source"]].update(aggregated_tensor_meta[info["source"]]["convert"])
                    del aggregated_tensor_meta[info["source"]]["convert"]

                # dump tensor meta in JSON
                model_tensor_meta_info_file_path = main_model_source_dir / (canonized_fs_input_name + "_dump.json")
                with model_tensor_meta_info_file_path.open("w") as outfile:
                    json.dump({info["source"] : aggregated_tensor_meta[info["source"]]}, outfile)
        except Exception as ex:
            raise RuntimeError(f"Cannot serialize tensor of type: {ttype} into a file, error: {ex}")

        # store aggregated inputs JSON info as "--input" param compatible format
        if model_serialization_path != Path():
            input_info_file_path = model_serialization_path / (ttype + "s_img.json")
            with input_info_file_path.open("w") as outfile:
                json.dump(aggregated_input_meta, outfile)

            # store aggregated tensors JSON info as "--input" param compatible format
            input_info_dumps_file_path = model_serialization_path / TensorsInfoPrinter.get_file_name_to_dump_model_source(ttype)
            with open(input_info_dumps_file_path, "w") as outfile:
                json.dump(aggregated_tensor_meta, outfile)

        return serialized_blob_paths, input_info_file_path, input_info_dumps_file_path

    def deserialize_output_tensor_descriptions(self, root_path : Path, model_name : str):
        ttype = "output"
        if ttype not in TensorInfo.types:
            raise RuntimeError(f"Incorrect tensor type to deserialize: {ttype}. Expected: {TensorInfo.types}")

        if not root_path.is_dir():
            raise RuntimeError(f"Cannot deserialize tensors as the provider directory doesn't exist: {root_path}")

        model_serialization_path = root_path / model_name
        if not model_serialization_path.is_dir():
            raise RuntimeError(f"Cannot deserialize tensors as the model info directory doesn't exist: {model_serialization_path}")

        model_sources_info_file_path = model_serialization_path / TensorsInfoPrinter.get_file_name_to_dump_model_source(ttype)
        if not model_sources_info_file_path.is_file():
            raise RuntimeError(f"Cannot deserialize tensors as the model info file doesn't exist: {model_sources_info_file_path}")

        model_sources_info = {}
        try:
            with model_sources_info_file_path.open("r") as file:
                model_sources_info = json.load(file)
        except json.JSONDecodeError as ex:
            raise RuntimeError(f"The file: {model_sources_info_file_path} contains no JSON data. Error: {ex}") from None

        for io, data in model_sources_info.items():
            necessary_fields = {"files", "element_type", "shape"}
            if not necessary_fields.issubset(data.keys()):
                raise RuntimeError(f"Cannot deserialize tensor info from file: {model_sources_info_file_path}, as one from required fields are missing: {necessary_fields}")
            if "layout" in data.keys():
                data["layout"] = layout_to_str(data['layout'])
        return model_sources_info
