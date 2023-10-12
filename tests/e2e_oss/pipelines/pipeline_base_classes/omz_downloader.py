import logging as log
import os
import sys

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from utils.downloader_utils import get_models
from utils.e2e.env_tools import Environment
from utils.e2e.ir_provider.omz_model_downloader import OMZModelDownloader
from utils.path_utils import import_from

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


class DownloaderBase(CommonConfig):
    models_pool = get_models()

    def prepare_prerequisites(self, *args, **kwargs):
        # Prepare model info only for test classes
        if getattr(self, '__is_test_config__', False):
            if not hasattr(self, "downloader_tag"):
                raise AttributeError("Test class {} doesn't have required attribute 'downloader_tag'".format(
                    self.__class__.__name__))
            self.model_info = self.find_in_models_pool(self.downloader_tag)
            omz_runner = OMZModelDownloader({'model_info': self.model_info, 'precision': self.precision})
            omz_runner._download()

            if self.model_info.framework != 'dldt':
                for key in self.model_info.__dict__.keys():
                    if "_to_onnx" in key and getattr(self.model_info, key) is not None:
                        from utils.downloader_utils import convert
                        out, err, retcode = convert(name=self.downloader_tag, precision=self.precision,
                                                    extra_mo_args=getattr(self, 'add_mo_args', {}))
                        if retcode != 0:
                            raise RuntimeError("Model conversion failed with error {}".format(err))
                        log.info("OMZ Model Converter stdout:\n{}".format(out))
                        break
                self.model_info.mo_args = self.fix_paths_in_mo_cmd(model_base_dir=str(omz_runner.model_base_dir))
                self.mo_argv = self.extract_model_info()

                if not hasattr(self, 'no_auto_preproc'):
                    if not (hasattr(self, 'h') and hasattr(self, 'w')):
                        self.h, self.w = self.get_h_w()
                    input_names = self.mo_argv.input.split(',')
                    input_name = input_names[0]
                    if len(self.mo_argv.mean_scale_values.keys()) != 0 and len(input_names):
                        mean = self.mo_argv.mean_scale_values[input_name]['mean']
                        self.mean = list(mean) if mean is not None else None
                        scale = self.mo_argv.mean_scale_values[input_name]['scale']
                        self.scale = list(scale) if scale is not None else None
                    else:
                        log.warning("Can't get preprocessing info for multi-input network or it's not defined in "
                                    "Model Optimizer command line. Please specify it manually if required.")
                        self.mean, self.scale = None, None

                self.mapping_file_location = str(omz_runner.ir_base_dir / self.model_info.name) + '.mapping'
            else:
                self.irs_map = {}
                for precision in self.model_info.precisions:
                    self.irs_map[precision] = {}

                for file in self.model_info.files:
                    ext = file.name.suffix.replace('.', '')
                    self.irs_map[str(file.name.parent)].update({ext: str(omz_runner.model_base_dir / file.name)})

    def find_in_models_pool(self, filter, check_unique=True):
        filtered = [model for model in self.models_pool if model.name == filter]
        if check_unique:
            assert not len(filtered) > 1, "More than one model with specified model tag {} found!".format(filter)
            assert not len(filtered) == 0, "No models with specified model tag {} found!".format(filter)
            return filtered[0]
        else:
            return filtered

    def fix_paths_in_mo_cmd(self, model_base_dir):
        fixed_args = []
        subdir = str(self.model_info.subdirectory)
        for arg in self.model_info.mo_args:
            # TODO (vurusovs): replace `Environment.env` use with `instance.environment`
            if '$dl_dir' in arg:
                fixed_args.append(arg.replace("$dl_dir", model_base_dir))
            elif '$conv_dir' in arg:
                fixed_args.append(arg.replace("$conv_dir", str(
                    (os.path.join(os.path.abspath(Environment.env.get('mo_out')), subdir)))))
            elif '$mo_dir' in arg:
                fixed_args.append(arg.replace("$mo_dir", os.path.dirname(Environment.env.get('mo_runner'))))
            elif '$config_dir' in arg:
                fixed_args.append(arg.replace("$config_dir", os.path.join(Environment.env.get('omz_root'), 'models',
                                                                          subdir)))
            else:
                fixed_args.append(arg)
        return fixed_args

    def extract_model_info(self):
        # TODO (vurusovs): replace `Environment.env` use with `instance.environment`
        mo_root = os.path.dirname(Environment.env.get('mo_runner'))

        with import_from(os.path.join(mo_root)):
            from openvino.tools.mo.utils.cli_parser import get_all_cli_parser
            from openvino.tools.mo.utils.cli_parser import get_placeholder_shapes
            from openvino.tools.mo.utils.cli_parser import parse_tuple_pairs
            from openvino.tools.mo.utils.cli_parser import get_mean_scale_dictionary

            parser = get_all_cli_parser()
            argv = parser.parse_args(self.model_info.mo_args)
            argv.output = argv.output.split(',') if argv.output else None
            argv.placeholder_shapes = get_placeholder_shapes(argv.input, argv.input_shape, argv.batch)
            mean_values = parse_tuple_pairs(argv.mean_values)
            scale_values = parse_tuple_pairs(argv.scale_values)
            argv.mean_scale_values = get_mean_scale_dictionary(mean_values, scale_values, argv.input)
        return argv

    def get_h_w(self):
        layout = getattr(self, 'input_data_layout', 'nchw')
        assert self.mo_argv.placeholder_shapes is not None, "No info about input shape in Model Optimizer " \
                                                            "command line. " \
                                                            "Please specify height and width manually if required."
        assert len(self.mo_argv.placeholder_shapes[0].keys()) == 1, "Can't get height and width for " \
                                                                    "multi-input network." \
                                                                    "Please specify it manually if required."
        shape = self.mo_argv.placeholder_shapes[0][self.mo_argv.input]
        if layout.lower() == 'nchw':
            return tuple(shape[2:])
        elif layout.lower() == 'nhwc':
            return tuple(shape[1:3])
        else:
            return None
