import logging as log
import os
import re
import sys
from pathlib import Path

from .provider import ClassProvider
from utils.e2e.env_tools import Environment
from utils.downloader_utils import download, convert

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


class OMZModelDownloader(ClassProvider):
    __action_name__ = "omz_model_downloader"

    def __init__(self, config):
        self.model_info = config['model_info']
        self.precision = config['precision']
        self.additional_mo_args = config.get("additional_args", {})
        # TODO (vurusovs): replace `Environment.env` use with `instance.environment`
        self.model_base_dir = Path(Environment.env.get('omz_models_out', os.getcwd())) / self.model_info.subdirectory
        self.ir_base_dir = Path(
            Environment.env.get('mo_out', os.getcwd())) / self.model_info.subdirectory / self.precision
        self.xml = None
        self.bin = None
        self.mo_log = None

    def get_ir(self, data=None):
        out, err, retcode = convert(name=self.model_info.name, precision=self.precision,
                                    extra_mo_args=self.additional_mo_args)
        if retcode != 0:
            match = re.search(r"\[ ERROR \]\s*The\s(\".*\")\sis not existing file", err)
            if match is not None:
                expected_path = err[slice(*match.regs[1])]
                log.error("Conversion failed! Original model file wasn't found at {}\n"
                          "Will try to download the model first and retry conversion".format(expected_path))
                self._download()
                out, err, retcode = convert(name=self.model_info.name, precision=self.precision,
                                            extra_mo_args=self.additional_mo_args)
                if retcode != 0:
                    raise RuntimeError("Model conversion failed with error {}".format(err))
            else:
                raise RuntimeError("Model conversion failed with error {}".format(err))
        log.info("OMZ Model Converter stdout:\n{}".format(out))
        self.xml = str(self.ir_base_dir / self.model_info.name) + '.xml'
        self.bin = str(self.ir_base_dir / self.model_info.name) + '.bin'

    def _download(self):
        download(name=self.model_info.name, model_base_dir=str(self.model_base_dir), precision=self.precision)
        return self._get_model_files()

    def _get_model_files(self):
        model_files = []
        for file in self.model_info.files:
            model_files.append(str(self.model_base_dir / file.name))
        return model_files
