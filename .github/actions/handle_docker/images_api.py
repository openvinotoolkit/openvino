# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import subprocess
from pathlib import Path
from typing import Iterable

from helpers import run, name_from_dockerfile


class Image:
    def __init__(self, name: str, dockerfile: Path, registry: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = name
        self.dockerfile = dockerfile
        self.registry = registry
        self.tag = 'latest'
        self.base_tag = None

    def __str__(self):
        return self.name

    def __eq__(self, img):
        return img.name == self.name if img else False

    def with_tag(self, tag: str):
        self.tag = tag
        return self

    def with_base_tag(self, tag: str):
        self.base_tag = tag
        return self

    def ref(self):
        return f"{self.registry}/{self.name}:{self.tag}"

    def base_ref(self):
        if not self.base_tag:
            return None
        return f"{self.registry}/{self.name}:{self.base_tag}"

    def push(self, dry: bool = False):
        cmd = f"docker push {self.ref()} "
        run(cmd, dry)

    def build(self, dry: bool = False, push: bool = True, docker_builder: str = None, import_cache: bool = True,
              export_cache: bool = True):
        cache_cmd = ""
        if import_cache:
            cache_cmd += f"--cache-from type=registry,ref={self.ref()}-cache "
            if self.base_tag:
                cache_cmd += f"--cache-from type=registry,ref={self.base_ref()}-cache "

        if export_cache:
            cache_cmd += f"--cache-to type=registry,ref={self.ref()}-cache,mode=max "

        build_cmd = f"docker buildx build --builder={docker_builder}" if docker_builder else "docker build"
        push_cmd = f"--push" if push else ""

        cmd = f"{build_cmd} " \
              f"--file {self.dockerfile} " \
              f"--tag {self.ref()} " \
              f"--build-arg REGISTRY={self.registry}/dockerio " \
              f"{cache_cmd} " \
              f"{push_cmd} " \
              "."

        run(cmd, dry)

    def tag_base(self, dry: bool = False):
        if not self.base_tag:
            raise AttributeError("Tag for base image is not specified")

        cmd = f"docker buildx imagetools create -t {self.ref()} {self.base_ref()}"

        run(cmd, dry)

    def is_missing(self, dry: bool = False, base: bool = False) -> bool:
        image = self.base_ref() if base else self.ref()
        if base and not image:
            self.logger.warning(f"Base ref for image {self.ref()} is missing")
            return True

        cmd = f"docker manifest inspect {image}"
        is_missing = False

        self.logger.info(cmd)
        if not dry:
            try:
                subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, text=True)
            except subprocess.CalledProcessError:
                self.logger.warning(f"{image} is missing in registry")
                is_missing = True

        return is_missing


# Making it a class, so it's a little easier to switch to a tree structure for building inherited images if we want
class ImagesHandler:
    def __init__(self, dry_run: bool = False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.images = dict()
        self.dry_run = dry_run

    def add_from_dockerfile(self, dockerfile: str | Path, dockerfiles_root: str | Path, registry: str, tag: str,
                            base_tag: str = None):
        image_name = name_from_dockerfile(dockerfile, dockerfiles_root)
        image = Image(image_name, Path(dockerfile), registry).with_tag(tag).with_base_tag(base_tag)
        self.add(image)

    def add(self, image: Image):
        self.images[image.name] = image

    def get(self, image_names: Iterable = None) -> list:
        images = [self.images[name] for name in image_names] if image_names is not None else self.images.values()
        return images

    def get_missing(self, image_names: Iterable = None, base: bool = False) -> list:
        missing_images = [image.name for image in self.get(image_names) if image.is_missing(self.dry_run, base)]
        return missing_images

    def build(self, image_names: Iterable = None, missing_only: bool = False, push: bool = True, builder: str = None,
              import_cache: bool = True, export_cache: bool = True):
        to_build = self.get(self.get_missing(image_names)) if missing_only else self.get(image_names)
        for image in to_build:
            image.build(self.dry_run, push, builder, import_cache, export_cache)
        return to_build

    def push(self, image_names: Iterable = None, missing_only: bool = False):
        to_push = self.get(self.get_missing(image_names)) if missing_only else self.get(image_names)
        for image in to_push:
            image.push(self.dry_run)

    def tag(self, image_names: Iterable = None):
        for image in self.get(image_names):
            image.tag_base(self.dry_run)
