# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
import sys

from distutils.util import strtobool
from helpers import *
from images_api import *


def parse_args():
    parser = argparse.ArgumentParser(description='Returns list of Docker images to build for a given workflow')
    parser.add_argument('-i', '--images', required=True, help='Comma-separated docker images')
    parser.add_argument('-d', '--dockerfiles_root', required=True, help='Path to dockerfiles')
    parser.add_argument('-r', '--registry', required=True, help='Docker registry name')
    parser.add_argument('-s', '--commit', required=False, help='Commit SHA. If not set, --pr is used')
    parser.add_argument('-b', '--docker_builder', required=False, help='Docker buildx builder name')
    parser.add_argument('--pr', type=int, required=False, help='PR number, if event is pull_request')
    parser.add_argument('--head_tag_file', default='.github/dockerfiles/docker_tag', help='Head docker tag file path')
    parser.add_argument('--base_tag_file', default=None, required=False, help='Base docker tag file path')
    parser.add_argument('--ref_name', required=False, default='', help='GitHub ref name')
    parser.add_argument('--repo', default='openvinotoolkit/openvino', help='GitHub repository')
    parser.add_argument('--docker_env_changed', type=lambda x: bool(strtobool(x)), default=True,
                        help='Whether PR changes docker env')
    parser.add_argument('--dockerfiles_changed', type=lambda x: bool(strtobool(x)), default=True,
                        help='Whether PR changes dockerfiles')
    parser.add_argument('--action_path', default='.github/actions/handle_docker', help='Path to this GitHub action')
    parser.add_argument('--push', action='store_true', required=False, help='Whether to push images to registry')
    parser.add_argument('--dry_run', action='store_true', required=False, help='Dry run')
    args = parser.parse_args()
    return args


def main():
    init_logger()
    logger = logging.getLogger(__name__)
    args = parse_args()
    for arg, value in sorted(vars(args).items()):
        logger.info(f"Argument {arg}: {value}")

    head_tag = Path(args.head_tag_file).read_text().strip()

    base_tag_exists = args.base_tag_file and Path(args.base_tag_file).exists()
    base_tag = Path(args.base_tag_file).read_text().strip() if base_tag_exists else None

    all_dockerfiles = Path(args.dockerfiles_root).rglob('**/*/Dockerfile')

    images = ImagesHandler(args.dry_run)
    for image in all_dockerfiles:
        images.add_from_dockerfile(image, args.dockerfiles_root, args.registry, head_tag, base_tag)

    requested_images = set(args.images.split(','))
    skip_workflow = False
    missing_only = False

    merge_queue_target_branch = next(iter(re.findall(f'^gh-readonly-queue/(.*)/', args.ref_name)), None)

    if args.pr:
        environment_affected = args.docker_env_changed or args.dockerfiles_changed
        if environment_affected:
            expected_tag = f'pr-{args.pr}'

            if head_tag != expected_tag:
                logger.error(f"Some of your changes affected Docker environment for CI. "
                             f"Please update docker tag in {args.head_tag_file} to {expected_tag}. "
                             f"For more details please see "
                             f"https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/ci/github_actions/docker_images.md")
                sys.exit(1)

    elif merge_queue_target_branch:
        environment_affected = head_tag != base_tag
        if environment_affected:
            logger.info(f"Environment is affected by PR(s) in merge group")
    else:
        environment_affected = False

    if environment_affected:
        changeset = get_changeset(args.repo, args.pr, merge_queue_target_branch, args.commit)
        changed_dockerfiles = [p for p in changeset if p.startswith(args.dockerfiles_root) and p.endswith('Dockerfile')]

        if args.docker_env_changed:
            logger.info(f"Common docker environment is modified, will build all requested images")
            changed_images = requested_images
        else:
            logger.info(f"Common docker environment is not modified, will build only changed and missing images")
            changed_images = set([name_from_dockerfile(d, args.dockerfiles_root) for d in changed_dockerfiles])

        unchanged_images = requested_images - changed_images
        unchanged_with_no_base = images.get_missing(unchanged_images, base=True)

        if unchanged_with_no_base:
            logger.info("The following images were unchanged, but will be built anyway since the base for them "
                        f"is missing in registry: {unchanged_with_no_base}")

        images_to_tag = unchanged_images.difference(unchanged_with_no_base)
        images_to_build = requested_images.intersection(changed_images).union(unchanged_with_no_base)

        only_dockerfiles_changed = len(changeset) == len(changed_dockerfiles)
        if only_dockerfiles_changed and not images_to_build:
            skip_workflow = True
    else:
        logger.info(f"Environment is not affected, will build only missing images, if any")
        images_to_build = requested_images
        images_to_tag = []
        missing_only = True

    if not images_to_build:
        logger.info(f"No images to build, will return the list of pre-built images with a new tag")

    built_images = images.build(images_to_build, missing_only, args.push, args.docker_builder)
    if not built_images:
        logger.info(f"No images were built, a new tag will be applied to a pre-built base image if needed")

    # When a custom builder is used, it allows to push the image automatically once built. Otherwise, pushing manually
    if args.push and not args.docker_builder:
        images.push(images_to_build, missing_only)

    if environment_affected and base_tag:
        images.tag(images_to_tag)

    images_output = images_to_output(images.get(requested_images))
    set_github_output("images", json.dumps(images_output))

    if skip_workflow:
        logger.info(f"Docker image changes are irrelevant for current workflow, workflow may be skipped")
        set_github_output("skip_workflow", str(skip_workflow))


main()
