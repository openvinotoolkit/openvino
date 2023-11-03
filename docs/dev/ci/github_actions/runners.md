# Overview of the Runners used in the OpenVINO GitHub Actions CI

The machines that execute the commands from the workflows are referred to as _runners_ in GitHub Actions.

There are two types of runners available in this repository:

* [GitHub Actions Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners) - runners provided and managed by GitHub
* [Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) - runners created and managed by the OpenVINO CI team and linked to the OpenVINO repositories 

## Available GitHub Actions Runners

GitHub provides runners with different combinations of available resources and software. 

The OpenVINO repositories make use of the following runners:

* [The default runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources): `ubuntu-22/20.04`, `windows-2019/2022`, `macos-12/13`
  * Used for not-so-intensive memory and CPU tasks
* [The larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners/about-larger-runners#machine-sizes-for-larger-runners): you can find the list of available larger runners [here](https://github.com/openvinotoolkit/openvino/actions/runners)
  * Used for memory and CPU heavy tasks

## Available Self-hosted Runners

## How to choose a Runner
