# OpenVINO Runners used by GitHub Actions CI

The machines that execute workflow commands are referred to as _runners_ in GitHub Actions.

Two types of runners are available in this repository:

* [GitHub Actions Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners) - runners provided and managed by GitHub
* [Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) - runners created and managed by the OpenVINO CI team and linked to the OpenVINO repositories

Generally, it is advised to use the GitHub Actions runners for light jobs, like labelers, code style checks, etc, whereas
longer workflows (such as builds or functional tests) should use the self-hosted runners.

The runners are specified for each job using the `runs-on` key.

An example `Build` job from the [`linux.yml`](./../../../../.github/workflows/linux.yml)
workflow, using the `aks-linux-16-cores-32gb` runner group:

```yaml
Build:
  ...
  runs-on: aks-linux-16-cores-32gb
  ...
```


## Available GitHub Actions Runners

GitHub provides runners with different combinations of available resources and software.
OpenVINO repositories make use of the following runners:

* [default runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources),
  used for not-so-intensive memory and CPU tasks: `ubuntu-22/20.04`, `windows-2019/2022`,
  `macos-12/13`, etc.

* [larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners/about-larger-runners#machine-sizes-for-larger-runners),
  used for memory and CPU-intensive tasks, listed in [the runners page](https://github.com/openvinotoolkit/openvino/actions/runners).


## Available Self-hosted Runners

The self-hosted runners are dynamically spawned for each requested pipeline.
Several configurations are available, which are identified by different group names.
The group names generally follow the pattern:
`aks-{OS}-{CORES_N}-cores-|{RAM_SIZE}gb|-|{ARCH}|`, where:

* `{OS}` - the operating system: `win`/`linux` (currently, only the Windows and Linux
  self-hosted runners are available).
* `{CORES_N}` - the number of cores available to the runners in the group: `4`/`8`/etc.
* `|{RAM_SIZE}gb|` - **_optional_**, the RAM size in GB available to the runners in the group: `8`/`16`/etc.
  * **Note**: The groups with unspecified `{RAM_SIZE}` consist of the runners with 32 GB of RAM
* `|{ARCH}|` - **_optional_**, the architecture of the runners in the group: `arm`
  * **Note**: The groups with unspecified `{ARCH}` consist of the `x86_64` runners

Examples:
* `aks-win-16-cores-32gb` - the Windows x86_64 runners with 16 cores and 32 GB of RAM available
* `aks-linux-16-cores-arm` - the Linux ARM64 runners with 16 cores and 32 GB of RAM available

The available configurations are:

|             | Group Name        | CPU Cores    | RAM in GB        | Architecture         | Examples                                           |
|-------------|-------------------|--------------|------------------|----------------------|----------------------------------------------------|
| Windows     | `aks-win-*`       | `4`/`8`/`16` | `8`/`16`/`32`    | `x86_64`<sup>*</sup> | `aks-win-4-cores-8gb`/`aks-win-16-cores-32gb`      |
| Linux       | `aks-linux-*`     | `4`/`8`/`16` | `16`/`32`        | `x86_64`<sup>*</sup> | `aks-linux-4-cores-16gb`/`aks-linux-16-cores-32gb` |
| Linux ARM64 | `aks-linux-*-arm` | `16`         | `32`<sup>*</sup> | `arm`                | `aks-linux-16-cores-arm`                           |

* `*` - Not specified in the group name

## How to Choose a Runner

The configuration of a runner required for a job (building, testing, other tasks) depends on the
nature of the job. Jobs that are more memory and/or CPU-intensive require a more robust configuration.

The `Build` job in the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow uses
the `aks-linux-16-cores-32gb` group as specified in the `runs-on` key:
```yaml
Build:
  ...
  runs-on: aks-linux-16-cores-32gb
  ...
```

The `aks-linux-16-cores-32gb` group has machines with 16-core CPU and 32 GB of RAM.
These resources are suitable for using in parallel by the build tools in the `Build` job.

The `C++ unit tests` job in the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow uses the `aks-linux-4-cores-16gb` group:
```yaml
CXX_Unit_Tests:
  name: C++ unit tests
  ...
  with:
    runner: aks-linux-4-cores-16gb
    ...
```

As the C++ tests can not use a large number of cores for parallel execution like
the build tools in the `Build` job, it is not beneficial to use the `aks-linux-16-cores-32gb` group for them.

Instead, it is advisable to use runners with more cores/RAM for tasks that can load them.

It is possible to experiment with different configurations before making a decision. You can
run a job on runners from different groups and compare the gains. If they are significant,
for example, 60 minutes on a 4-core runner compared to 15 minutes on a 16-core runner,
it is better to use the configuration with more cores.
