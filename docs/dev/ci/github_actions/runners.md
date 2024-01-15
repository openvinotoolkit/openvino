# Overview of the Runners used in the OpenVINO GitHub Actions CI

The machines that execute the commands from the workflows are referred to as _runners_ in GitHub Actions.

Two types of runners are available in this repository:
   
* [GitHub Actions Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners) - runners provided and managed by GitHub
* [Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) - runners created and managed by the OpenVINO CI team and linked to the OpenVINO repositories 

The runners are specified for each job using the `runs-on` key. 

An example `Build` job from the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow:
```yaml
Build:
  ...
  runs-on: aks-linux-16-cores-32gb
  ...
```

The `aks-linux-16-cores-32gb` runners group is used for this job.

## Available GitHub Actions Runners

GitHub provides runners with different combinations of available resources and software. 

The OpenVINO repositories make use of the following runners:

* [The default runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources): `ubuntu-22/20.04`, `windows-2019/2022`, `macos-12/13`, etc.
  * Used for not-so-intensive memory and CPU tasks
* [The larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners/about-larger-runners#machine-sizes-for-larger-runners): you can find the list of the available larger runners [here](https://github.com/openvinotoolkit/openvino/actions/runners)
  * Used for memory and CPU heavy tasks

## Available Self-hosted Runners

The self-hosted runners are dynamically spawned for each requested pipeline. 
Several configurations of the self-hosted runners are available, they are identified by different group names.

The group names generally follow the pattern: `aks-{OS}-{CORES_N}-cores-|{RAM_SIZE}gb|-|{ARCH}|`, where:
* `{OS}` - the operating system: `win`/`linux`
  * **Note**: Currently, only Windows and Linux self-hosted runners are available.
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

## How to choose a Runner

The configuration of a runner required for a job (building, testing, etc.) stems from the nature of the job: the more memory and/or CPU-intensive it is, 
the more robust configuration is required.

The `Build` job in the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow uses the `aks-linux-16-cores-32gb` group as specified in the `runs-on` key:
```yaml
Build:
  ...
  runs-on: aks-linux-16-cores-32gb
  ...
```

This group has machines with 16 core CPU and 32 GB of RAM, which could be utilized in parallel by the build tools used in the `Build` job. 

The `C++ unit tests` job in the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow uses the `aks-linux-4-cores-16gb` group:
```yaml
CXX_Unit_Tests:
  name: C++ unit tests
  ...
  with:
    runner: aks-linux-4-cores-16gb
    ...
```

As the C++ tests could not utilize the large number of cores for parallel execution as the build tools in the `Build` job could, 
it would be pointless to use the `aks-linux-16-cores-32gb` group for them.

The advice is to use runners with more cores/RAM size for the tasks that could load them.

It is possible to experiment with different configurations before deciding, i.e.,
run a job on runners from different groups and observe the gains; if they are significant, e.g., 60 minutes on a 4-core runner vs. 15 minutes on a 16-core runner, 
it is better to use those with more cores.
