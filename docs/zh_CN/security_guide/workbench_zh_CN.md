# 深度学习工作台安全性 {#openvino_docs_security_guide_workbench_zh_CN}

深度学习工作台是一个可在 Docker 容器内运行的 Web 应用。

## 运行深度学习工作台

除非必要，否则请将与深度学习工作台的连接限制为 `localhost` (127.0.0.1)，以便仅通过构建 Docker 容器的设备访问该工作台。

使用 `docker run` [开始从 Docker Hub 启动深度学习工作台](@ref workbench_docs_Workbench_DG_Run_Locally)时，限制主机 IP 127.0.0.1 的连接。
例如，使用 `-p 127.0.0.1:5665:5665` 命令将主机 IP 的连接限制为端口 `5665`。如需了解详细信息，请参见[容器网络](https://docs.docker.com/config/containers/container-networking/#published-ports)。

## 身份验证安全性

深度学习工作台使用[身份验证令牌](@ref workbench_docs_Workbench_DG_Authentication)访问应用。每次启动深度学习工作台时，启动深度学习工作台的脚本都会创建一个身份验证令牌。任何拥有身份验证令牌的人都可以使用深度学习工作台。

使用深度学习工作台完成工作后，请注销以防止在未经身份验证的情况下从同一浏览器会话使用深度学习工作台。

要使身份验证令牌完全无效，请[重新启动深度学习工作台](@ref workbench_docs_Workbench_DG_Docker_Container)。

## 使用 TLS 保护通信

[配置传输层安全性 (TLS) ](@ref workbench_docs_Workbench_DG_Configure_TLS)，以保持身份验证令牌加密。
