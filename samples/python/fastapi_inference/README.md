# OpenVINO FastAPI Inference Sample

This sample demonstrates how to deploy an **OpenVINO™ IR model** as a lightweight
**REST inference service** using **FastAPI** and **Uvicorn**.

It is intended as a **reference example** for building simple HTTP-based inference
services on top of the OpenVINO Runtime.

---

## Features

- Uses OpenVINO Runtime Python API
- REST API built with FastAPI
- Automatic OpenAPI / Swagger UI
- Model loaded once at application startup
- CPU device by default (configurable via environment variables)

---

## Directory Structure

```text
fastapi_inference/
├── app.py            # FastAPI application
├── model_utils.py    # OpenVINO model wrapper
├── schemas.py        # Request/response schemas
├── models/           # OpenVINO IR models
│   └── public/
│       └── mobilenet-v2-pytorch/
│           └── FP16/
│               ├── mobilenet-v2-pytorch.xml
│               └── mobilenet-v2-pytorch.bin
└── README.md