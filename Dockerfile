# imagem oficial do Ubuntu como base
FROM ubuntu:latest

LABEL maintainer="PUC"

ARG DEBIAN_FRONTEND=noninteractive

# Dependências básicas (inclui curl/unzip para instalar Terraform e AWS CLI)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        unzip \
        wget \
        git \
        sudo \
        openssh-client \
        iputils-ping \
        python3-pip \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Ambiente Python isolado + Databricks CLI
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir databricks-cli

ENV PATH="/opt/venv/bin:${PATH}"

# Terraform
ENV TERRAFORM_VERSION=1.8.2
RUN curl -fsSL "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip" -o /tmp/terraform.zip && \
    unzip -q /tmp/terraform.zip -d /usr/local/bin && \
    rm -f /tmp/terraform.zip

# AWS CLI v2
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip

# Ponto de montagem
RUN mkdir /iac
VOLUME /iac

CMD ["/bin/bash"]
