terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.32.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

provider "databricks" {
  profile = "DEFAULT"  # já existe para workspace-level
}

