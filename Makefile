# Makefile for running the vae package remotely via the Colab CLI.
# Requires: colab CLI (https://github.com/googlecolab/colabtools)
#
# Usage examples:
#   make colab-new            # provision a CPU session
#   make colab-new GPU=A100   # provision a GPU session
#   make colab-all            # upload, install deps, and run
#   make colab-stop           # tear down the session

SESSION_NAME ?= jax-vae
GPU          ?=
TPU          ?=
LOCAL_SRC    := src/vae
REMOTE_DIR   := /content/vae

.PHONY: help \
        colab-new colab-new-gpu colab-status \
        colab-upload colab-install \
        colab-run colab-run-exec colab-download-images \
        colab-logs colab-stop colab-clean colab-all

help:
	@echo "Colab remote execution targets:"
	@echo "  colab-new          Create a new Colab session (CPU by default)"
	@echo "  colab-new-gpu      Create a new Colab session with --gpu T4"
	@echo "  colab-status       Show session status"
	@echo "  colab-upload       Upload src/vae/ to $(REMOTE_DIR) on the VM"
	@echo "  colab-install      Install jax and matplotlib on the VM"
	@echo "  colab-run          Run via piped colab console (python -m vae)"
	@echo "  colab-run-exec     Run via colab exec with 1-hour timeout (recommended)"
	@echo "  colab-download-images  Pull generated /tmp/mnist_vae_*.png to output/"
	@echo "  colab-logs         Show last 20 log entries"
	@echo "  colab-stop         Stop the session (saves compute units)"
	@echo "  colab-clean        Stop session and delete local output/*.png"
	@echo "  colab-all          Upload + install + run (uses colab-run-exec)"

## Provisioning

colab-new:
ifeq ($(GPU),)
ifeq ($(TPU),)
	colab new -s $(SESSION_NAME)
else
	colab new -s $(SESSION_NAME) --tpu $(TPU)
endif
else
	colab new -s $(SESSION_NAME) --gpu $(GPU)
endif
	colab status -s $(SESSION_NAME)

colab-new-gpu:
	$(MAKE) colab-new GPU=T4

colab-status:
	colab status -s $(SESSION_NAME)

## Sync & dependencies

colab-mkdir:
	echo "mkdir -p $(REMOTE_DIR)" | colab console -s $(SESSION_NAME)

colab-upload: colab-mkdir
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/__init__.py $(REMOTE_DIR)/__init__.py
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/datasets.py   $(REMOTE_DIR)/datasets.py
	colab upload -s $(SESSION_NAME) $(LOCAL_SRC)/__main__.py  $(REMOTE_DIR)/__main__.py

colab-install:
	colab install -s $(SESSION_NAME) jax matplotlib

## Execution

# Option 1: colab console (simple, but no explicit timeout control)
colab-run: colab-upload
	echo "cd $(REMOTE_DIR)/.. && python -m vae" | colab console -s $(SESSION_NAME)

# Option 2: colab exec (preferred by Colab CLI docs; supports --timeout)
# The wrapper script runs `python -m vae` on the remote side so relative
# imports in __main__.py work correctly.
colab-run-exec: colab-upload
	colab exec -s $(SESSION_NAME) --timeout 3600 -f scripts/colab_run.py

## Artifacts

colab-download-images:
	mkdir -p output
	@echo "Downloading images from /tmp on the remote VM..."
	@for i in 000 001 002 003 004 005 006 007 008 009; do \
		colab download -s $(SESSION_NAME) /tmp/mnist_vae_$$i.png output/mnist_vae_$$i.png 2>/dev/null || true; \
	done
	@echo "Images saved to output/"

colab-logs:
	colab log -s $(SESSION_NAME) -n 20

## Teardown

colab-stop:
	colab stop -s $(SESSION_NAME)

colab-clean: colab-stop
	rm -f output/*.png

## Convenience combo

# Default full workflow uses the exec path for better timeout handling.
colab-all: colab-upload colab-install colab-run-exec
