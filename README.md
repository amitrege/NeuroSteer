# NeuroSteer

Small activation steering tools for Llama-style decoder models.

This repo is intentionally narrow:

- Hugging Face decoder-only models with `model.layers`
- Decoder block outputs only
- CPU-testable core runtime with optional GPU notebook workflows kept outside version control

The library is organized around:

- locating decoder blocks
- reading activations
- writing reversible changes
- fitting low-rank bases from prompt pairs
- applying fixed, thresholded, lagged, or preview-based steering
- recording traces for debugging

## Colab setup

For Colab, install the package before importing anything from the repo:

```bash
%cd /content/neurosteer
%pip install -q -e .
```

Then verify the active versions:

```bash
!python -c "import sys, numpy, torch, transformers; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"
```

Do not run `pip install --upgrade numpy` mid-session in Colab for this repo. If you change core binary packages after the runtime has already imported them, Colab will usually require a full restart.

## Reproducible CPU test stack

If you want the exact CPU environment used during local validation, install:

```bash
pip install -r validated-cpu-requirements.txt
pip install -e .
```
