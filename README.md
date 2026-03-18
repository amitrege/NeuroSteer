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
