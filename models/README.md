# Model Directory

This directory contains all trainable artefacts, including reference architectures, experimental prototypes, and production-ready checkpoints. Use it as the central catalogue for maintaining model implementations.

## Current Contents
- `mlp.py` — baseline multilayer perceptron suitable for tabular or flattened inputs.
- `cnn.py` — convolutional architecture optimised for rapid experimentation on image data.

Extend these examples when developing new approaches or refining existing baselines.

## Organisational Principles
- Keep each architecture self-contained with explicit constructor parameters.
- Extract shared layers, losses, or utilities into dedicated modules or the shared `utils/` package.
- Version significant milestones using tags, checkpoints, or concise `MODEL_CARD.md` files.
- Document design choices inline when behaviour is not self-evident.

## Adding a New Model
1. Create a descriptively named module (for example, `transformer_text_cls.py`) and implement the architecture.
2. Provide a factory function such as `build_model(config)` to simplify instantiation in training scripts.
3. Include quick-start guidance or usage examples within the module docstring.
4. Update related experiments, configuration files, or notebooks to reference the new component.

## Recommended Practices
- Select conservative default hyperparameters so that reference runs succeed on modest hardware.
- Prefer composition over inheritance to preserve readability and testability.
- Ensure training-time assumptions (for example, input dimensions or normalisation) are mirrored in preprocessing workflows under `data/`.
- Start from a simple baseline, validate performance, and iterate incrementally.

Each addition to this directory should clarify how the project approaches modelling and provide a reliable foundation for future work.
