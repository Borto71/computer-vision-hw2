# Utilities

This directory collects reusable helpers, shared abstractions, and project-wide constants. It functions as the common toolkit that supports notebooks, data pipelines, and training workflows.

## Core Artifacts
- `utils.py` — general-purpose utilities (for example, logging hooks, metric formatters, seeding helpers) that are referenced across multiple modules.
- Configuration loaders, argument parsers, and serialization helpers whose scope extends beyond `models/` or `data/`.

When the number of helpers grows, partition them into focused modules such as `metrics.py` or `visualization.py` to preserve clarity.

## Design Principles
- Favour pure functions when possible; any side effects should be intentional and well documented.
- Provide docstrings that explain intent and context so that new contributors can adopt the utilities confidently.
- Prevent circular dependencies by ensuring that utilities depend only on core libraries, not on higher-level application code.
- Add lightweight unit tests or illustrative examples when the expected behaviour is not obvious.

## Inclusion Criteria
1. The logic is broadly applicable across multiple components of the repository.
2. The helper eliminates duplication that would otherwise be addressed through copy-and-paste code.
3. The functionality is small and stable; larger subsystems should be promoted to a dedicated top-level package.

## Maintenance Guidance
- Remove obsolete or unused helpers during refactoring efforts to minimise cognitive load.
- Prefer explicit imports (for example, `from utils.metrics import accuracy`) over wildcard imports.
- After modifying shared helpers, rerun the relevant smoke tests or scripts under `scripts/` (if available) to confirm compatibility.

Treat this directory as a curated toolbox; each addition should demonstrably simplify ongoing development.
