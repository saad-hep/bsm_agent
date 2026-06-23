# bsm_agent

`bsm_agent` is a symbolic toolkit for building renormalizable Standard Model extensions and deriving their gauge-invariant operator content, component-expanded interactions, electroweak symmetry breaking relations, and tree-level mass matrices.

The repository also includes the LLM-assisted workflow used to generate model reports and examples discussed in the accompanying paper. In that sense, this repo is both:

- a symbolic physics package
- a reproducible agent-assisted model-building workflow

## Features

- symbolic construction of renormalizable SM extensions
- automatic generation of gauge-invariant scalar, fermion-mass, and Yukawa operators
- component expansion of operators
- electroweak symmetry breaking support
- tree-level scalar and fermion mass-matrix extraction
- report generation in LaTeX/PDF form
- optional multi-VEV EWSB reporting for neutral scalar sectors
- chat/agent entrypoint for LLM-assisted model construction

## Installation

Create a Python environment and install from the repository root:

```bash
pip install -e .
```

## Citation

If you use `bsm_agent` in your research or reference the LLM-assisted workflow, please cite the accompanying paper:

```bibtex
@article{Saad:2026pan,
    author = "Saad, Shaikh",
    title = "{Large Language Model-Assisted Framework for BSM Model Building}",
    eprint = "2606.21316",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "6",
    year = "2026"
}
```

## Quick Start  
Please see Appendix A of the arXiv pre-print:

*Shaikh Saad, "Large Language Model-Assisted Framework for BSM Model Building", [arXiv:2606.21316 [hep-ph]](https://arxiv.org/abs/2606.21316)*


