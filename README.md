# Script used in: Certification of genuine multipartite entanglement in spin ensembles with measurements of total angular momentum [arXiv-2311.00806]
Script used in [arXiv:2311.00806](https://arxiv.org/abs/2311.00806) for finding upper and lower bounds of the separable bound of the precession protocol. See the paper for more detail.

# Usage
For upper bounds
    julia SDP.jl d₁ d₂ K₁ K₂ [file]
For lower bounds
    julia SPI.jl d₁ d₂ K₁ K₂ [file]

## Arguments
- `d₁::Integer`: Dimension d₁ = 2j₁ + 1 of the first spin system
- `d₂::Integer`: Dimension d₁ = 2j₂ + 1 of the first spin system
- `K₁::Integer`: Minimum (odd) number of measurement settings
- `K₂::Integer`: Maximum (odd) number of measurement settings
- `file::String`: Data will be saved in $file-K$K.csv. Returns to standard output if none specified

# Package versions
For the reported results, version 1.6.6 of `julia` was used, with the packages
- `JuMP`: v1.12.0
- `COSMO`: v0.8.8
- `SCS`: v1.1.2
- `WignerSymbols`: v2.0.0
- `Memoization`: v0.2.1
- `ArnoldiMethod`: v0.2.0

# Generated Data
The generated data used in the paper is also available in the `data` folder. `SDP-K$K.csv` and `SPI-K$K.csv` were plotted in Figs. 2–5, 8, A1–A3; while `SDP-Spin-One.csv` and `SPI-Spin-One.csv` were plotted in Fig. 6.
