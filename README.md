# Low-Chrome White Cast Iron Mill Ball Solidification Simulation

A physics-informed 1D radial finite difference model for predicting 
solidification behaviour and casting defect risk in low-chromium white 
cast iron grinding balls (60–120 mm) in a permanent metal mould.

## Quick Start — Run in Browser (No Installation Required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songeyargky/CastingSimulation/blob/main/notebook.ipynb)

Click the badge above. No Python installation needed.

## What the Simulation Does

- Solves the 1D radial heat equation using backward-Euler FDM (N=40 nodes)
- Models two-stage latent heat release (primary + eutectic)
- Predicts four casting defect risk indices: Misrun, Cold Shut, 
  Surface Cracking, and Warpage
- Runs a 400-scenario parameter sweep over T_pour, T_mold, and diameter
- Produces GP response surfaces, RF sensitivity analysis, and 
  composite risk heatmaps

## Ball Sizes

60 mm | 80 mm | 100 mm | 120 mm

## Alloy

Low-chrome white cast iron: ~3 wt% C, 1–3 wt% Cr

## Module

Module L99 — Computational Casting Engineering

## Dependencies

See requirements.txt. All open-source (NumPy, Matplotlib, Pandas, 
scikit-learn, SciPy).

## Live Web Application
https://castingsimulation-jc8rybquyjkzfxbfd7tdex.streamlit.app/