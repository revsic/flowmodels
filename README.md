# flowmodels

PyTorch-implementations of Flow Models for toy data

## Usage

Install the package.

```bash
git clone https://github.com/revsic/flowmodels
cd flowmodels && pip install -e .
```

Here is the sample code[[samples/ddpm.ipynb](./samples/ddpm.ipynb)]:

```py
import torch.nn as nn

from flowmodels import DDPM, DDIMScheduler


model = DDPM(nn.Sequential(...), DDIMScheduler())

# update
optim = torch.optim.Adam(model.parameters(), LR)
for i in range(TRAIN_STEPS):
    optim.zero_grad()
    model.loss(batch).backward()
    optim.step()

# sample
sampled, trajectory = model.sample(torch.randn(...))
```

## Implemented Models

| Method | Paper | Imports & Examples |
|---|---|---|
| Drifting Model | *Generative Modeling via Drifting*[[arXiv:2602.04770](https://arxiv.org/abs/2602.04770)] | `DriftingModel`<br>[samples/drifting.ipynb](./samples/drifting.ipynb) |
| AYF-EMD | *Align Your Flow: Scaling Continuous-Time Flow Map Distillation*[[arXiv:2506.14603](https://arxiv.org/abs/2506.14603)] | `AlignYourFlow`<br>[samples/alignyourflow.ipynb](./samples/alignyourflow.ipynb) |
| MeanFlow | *Mean Flows for One-step Generative Modeling*[[arXiv:2505.13447](https://arxiv.org/abs/2505.13447)] | `MeanFlow`<br>[samples/meanflow.ipynb](./samples/meanflow.ipynb) |
| IMM | *Inductive Moment Matching*[[arXiv:2503.07565](https://arxiv.org/abs/2503.07565)] | `InductivMomentMatching`<br>[samples/imm.ipynb](./samples/imm.ipynb) |
| f-DMD | *One-step Diffusion Models with f-Divergence Distribution Matching*[[arXiv:2502.15681](https://arxiv.org/abs/2502.15681)] | `DistributionMatchingDisillation`, method `dmd2` with `h="jensen-shannon"`<br>[samples/dmd.ipynb](./samples/dmd.ipynb) |
| FlowEdit | *Inversion-Free Text-Based Editing Using Pre-Trained Flow Models*[[arXiv:2412.08629](https://arxiv.org/abs/2412.08629)] | `FlowEditSolver`<br>[samples/flowedit.ipynb](./samples/flowedit.ipynb) |
| FireFlow | *Fast Inversion of Rectified Flow for Image Semantic Editing*[[arXiv:2412.07517](https://arxiv.org/abs/2412.07517)] | `FireFlowSolver`, `FireFlowInversion`<br>[samples/rf.ipynb](./samples/rf.ipynb), 4.5. Inversion Methods |
| Controlled ODE | *Semantic Image Inversion And Editing Using Rectified Stochastic Differential Equations*[[arXiv:2412.00100](https://arxiv.org/abs/2412.00100)] | `ControlledODESolver`, `ControlledODEInversion`<br>[samples/rf.ipynb](./samples/rf.ipynb), 4.5. Inversion Methods |
| RF-Solver | *Taming Rectified Flow for Inversion and Editing*[[arXiv:2411.04746](https://arxiv.org/abs/2411.04746)] | `RFSolver`, `RFInversion`<br>[samples/rf.ipynb](./samples/rf.ipynb), 4.5. Inversion Methods |
| CAF | *Constant Acceleration Flow*[[arXiv:2411.00322](https://arxiv.org/abs/2411.00322)] | `ConstantAccelerationFlow`<br>[samples/caf.ipynb](./samples/caf.ipynb) |
| Shortcut Model | *One Step Diffusion via Shortcut Models*[[arXiv:2410.12557](https://arxiv.org/abs/2410.12557)] | `ShortcutModel`, `ShortcutEulerSolver`<br>[samples/shortcut.ipynb](./samples/shortcut.ipynb) |
| Rectified Diffusion | *Straightness Is Not Your Need in Rectified Flow*[[arXiv:2410.07303](https://arxiv.org/abs/2410.07303)] | `RectifiedDiffusion`<br>[samples/rd.ipynb](./samples/rd.ipynb) |
| sCT | *Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models*[[arXiv:2410.11081](https://arxiv.org/abs/2410.11081)] | `ScaledContinuousCM`, `ScaledContinuousCMScheduler`<br>[samples/sct.ipynb](./samples/sct.ipynb) |
| Consistency Flow Matching | *Defining Straight Flows with Velocity Consistency*[[arXiv:2407.02398](https://arxiv.org/abs/2407.02398)] | `ConsistencyFlowMatching`<br>[samples/consistencyfm.ipynb](./samples/consistencyfm.ipynb) |
| ECT | *Consistency Models Made Easy*[[arXiv:2406.14548](https://arxiv.org/abs/2406.14548)] | `EasyConsistencyTraining`<br>[samples/ect.ipynb](./samples/ect.ipynb) |
| FMM | *Flow map matching with stochastic interpolants: A mathematical framework for consistency models*[[arXiv:2406.07507](https://arxiv.org/abs/2406.07507)] | `FlowMapMatching`<br>[samples/fmm.ipynb](./samples/fmm.ipynb) |
| DMD2 | *Improved Distribution Matching Distillation for Fast Image Synthesis*[[arXiv:2405.14867](https://arxiv.org/abs/2405.14867)] | `DistributionMatchingDisillation`, method `dmd2`<br>[samples/dmd.ipynb](./samples/dmd.ipynb) |
| DMD | *One-step Diffusion with Distribution Matching Distillation*[[arXiv:2311.18828](https://arxiv.org/abs/2311.18828)] | `DistributionMatchingDisillation`, method `dmd`<br>[samples/dmd.ipynb](./samples/dmd.ipynb) |
| InstaFlow | *One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*[[arXiv:2309.06380](https://arxiv.org/abs/2309.06380)] | `InstaFlow`<br>[samples/instaflow.ipynb](./samples/instaflow.ipynb) |
| Consistency Models | *Consistency Models*[[arXiv:2303.01469](https://arxiv.org/abs/2303.01469)] | `ConsistencyModel`, `MultistepConsistencySampler`<br>[samples/cm.ipynb](./samples/cm.ipynb) |
| DSBM | *Diffusion Schrodinger Bridge Matching*[[arXiv:2303.16852](https://arxiv.org/abs/2303.16852)] | `DiffusionSchrodingerBridgeMatching`, `ModifiedVanillaEulerSolver`<br>[samples/dsbm.ipynb](./samples/dsbm.ipynb) |
| Rectified Flow | *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*[[arXiv:2209.03003](https://arxiv.org/abs/2209.03003)] | `RectifiedFlow`, `VanillaEulerSolver`<br>[samples/rf.ipynb](./samples/rf.ipynb) |
| VPSDE / VESDE / PF-ODE | *Score-Based Generative Modeling through Stochastic Differential Equations*[[arXiv:2011.13456](https://arxiv.org/abs/2011.13456)] | `VPSDE`, `VPSDEAncestralSampler`, `VPSDEScheduler`<br>`VESDE`, `VESDEAncestralSampler`, `VESDEScheduler`<br>`ProbabilityFlowODESampler`<br>[samples/vpsde.ipynb](./samples/vpsde.ipynb), [samples/vesde.ipynb](./samples/vesde.ipynb), [samples/ddpm.ipynb](./samples/ddpm.ipynb) |
| DDIM | *Denoising Diffusion Implicit Models*[[arXiv:2010.02502](https://arxiv.org/abs/2010.02502)] | `DDIMScheduler`, `DDIMSampler`<br>[samples/ddpm.ipynb](./samples/ddpm.ipynb), 4. Test the model |
| DDPM | *Denoising Diffusion Probabilistic Models*[[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)] | `DDPM`, `DDPMScheduler`, `DDPMSampler`<br>[samples/ddpm.ipynb](./samples/ddpm.ipynb) |
| NCSN | *Generative Modeling by Estimating Gradients of the Data Distribution*[[arXiv:1907.05600](https://arxiv.org/abs/1907.05600)] | `NCSN`, `NCSNScheduler`, `AnnealedLangevinDynamicsSampler`<br>[samples/ncsn.ipynb](./samples/ncsn.ipynb) |