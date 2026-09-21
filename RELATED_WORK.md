# RobustPixelMaker: Related Work

**Purpose.** This document surveys the prior art that RobustPixelMaker (RPM) builds on, positions RPM against its closest neighbours, and states explicitly what is borrowed versus what had to be invented. It is the evidence base for the library's novelty claims. Every pillar of RPM has precedent; the contribution is a specific synthesis plus one genuinely unclaimed idea (representation-marginalised, pixel-direct *selection* under a leakage-safe contract). Read this before drafting the paper's Related Work section or defending novelty in review.

Last updated: 2026-09-15.

**Status of this document.** It was written before the library was built, as the
evidence base for a novelty claim, and parts of it are still in that forward-looking
voice. Where an experiment it proposed has since been run, the result is recorded
beside the proposal rather than replacing it, so the prediction and the outcome can be
read against each other. Section 12 carries the important case: the make-or-break test
was run and its stated criterion was **not** met.

---

## 0. Lineage: the parent frameworks (Barnard group)

- **RobustModelMaker (RMM)**, Barnard, *RobustModelMaker: Coupling Bootstrap Stability Selection with Leakage-Safe Nested Cross-Validation for Scientific Machine Learning* (2026), [arXiv:2606.01566](https://arxiv.org/pdf/2606.01566). The tabular parent. Couples Meinshausen-Bühlmann bootstrap stability selection with strict nested CV, fits all preprocessing/selection inside each outer fold, reports a stability-tested subset + leakage-safe score + the score-stability frontier. RPM is the pixel/voxel-direct sibling; it inherits RMM's contract, its reproducibility discipline, its `preserved/sig.better/sig.worse` verdict, and its Jaccard stability metric.
- **Variance Tolerance Factors (VTF)**, Li & Barnard, *Variance Tolerance Factors For Interpreting All Neural Networks*, IJCNN 2023, DOI 10.1109/IJCNN54540.2023.10191646. Introduces a mask layer over a **frozen base model** trained to recover base performance; mask weights = importance in a Rashomon set. RPM reuses the freeze-base/train-mask trick for tractability and the Rashomon framing. VTF already demonstrated pixel-level masks on MNIST (registered, near-binary, the easy case RPM must generalise beyond).
- **BenchMake**, Barnard (2024), archetypal train/test partitioning, [github.com/amaxiom/benchmake](https://github.com/amaxiom/benchmake). Adversarial, representative splits used by RMM's benchmark suite; RPM should use it for conservative, distribution-aware image splits.

**Gap RPM fills relative to the parents:** RMM works on tabular columns; VTF explains a single frozen net on (mostly) registered pixels. Neither does *reproducible, leakage-safe, representation-agnostic selection directly on image pixels/voxels that yields a robust predictor*.

---

## 1. Stability selection (statistics)

- **Meinshausen & Bühlmann**, *Stability Selection*, JRSS-B 72(4):417-473, 2010. Aggregate selections over bootstrap subsamples; selection frequency πₖ; threshold τ. The statistical core RPM ports to pixel space.
- **Shah & Samworth**, *Variable selection with error control: another look at stability selection*, JRSS-B 75(1):55-80, 2013. Complementary-pairs subsampling + explicit bound on expected false selections. **Relevant to RPM's cost problem:** tighter error control from fewer subsamples → fewer inner fits in the bootstrap loop.
- **Hofner, Boccuto & Göker**, *Controlling false discoveries … boosting with stability selection*, BMC Bioinformatics 16:144, 2015. Stability selection inside gradient boosting; precedent for embedding it in a non-linear learner.

**Borrowed:** the πₖ / τ machinery and the above-median per-run selection rule (algorithm-agnostic). **Gap:** none of this is spatial; the pixel autocorrelation problem (below) is new to the image setting.

---

## 2. Leakage-safe evaluation / nested cross-validation

- **Varma & Simon**, *Bias in error estimation when using CV for model selection*, BMC Bioinformatics 7:91, 2006. Documents optimistic bias.
- **Cawley & Talbot**, *On over-fitting in model selection and subsequent selection bias*, JMLR 11:2079-2107, 2010.
- **Krstajic et al.**, *Cross-validation pitfalls …*, J. Cheminformatics 6:10, 2014.
- **Vabalas et al.**, *Machine learning algorithm validation with a limited sample size*, PLoS One 14(11):e0224365, 2019. Small-sample inflation grows with search space; nested CV removes it. Identifies low-hundreds N as worst regime, the scientific-imaging operating point.
- **TRIPOD**, Collins et al., BMJ 350:g7594, 2015; **Roberts et al.**, *Common pitfalls … COVID-19 imaging*, Nat. Mach. Intell. 3:199-217, 2021 (none of 62 imaging studies clinically usable, largely due to leakage). **Directly motivates RPM's imaging leakage discipline (esp. subject-level grouping).**

**Borrowed wholesale:** RMM's per-fold "fit-on-train-only" contract. **Gap / must-invent:** image pipelines have *more* leakage channels (registration, normalisation stats, superpixel maps, augmentation policy, the baseline-fill image), each of which must be fitted inside the fold; subject/specimen-level `GroupKFold` is mandatory (slice leakage).

---

## 3. Differentiable / learnable-mask feature selection (prior art for the mask mechanism)

- **Balın, Abid & Zou**, *Concrete Autoencoders: Differentiable Feature Selection and Reconstruction*, ICML 2019, [arXiv:1901.09346](https://arxiv.org/pdf/1901.09346). Concrete selector layer → k discrete input features by test time.
- **Yamada et al.**, *Feature Selection using Stochastic Gates (STG)*, ICML 2020. Bernoulli-relaxed gates with an L0-style count penalty, end-to-end.
- **SLM: End-to-end Feature Selection via Sparse Learnable Masks**, 2023, [arXiv:2304.03202](https://arxiv.org/abs/2304.03202). Learnable sparse mask maximising MI(selected; label); **explicitly visualises learned pixel masks on MNIST/fMNIST**. Closest prior art for "learnable mask selects pixels."
- **Louizos, Welling & Kingma**, *Learning Sparse Neural Networks through L0 Regularization*, ICLR 2018, [arXiv:1712.01312](https://arxiv.org/abs/1712.01312). **Hard-Concrete gates**, stochastic, differentiable, exactly 0/1 at test. RPM's recommended mask parameterisation.

**Borrowed:** the mask parameterisation (sigmoid+L1 for the MVP, Hard-Concrete for true L0). **Explicit non-claim:** RPM does **not** claim the learnable mask as novel; Concrete AE / STG / SLM own it, including pixel visualisation. Leading with "we learn a mask" would read as incremental.

---

## 4. Rashomon-set variable importance (conceptual ancestor)

- **Fisher, Rudin & Dominici**, *Model Class Reliance (MCR): … from the Rashomon perspective*, JMLR 20(177):1-81, 2019. Range of variable importance across all well-performing models.
- **Dong & Rudin**, *Variable Importance Clouds / Exploring the cloud of variable importance for the set of all good models*, Nat. Mach. Intell. 2:810-824, 2020. The set of model-reliance vectors across the Rashomon set.
- **A Guide to Feature Importance Methods for Scientific Inference**, 2024, [arXiv:2404.12862](https://arxiv.org/pdf/2404.12862). Useful framing citation for the "importance for science" audience.

**Relationship:** RPM's representation-marginalisation is the empirical, pixel-space, learnable-mask analogue of "importance across all good models." MCR/VIC compute importance *ranges* analytically/by search over a model class for *tabular* features; RPM *marginalises over representations empirically in pixel space and returns a predictor*. Cite as lineage (and VTF already invoked Rashomon sets), not as something RPM invents.

---

## 5. Masking mechanisms for images (how to represent "absent")

- **Liu et al. (NVIDIA)**, *Image Inpainting for Irregular Holes Using Partial Convolutions*, ECCV 2018, [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.html). Convolution **masked and renormalised to be conditioned on valid pixels only**, with an automatically updated mask propagated through the forward pass. The key idea RPM keeps: masked = *absent*, not zero.
- **Yu et al.**, *Free-Form Image Inpainting with Gated Convolution*, ICCV 2019, [arXiv:1806.03589](https://arxiv.org/abs/1806.03589). Generalises partial conv to a **learnable, soft, per-channel/location** gate. Conceptual justification for RPM using a *soft learnable* mask rather than PConv's hard data-given mask.
- **He et al.**, *Masked Autoencoders Are Scalable Vision Learners (MAE)*, CVPR 2022. Random patch masking for pretraining, a candidate mask-native backbone / SSL pretext for the representation ensemble.

**Design decision:** RPM drops PConv-the-layer but **keeps PConv-the-renormalisation** on a soft learnable mask: `F̃ = (m ⋆ F)/(m ⋆ 1 + ε)`. This avoids the zero-fill out-of-distribution confound that PConv was invented to solve, while giving the learnable soft gate that Yu et al. argued for. Simpler fallback: neutral baseline fill (per-fold mean/blur).

---

## 6. Attribution / saliency and its reproducibility problems (why "selection" ≠ "explanation")

- **Selvaraju et al.**, Grad-CAM, ICCV 2017; **Petsiuk et al.**, RISE, BMVC 2018; occlusion (Zeiler & Fergus 2014). Post-hoc, per-image attribution.
- **Adebayo et al.**, *Sanity Checks for Saliency Maps*, NeurIPS 2018; **Kindermans et al.**, *The (Un)reliability of saliency methods*, 2019. Documented fragility/instability of saliency, the reproducibility crisis RPM's stability machinery answers.
- **Harmonizing Feature Attributions Across Deep Learning Architectures**, 2023, [arXiv:2307.02150](https://arxiv.org/pdf/2307.02150). Cross-architecture attribution consistency.
- **Fidelity of Ensemble Aggregation for Saliency Maps**, 2022, [arXiv:2207.01565](https://arxiv.org/pdf/2207.01565). Aggregating attributions across models.
- **Reproducibility of Brain-Age Saliencies Across DNN Architectures**, [PMC10737381](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10737381/). Cross-architecture saliency reproducibility **in neuroimaging**, closest on the representation axis, in RPM's target domain.

**Critical distinction to hold in the paper:** all of these are **post-hoc explanation**: they interpret an already-fitted model. RPM is a **pre-training selection** that produces a mask + robust predictor with error control. Do not let a reviewer collapse RPM into "yet another attribution-stability paper."

---

## 7. Ensembling for selection stability (the CLOSEST threats)

- **Gyawali, Liu, Zou & He**, *Ensembling improves stability and power of feature selection for deep learning models*, MLCB / PMLR 200, 2022, [PMLR](https://proceedings.mlr.press/v200/gyawali22a.html). **Single most dangerous neighbour.** Ensembles feature-importance across *good* models (chosen by CV loss), over **hyperparameters and epochs**, with a **knockoff** framework for FDR control. Improves FS stability + power on simulated + real data.
- **Ensemble feature selection (bioinformatics)**, Abeel et al., Bioinformatics 26(3):392-398, 2010; Haury et al., PLoS One 6(12):e28210, 2011. Ensemble/bootstrap selection improves signature stability at no accuracy cost, the tabular precedent RMM already cites.
- **Barber & Candès**, *Controlling the FDR via knockoffs*, Annals of Statistics 2015. Error-control alternative to bootstrap-frequency; used by Gyawali.

**How RPM differs from Gyawali (must state explicitly):** (i) ensembles over **representations/architectures** as a deliberate nuisance-variable marginalisation, not hyperparameters/epochs; (ii) a **learnable renormalised pixel mask → predictor**, not importance-score ensembling + knockoffs; (iii) **pixel/voxel-direct** on images, not tabular/genomic; (iv) **leakage-safe nested-CV coupling**, absent from Gyawali. The naive claim "ensembling for FS stability is new" is **false**; do not make it.

---

## 8. Background bias, shortcut learning, robustness-via-masking (the "robust predictor" payoff)

- **Masking Strategies for Background Bias Removal in Computer Vision Models**, 2023, [ar5iv:2308.12127](https://ar5iv.labs.arxiv.org/html/2308.12127). Background masking improves OOD; **early masking generalises best** (CNN & ViT). Direct support for RPM's robustness thesis and for masking at the input rather than late.
- **Bassi et al. (ISNet / Background Relevance Minimisation)**, *Improving DNN generalisation and robustness to background bias via LRP optimisation*, Nature Communications 14, 2023, [s41467-023-44371-z](https://www.nature.com/articles/s41467-023-44371-z). Suppress background via LRP/segmentation supervision.
- **Xiao et al.**, *Noise or Signal: The Role of Image Backgrounds in Object Recognition*, ICLR 2021; **Geirhos et al.**, *Shortcut Learning in Deep Neural Networks*, Nat. Mach. Intell. 2020. Establish that backgrounds drive shortcut learning.

**Key distinction:** existing background-masking methods use **ground-truth segmentation or LRP supervision** to decide what to suppress. RPM **discovers** the informative region via reproducible, **label-free** selection with error control. The experiment this implies, showing RPM matches or beats segmentation-supervised masking *without* masks or labels, **has not been run**: no benchmark in the suite compares against a segmentation-supervised method. The distinction above is therefore an argument, not a result, and should not be cited as one.

---

## 9. Domain: neuroimaging feature/voxel selection & the reproducibility crisis

- **Exploring stability-based voxel selection methods in MVPA**, [PMC4999569](https://pmc.ncbi.nlm.nih.gov/articles/PMC4999569/) (2016). Bootstrapped stability selection on fMRI voxels with classical selectors (MI, RFE-SVM, L1/L2), scored on accuracy **and** stability. **Closest domain prior art**; a reviewer will cite it. RPM differs: learnable renormalised mask, representation-marginalised, leakage-coupled, pixel-direct (not pre-reduced voxel activations).
- **Stable Feature Selection from Brain sMRI**, 2015, [arXiv:1503.07508](https://arxiv.org/pdf/1503.07508).
- **Marek et al.**, *Reproducible brain-wide association studies require thousands of individuals*, Nature 603:654-660, 2022, [Nature](https://www.nature.com/articles/s41586-022-04492-9). ~50k subjects (ABCD, HCP, UK Biobank); most prior BWAS underpowered → inflated, irreproducible effects. **The headline motivation** for treating imaging reproducibility as a first-class deliverable.
- **Mwangi et al.**, *A review of feature reduction techniques in neuroimaging*, Neuroinformatics 12:229-244, 2014.

**Why neuroimaging is the lead application:** the pain (irreproducibility) is acute and published; the audience already values stability; registration to atlas space makes the global-mask regime well-posed; atlas ROIs give interpretable selected regions.

---

## 10. Domain: electron microscopy / materials imaging (the hard, high-value case)

- **Ede**, *Deep learning in electron microscopy*, Mach. Learn.: Sci. Technol. 2021, [IOP](https://iopscience.iop.org/article/10.1088/2632-2153/abd614). Survey; DL used for detection/segmentation/classification/denoising.
- STEM/SEM denoising & noise-level classification (Noise2Void-style; EstimateNoiseSEM 2025). Establishes that TEM/SEM "signal-through-noise" is an active, hard problem.
- **Barnard & Opletal**, gold nanoparticle dataset (CSIRO, 2019) & *Selecting ML models for metallic nanoparticles*, Nano Futures 4:035003, 2020. In-group data for a materials application.

**Why TEM is the second paper, not the first:** wide-field micrographs are **not registered**: background (vacuum, support film, scale bar) is content-defined, not location-defined. This forces the **conditional-mask** regime, where the stable object is a *gating function*, not a coordinate region, and the stability metric changes. Exception: cryo-EM-style single-particle cropping *re-registers* the data and admits the global mask.

---

## 11. Granularity: superpixels / patches (beating spatial autocorrelation)

- **Achanta et al.**, *SLIC Superpixels Compared to State-of-the-art Superpixel Methods*, IEEE TPAMI 34(11):2274-2282, 2012.
- Atlas ROIs (e.g. AAL, Harvard-Oxford) for registered MRI, anatomically named selection units.

**Why it matters:** neighbouring pixels are near-equivalent, so pixel-level masks swap freely and never stabilise (the image analogue of RMM's within-family descriptor correlation). Selecting at **superpixel / patch / atlas-ROI** granularity + a spatial-smoothness (TV) prior on the mask restores stability while staying pixel-indexed (a regulariser on `m`, **not** a feature transform, consistent with the "no engineered collapse" philosophy). Bootstrap-frequency averaging further absorbs jitter.

---

## 12. Synthesis: what we borrow vs where the gap is

| Component | Nearest prior art | RPM stance |
|---|---|---|
| Bootstrap πₖ/τ stability selection | Meinshausen-Bühlmann; Shah-Samworth | **Borrow** wholesale |
| Leakage-safe nested CV contract | RMM; Varma-Simon; Cawley-Talbot | **Borrow** from RMM |
| Learnable soft mask / L0 gate | Concrete AE; STG; SLM; Louizos Hard-Concrete | **Borrow**; not claimed |
| Renormalised "absent" masking | Partial conv (Liu); gated conv (Yu) | **Borrow the renormalisation**, soft+learnable |
| Freeze-base, train-mask for tractability | VTF | **Borrow** |
| Importance across all good models | MCR; VIC (Dong-Rudin) | **Borrow as lineage** |
| Ensembling for FS stability | **Gyawali et al. 2022** | **Distinguish**: representations not hyperparams; mask→predictor not scores+knockoffs; images; leakage-coupled |
| Cross-architecture attribution stability | Harmonizing; brain-age saliency | **Distinguish**: selection not post-hoc explanation |
| Background-masking → robustness | ISNet/BRM; 2308.12127 | **Distinguish**: label-free discovery, not segmentation-supervised |
| Stability selection on voxels | MVPA (PMC4999569) | **Distinguish**: learnable mask, representation-marginalised, pixel-direct |
| **Representation-as-nuisance marginalisation for pixel-direct *selection* under a leakage-safe contract, yielding a robust predictor + reproducible region** | **none found** | **INVENT (the contribution)** |

**One-line novelty (defensible):** *reproducible, leakage-safe feature selection that works directly on image pixels/voxels and does not commit to any representation, because it marginalises the selection over both data resamples and representations, so the retained region is what survives when the representation is treated as noise.*

**Make-or-break experiment:** ablate the representation axis (single frozen representation ≈ Concrete/STG/SLM vs representation-marginalised). If marginalisation does not measurably improve stability, the contribution collapses into prior art.

### The make-or-break experiment has been run, and the claim above must be read against it

This section was written before the evidence. It has not been rewritten, so that the
prediction and the result can be compared, but **the novelty statement above is
stronger than what the data supports** and should not be used as drafted.

**The premise holds.** Lenses agree with each other at Jaccard only 0.37 to 0.41, so a
single-lens result genuinely is a lens-specific result. That much is measured, not
assumed, and it is the part of the argument that survives intact.

**The criterion as stated was not met.** On the synthetic control marginalisation lands
*below* the best single lens on stability, 0.890 against 0.923, and on recovery, 0.967
against 0.978, at the same coverage. It does not "measurably improve stability" over the
best lens. What it does is land within 0.03 of the best while far exceeding the average
lens (0.62) and the worst (0.40), without having to know in advance which lens was
good. On breast the raw table is confounded (coverage and stability correlate at
r = +0.99, so the lens keeping three quarters of the image looks the most stable);
compared at matched coverage marginalisation is the best of the comparable three,
though all are poor. FINDINGS sec.9.

**So the defensible claim is insurance, not victory:** marginalisation removes the risk
of choosing a bad representation rather than beating a good one. That is a real and
statable contribution, and it is weaker than "the retained region is what survives when
the representation is treated as noise", which implies a gain this ablation did not find.

**An earlier, stronger figure should not be cited.** A first run reported marginalised
stability 0.960, beating the best lens. That was a seed-collision defect in the ensemble
path (see CHANGELOG.md); with independent seeds it is 0.890, while
every single-lens figure was unchanged to three decimals.

**Two later results bear on the rest of this document.** The learned mask, which section
5 above explicitly does *not* claim as novel, adds nothing over random-forest importance
when the bootstrap wrapper is held fixed, and is the less stable of the two, 0.67
against 0.89 (FINDINGS sec.15). And the mask's head cannot recover a purely interactive
region at all, 1.2x the chance rate, which is a structural limit of the freeze-base
scheme rather than a tuning problem (sec.19). Neither touches the leakage-safe contract
or the marginalisation axis, but both narrow what can be claimed for the mask itself.

Anyone drafting a paper from this document should start from FINDINGS sections 9, 14,
15, 16 and 19 rather than from the table above.

---

## 13. Grouped reference list (with links)

**Parents:** RMM [arXiv:2606.01566](https://arxiv.org/pdf/2606.01566) · VTF (IJCNN 2023, DOI 10.1109/IJCNN54540.2023.10191646) · BenchMake [github](https://github.com/amaxiom/benchmake).
**Stability selection:** Meinshausen-Bühlmann JRSS-B 2010 · Shah-Samworth JRSS-B 2013 · Hofner BMC Bioinf. 2015.
**Nested CV / leakage:** Varma-Simon 2006 · Cawley-Talbot JMLR 2010 · Krstajic 2014 · Vabalas PLoS One 2019 · TRIPOD BMJ 2015 · Roberts NMI 2021.
**Learnable masks:** Concrete AE [1901.09346](https://arxiv.org/pdf/1901.09346) · STG ICML 2020 · SLM [2304.03202](https://arxiv.org/abs/2304.03202) · Louizos L0 [1712.01312](https://arxiv.org/abs/1712.01312).
**Rashomon importance:** MCR JMLR 2019 · VIC/Dong-Rudin NMI 2020 · Guide to FI [2404.12862](https://arxiv.org/pdf/2404.12862).
**Image masking mechanisms:** Partial conv (Liu, ECCV 2018) · Gated conv [1806.03589](https://arxiv.org/abs/1806.03589) · MAE CVPR 2022.
**Attribution & reproducibility:** Grad-CAM 2017 · RISE 2018 · Adebayo NeurIPS 2018 · Harmonizing [2307.02150](https://arxiv.org/pdf/2307.02150) · Ensemble saliency [2207.01565](https://arxiv.org/pdf/2207.01565) · Brain-age saliency [PMC10737381](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10737381/).
**Ensembling for FS stability:** Gyawali PMLR 200 2022 [link](https://proceedings.mlr.press/v200/gyawali22a.html) · Abeel 2010 · Haury 2011 · Knockoffs (Barber-Candès 2015).
**Robustness via masking:** Masking strategies [2308.12127](https://ar5iv.labs.arxiv.org/html/2308.12127) · ISNet/BRM [Nat.Comms 2023](https://www.nature.com/articles/s41467-023-44371-z) · Xiao ICLR 2021 · Geirhos NMI 2020.
**Neuroimaging:** MVPA voxel stability [PMC4999569](https://pmc.ncbi.nlm.nih.gov/articles/PMC4999569/) · Stable sMRI FS [1503.07508](https://arxiv.org/pdf/1503.07508) · Marek Nature 2022 [link](https://www.nature.com/articles/s41586-022-04492-9) · Mwangi 2014.
**Microscopy/materials:** Ede MLST 2021 [IOP](https://iopscience.iop.org/article/10.1088/2632-2153/abd614) · Barnard-Opletal Nano Futures 2020.
**Benchmark image collections:** Yang et al., *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data 10:41, 2023 (standardised 28x28 biomedical sets; used for the chest X-ray and blood-cell microscopy benchmarks) · Deng, MNIST, IEEE Signal Processing Magazine 2012.
**Granularity:** SLIC (Achanta TPAMI 2012).
**Influence functions (VTF basis):** Koh & Liang, ICML 2017.
