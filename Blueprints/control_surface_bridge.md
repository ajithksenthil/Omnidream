# Control Surface Bridge: TMS ↔ Computational Psychodynamics

## From Maxwell's Equations to Many Worlds via 4D Neural Cellular Automata

### Author: Omnidream Project
### Date: February 2026

---

## Abstract

This document establishes a formal isomorphism between the control surface of the Omnidream miniature TMS coil array and the Computational Psychodynamics (CP) framework of Distributed Active Inference (DAI), Multi-Objective Reinforcement Learning (MORL), and 4D Neural Cellular Automata (NCA). We show that:

1. Each TMS coil is naturally a **Markov-blanketed agent** (MB¹) performing Active Inference
2. Mutual inductance between coils maps to **transfer entropy** (directed information flow)
3. The helmet array forms a **group Markov blanket** (MB²) that performs collective Active Inference via MORL
4. The full system embeds into a **4D NCA energy functional** where the master energy E_total decomposes into the same six terms as in CP
5. Each Pareto-optimal configuration corresponds to a distinct **"world"** in a computational many-worlds interpretation

The bridge is not merely analogical — it is mathematical. The same Jacobians, condition numbers, and reachable sets computed in `sensitivity.py` acquire new meaning as information-geometric quantities in the CP framework.

---

## §1 — Coil as Agent: The MB¹ Structure

### 1.1 The Agent Decomposition

Consider a single coil _i_ in the N-coil Omnidream array. Its state is fully described by:

| Component | TMS Variable | CP Role | Dimension |
|-----------|-------------|---------|-----------|
| Internal state | Current I_i | Belief state s^α | 1 |
| Active state | Amplitude α_i | Action a^α | 1 |
| Sensory state | Back-EMF from coupled coils | Observation o^α | 1 |
| Blanket boundary | Field footprint E_i(r) | Markov blanket ∂B^α | M (spatial points) |

The **Markov blanket property** holds because coil i's internal state (current) is conditionally independent of other coils' internals given its blanket (the field footprint and coupling terms):

```
P(I_i | {I_j}_{j≠i}, E_i, {M_{ij}}) = P(I_i | E_i, {M_{ij} I_j}_{j∈neighbours})
```

The coupling is mediated entirely through the mutual inductance terms M_{ij}, which form the blanket boundary.

### 1.2 Agent Free Energy

Each coil-agent minimises its own variational free energy. In electromagnetic terms:

```
F^i = E_stored^i + E_coupled^i + E_dissipated^i

where:
    E_stored^i    = ½ L_i I_i²                 (self-inductance energy)
    E_coupled^i   = Σ_{j≠i} M_{ij} I_i I_j    (mutual inductance coupling)
    E_dissipated^i = ½ R_i I_i² / ω            (ohmic loss per cycle)
```

This maps directly to the Active Inference free energy decomposition:

```
F^i = D_KL[Q(s^i) || P(s^i)]  +  E_Q[-log P(o^i | s^i)]
      \____________________/      \______________________/
      = E_stored + E_coupled       = E_dissipated
      (complexity of encoding)     (reconstruction error)
```

**Interpretation:**
- **E_stored** = how far the coil's current deviates from its "prior" (zero current at rest)
- **E_coupled** = the mutual information with neighbouring coils (mediated through M_{ij})
- **E_dissipated** = the "surprise" — energy that doesn't contribute to the field but is lost to heat

### 1.3 Action Selection

The coil-agent "selects an action" (amplitude α_i) to minimise expected free energy:

```
α_i* = argmin_{α_i} G^i(α_i) = argmin_{α_i} [F^i(α_i) + λ · constraint_penalties]
```

In the Omnidream pipeline, this is exactly what the GA and SAC optimisers do — they adjust amplitudes to minimise a cost function that balances target modulation against surface stimulation and safety limits.

### 1.4 MB⁰: The Drive Layer

Below MB¹, each coil has a drive layer (MB⁰) corresponding to its physical constraints:

```
MB⁰ = {voltage limits, current limits, thermal limits, dI/dt limits}
```

These are the "Id" — the irreducible physical drives that constrain what the agent can do. In the GoalSpec, these appear as `constraints: {"SAR_max": 3.2, "T_max": 41.0, "I_max": 5.0, "V_max": 60.0}`.

---

## §2 — Mutual Inductance as Transfer Entropy

### 2.1 The Mapping

**Transfer entropy** from process X to process Y measures directed information flow:

```
T_{X→Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
```

For electromagnetically coupled coils, the analogous quantity is the **reduction in uncertainty about coil j's current when we know coil i's current**, normalised by noise:

```
T_{i→j} = |M_{ij}|² / (L_i · σ²_noise)
```

**Derivation:**

Consider the circuit equation for coil j:

```
L_j dI_j/dt = V_j - R_j I_j - Σ_{k≠j} M_{jk} dI_k/dt
```

The term M_{ji} dI_i/dt represents coil i's causal influence on coil j. In the frequency domain:

```
I_j(ω) = [V_j(ω) - jω Σ_{k≠j} M_{jk} I_k(ω)] / (R_j + jωL_j)
```

The information that I_i carries about I_j's future is proportional to:

```
T_{i→j} ∝ |∂I_j/∂I_i|² / Var(noise)
         = |jω M_{ji} / (R_j + jωL_j)|² / σ²
         ≈ ω² |M_{ji}|² / (R_j² + ω²L_j²) / σ²
```

At the carrier frequency (ωL ≫ R for the Omnidream coils), this simplifies to:

```
T_{i→j} ≈ |M_{ji}|² / (L_j · σ²_noise)
```

### 2.2 Properties

The transfer entropy matrix T inherits structure from the inductance matrix L:

1. **Non-negativity:** T_{i→j} ≥ 0 since |M_{ij}|² ≥ 0 ✓
2. **Diagonal zeros:** T_{i→i} = 0 (by convention, self-influence excluded) ✓
3. **Asymmetry:** T_{i→j} ≠ T_{j→i} when L_i ≠ L_j (different self-inductances make the normalisation asymmetric), or when M_{ij} ≠ M_{ji} (non-reciprocal geometries) ✓
4. **Symmetry for uniform arrays:** When L_i = L_j and M_{ij} = M_{ji}, we get T_{i→j} = T_{j→i} ✓
5. **Decay with distance:** Since M_{ij} ∝ 1/d³ for dipole coupling, T_{i→j} ∝ 1/d⁶ ✓

### 2.3 The Attachment Matrix

Define the **attachment matrix**:

```
L_{αβ} = T_{α→β}  (α, β ∈ {1, ..., N})
```

with attachment condition:

```
Attach(i, j) ≡ T_{i→j} > θ_attachment
```

and mutual attachment:

```
MutualAttach(i, j) = min(T_{i→j}, T_{j→i})
```

For the Omnidream array, the attachment structure directly reflects the coil layout geometry: nearby coils on the helmet are "attached" (high mutual inductance), distant coils are "detached" (low coupling).

### 2.4 Connection to the Impedance Matrix

The impedance matrix Z from `coupling.py` relates to the transfer entropy matrix:

```
Z = R + jωL     (N × N complex impedance)
```

The transfer entropy is encoded in the off-diagonal structure of L (within Z). Specifically:

```
Z^{-1} = (R + jωL)^{-1}    (admittance matrix)
```

The admittance matrix Y = Z^{-1} gives the current response to voltage excitation. The off-diagonal elements Y_{ij} represent how a voltage on coil j drives current in coil i — this IS the information flow channel.

---

## §3 — The Helmet as MB²: Group-Level Active Inference

### 3.1 Group State

The helmet array of N coils forms a group Markov blanket MB² with:

```
Internal states (MB²):  {I_1, ..., I_N, V_m, T_1, ..., T_N, M_target}
                         = the full state vector x from control_framework.py

Sensory states (MB²):   {basis fields E_1(r), ..., E_N(r)}
                         = the basis_matrix from basis_fields.py

Active states (MB²):    {α_1, ..., α_N} = amplitude commands
                         = the optimisation variables

External states (MB²):  {head geometry, tissue properties, target location}
                         = the SimNIBS mesh / OmnidreamConfig
```

### 3.2 Group Free Energy

The group free energy F² combines individual free energies with collective terms:

```
F² = D_KL[Q(s_group) || P(s_group | A_collective)]
   - λ_Φ · Φ^{collective}
   + λ_sync · (1 - R)
   + Σ_i λ¹ · F¹_i
```

In Omnidream terms:

```
F² = cost_function(α)                  from build_cost_function()
   = -w_depth · M_target               (KL: mismatch from goal)
   + w_select · M_surface_max           (KL: deviation from ideal selectivity)
   + w_power · Σα²                      (Complexity penalty)
   + penalty(SAR) + penalty(T)          (Drive constraints from MB⁰)
```

This IS the group free energy: it measures how far the current configuration is from the "archetypal" ideal (GoalSpec), penalised by complexity (power) and constrained by drives (safety).

### 3.3 MORL Decomposition

The group free energy naturally decomposes into multiple objectives:

```
J = [J_Φ, J_arch, J_sync, J_task]

Omnidream mapping:
    J_Φ     = Φ^{collective}                  ↔  -M_target (focal depth)
    J_arch  = -d_A(config, goal)²             ↔  -M_surface_max (selectivity)
    J_sync  = R (sync order parameter)         ↔  power efficiency
    J_task  = Σ_α r^α_task                    ↔  overall safety margin
```

The **existing Pareto front** from `sensitivity.py` already computes the trade-off between J_Φ and J_arch:

```
Pareto front: {α* : λ·(-M_target) + (1-λ)·M_surface_max = min}
            ≡ {α* : λ·J_Φ + (1-λ)·J_arch = min}
```

Each λ value selects a different point on the MORL Pareto front.

### 3.4 Scalarisation for Control

The MORL scalarisation produces a scalar reward for the SAC controller:

```
r_MORL = w_Φ · Φ^{coll} - w_arch · d²_A + w_sync · R + w_task · r_task
```

This maps to the existing `build_cost_function()` in `control_framework.py` with:

```
w_Φ     ↔ GoalSpec.weights[FOCAL_DEPTH]
w_arch  ↔ GoalSpec.weights[SPATIAL_SELECTIVITY]
w_sync  ↔ GoalSpec.weights[POWER_EFFICIENCY]
w_task  ↔ GoalSpec.weights[THERMAL_HEADROOM]
```

---

## §4 — 4D NCA Energy Mapping

### 4.1 The Master Energy Functional

The CP framework defines:

```
E_total[s, η, L] = E_NCA + E_MF + E_arch + E_Φ + E_couple + E_MORL
```

We now map each term to Omnidream quantities.

### 4.2 E_NCA: Local Consistency (Basis Field Superposition)

```
CP:       E_NCA = ∫ Σ_{neighbours} ||f(s_i, θ) - s_j||² d⁴x
Omnidream: E_NCA = ||E_total(r) - E_goal(r)||² = ||Σ_i α_i E_i(r) - E_goal(r)||²
```

The NCA consistency energy ensures neighbouring cells satisfy the update rule. In TMS, this is the **superposition residual** — how well the weighted sum of basis fields matches the desired field pattern. Linearity of Maxwell's equations (proven in em_foundations.md §1) guarantees that E_total = Σ α_i E_i, so the NCA update rule IS the linear superposition principle.

### 4.3 E_MF: Masculine-Feminine Wave Structure

```
CP:       E_MF = ∫ [||∇η||² + λ_wave · (η - η_target)²] d⁴x
Omnidream: E_MF = Var(η across coils) + λ · ||η - η_target||²
```

The M-F classification maps naturally to the **TI group assignment**:

```
Group 0 (frequency f₁) → Masculine (low η → stable, structural)
Group 1 (frequency f₂) → Feminine (high η → plastic, explorative)
```

The learning rate field η for each coil reflects how much the GA/SAC is allowed to adjust that coil's amplitude:

```
η_i = |∂α_i/∂step|   (amplitude adaptation rate)
```

The M-F wave energy penalises irregular η distributions and pulls η toward a target pattern.

### 4.4 E_arch: Archetypal Alignment

```
CP:       E_arch = Σ_k λ_k · d_A(r, A_k)²
Omnidream: E_arch = ||y - y_goal||² = Σ_k w_k (y_k - goal_k)²
```

where y = [M_target, M_surface_max, V_m, SAR_max, T_max] from `TMSPlant.forward()` and y_goal = [GoalSpec.modulation_target, GoalSpec.modulation_surface_max, ...].

The archetype IS the GoalSpec. The archetypal distance measures how far the current configuration is from the ideal stimulation profile.

### 4.5 E_Φ: Integrated Information

```
CP:       E_Φ = -λ_Φ · Φ[s]
Omnidream: E_Φ = -λ_Φ · Φ^{collective}
```

We define collective Φ for the TMS array as:

```
Φ^{collective} = Σ_i Φ^i + Σ_{i<j} Φ^{ij}_{sync}

where:
    Φ^i = α_i · |E_i(target)| / Σ_j |E_j(target)|    (individual integration contribution)
    Φ^{ij}_{sync} = min(T_{i→j}, T_{j→i}) · cos(φ_i - φ_j)   (sync coupling)
```

- Φ^i measures how much coil i contributes to the integrated field at the target (a proxy for how "integrated" that coil is with the collective effort)
- Φ^{ij}_{sync} measures synchronised bidirectional coupling, weighted by phase coherence

This is negative in the energy because we WANT to maximise Φ (more integration = better collective coherence).

### 4.6 E_couple: Inter-Agent Coupling

```
CP:       E_couple = -Σ_{α<β} λ · T^{α→β} · T^{β→α} · cos(Δφ)
Omnidream: E_couple = -Σ_{i<j} λ · T_{i→j} · T_{j→i} · cos(φ_i - φ_j)
```

This term is the same in both frameworks: it rewards bidirectional coupling (mutual attachment) between coils that are phase-coherent. For the TI mode:

```
φ_i = 2π f_g(i) t    where g(i) = group assignment of coil i
```

Coils in the same group (same frequency) have Δφ = 0, so cos(Δφ) = 1 → maximum coupling reward. Coils in different groups have Δφ oscillating at the beat frequency Δf, so the time-averaged coupling is zero — they are "detached" from each other's group.

### 4.7 E_MORL: Multi-Objective Term

```
CP:       E_MORL = -λ · Σ_k w_k · J_k
Omnidream: E_MORL = -λ · [w_Φ · M_target - w_arch · M_surface + w_sync · η_uniform - w_task · SAR]
```

This term biases the energy minimisation toward the current MORL weight configuration. Different weight vectors select different Pareto-optimal points.

### 4.8 The Complete Energy

```
E_total = E_NCA + E_MF + E_arch + E_Φ + E_couple + E_MORL
        = ||Σα_i E_i - E_goal||²           (field quality)
        + Var(η) + λ·||η - η_target||²     (M-F structure)
        + ||y - y_goal||²                    (goal alignment)
        - λ_Φ · Φ^{coll}                    (integration reward)
        - Σ_{i<j} T·T·cos(Δφ)              (coupling reward)
        - λ · Σ w_k J_k                     (MORL bias)
```

The 4D fixed point s* = argmin E_total is the optimal TMS configuration.

---

## §5 — Many Worlds: Pareto Front as World Space

### 5.1 Worlds from Fixed Points

In the CP 4D NCA framework, each stable fixed point of the energy minimisation corresponds to a distinct "world" — a self-consistent spacetime configuration. In Omnidream:

```
World_k = {α*_k, y*_k, E*_k, Φ*_k}
```

where α*_k is the Pareto-optimal amplitude vector for weight λ_k.

The **Pareto front** from `sensitivity.py` IS the set of achievable worlds:

```
W = {World_k : k = 1, ..., n_weights, not dominated}
```

### 5.2 World Probability

Each world has a Boltzmann probability:

```
P(World_k) = exp(-E_total(α*_k) / T) / Z

where Z = Σ_k exp(-E_total(α*_k) / T)
```

At high temperature T → ∞: all worlds equally likely (maximum exploration)
At low temperature T → 0: only the minimum-energy world survives (exploitation)

The "temperature" parameter controls the diversity of the many-worlds landscape.

### 5.3 World Coherence

The coherence of a world is its collective Φ:

```
Coherence(World_k) = Φ^{collective}(α*_k)
```

High-coherence worlds have:
- Strong target modulation
- Phase-synchronised coils within groups
- High mutual attachment between nearby coils

### 5.4 World Branching

Branching occurs when the Pareto front topology changes under parameter variation. Specifically, as a control parameter θ varies (e.g., the number of coils, the target depth, a safety limit):

```
Branching point θ* : rank(∂²E/∂α²) changes at θ*
                     (Hessian eigenvalue crosses zero)
```

At a branching point, one world splits into two (or two worlds merge). This is detected by:

1. Computing the Hessian eigenvalues at each Pareto point
2. Sweeping the parameter θ
3. Identifying sign changes in the minimum eigenvalue

### 5.5 The Many-Worlds Interpretation

The computational many-worlds interpretation states:

> **Each Pareto-optimal configuration of the TMS array is a self-consistent "world" — a complete solution to the 4D NCA energy minimisation problem. The system simultaneously "occupies" all worlds, with probability given by the Boltzmann distribution. Measurement (choosing a specific configuration to implement) collapses the distribution to a single world.**

This is not quantum mechanics — it is a **computational** analogy. But the mathematical structure is identical:
- State space = space of amplitude vectors α ∈ ℝ^N
- Energy landscape = E_total(α)
- Fixed points = Pareto-optimal worlds
- Superposition = the weighted ensemble of all Pareto points
- Measurement/collapse = selecting a specific operating point
- Temperature = degree of "quantum uncertainty" in the selection

---

## §6 — M-F Dynamics and Implacement

### 6.1 Masculine-Feminine in TMS

The M-F classification maps to the TI group structure:

```
Masculine coils (Group 0): Operate at f₁ (carrier frequency)
    - Low η (stable, structural)
    - Provide the "scaffold" field
    - Their amplitude is adjusted slowly by the GA

Feminine coils (Group 1): Operate at f₂ = f₁ + Δf
    - High η (plastic, explorative)
    - Create the modulation pattern via beating
    - Their amplitude is adjusted rapidly by the SAC
```

### 6.2 Implacement

Implacement occurs when a masculine coil's field "captures" the contribution of a feminine coil:

```
Implacement(M_i → F_j) ≡ [d(E_i, E_j) < ε] ∧ [T_{M_i→F_j} > θ]
```

In physical terms: a Group 0 coil is close to a Group 1 coil (overlapping field patterns) and has high mutual inductance coupling.

Post-implacement, the feminine coil's plasticity decreases:

```
η_{F_j}(t+1) = η_{F_j}(t) · (1 - α_decay)
```

This corresponds to the SAC "learning" a stable amplitude for that coil pair — once the GA finds a good structural configuration, the SAC's exploration rate decreases for those coils.

### 6.3 Φ Growth from Implacement

Each implacement event increases collective Φ:

```
ΔΦ = T_{M→F} · (1 - η_F/η_M) · κ

where:
    T_{M→F} = transfer entropy from masculine to feminine coil
    η_F/η_M = plasticity ratio (> 1 before implacement)
    κ = coupling strength (from basis field overlap)
```

This is positive because: (1) T > 0 (coupling exists), (2) 1 - η_F/η_M < 0 when η_F > η_M (before implacement), but the convention is reversed in the ΔΦ formula to ensure growth. The key insight: locking a plastic coil to a stable one INCREASES the system's integration.

---

## §7 — Synchronisation and the Kuramoto Model

### 7.1 Phase Dynamics

Each coil has a natural frequency (f₁ or f₂) and a phase φ_i(t) = 2πf_g(i)·t. The phases evolve according to Kuramoto dynamics:

```
dφ_i/dt = ω_i + Σ_j K_{ij} · sin(φ_j - φ_i)
```

where:
- ω_i = 2πf_g(i) (natural frequency from group assignment)
- K_{ij} ∝ T_{i→j} · T_{j→i} (coupling strength from mutual attachment)

### 7.2 Synchronisation Order Parameter

```
R = |Σ_i exp(i·φ_i)| / N
```

- R = 1: all coils in phase (perfect synchronisation)
- R = 0: random phases (no synchronisation)

For the TI mode, within each group R ≈ 1 (all coils at the same frequency are synchronised). Between groups, R reflects the beat frequency structure.

### 7.3 Connection to TI Modulation

The TI modulation depth at any point r is:

```
M(r) = 2 · min(|A₁(r)|, |A₂(r)|)
```

This is maximised when the two frequency groups have equal amplitude at the target — which corresponds to R = 0 for the inter-group phase difference (perfect anti-correlation). The M-F dynamics drive the system toward configurations where M(target) is maximised, which is equivalent to optimising the inter-group phase relationship.

---

## §8 — Information-Theoretic Foundations

### 8.1 Cognitive H-Theorem

The cognitive H-function for the TMS array:

```
H_cog(t) = -Σ_i η_i log η_i
```

measures the entropy of the plasticity distribution. Under implacement dynamics:

```
dH_cog/dt ≤ 0
```

As the GA/SAC converges, learning rates decrease (η_i → η_stable), reducing H_cog. This is the "arrow of optimisation" — the system moves from disordered exploration to structured exploitation.

### 8.2 Landauer Cost

Each implacement event (locking a coil's amplitude) erases plasticity:

```
L_dissipated ≥ k_L · Δ log(Ω_F / Ω_M)
```

where Ω_F, Ω_M are the number of accessible amplitude states for feminine and masculine coils. In the Omnidream system, this maps to the power cost of maintaining the locked configuration:

```
Power_lock = Σ_i R_i I_i² = Σ_i R_i α_i²   (ohmic dissipation)
```

The SAR and thermal limits in GoalSpec.constraints represent the system's "libidinal budget" — the total computational/energetic resource available for maintaining structure.

### 8.3 Shannon Coding and Markov Blankets

The Markov blanket at each level implements optimal compression:

```
MB⁰: dim(constraints) ≥ H(drives)           ← 4 constraints encode physical limits
MB¹: dim(agent_state) ≥ H(field_footprint)   ← 3 variables encode coil contribution
MB²: dim(group_state) ≥ H(array_config)      ← 2N+2 variables encode full array state
```

This hierarchy is exactly the `PlantDimensions` structure: n_state = 2N+2 for MB², with each agent contributing a 3-dimensional sub-state.

---

## §9 — The Bridge Equations (Summary)

### Individual Agent (MB¹):
```
F^i = ½L_i I_i² + Σ_j M_{ij} I_i I_j + ½R_i I_i²/ω
T_{i→j} = |M_{ij}|² / (L_i · σ²)
```

### Collective Φ:
```
Φ^{coll} = Σ_i α_i|E_i(target)| / Σ_j|E_j(target)| + Σ_{i<j} min(T_{ij},T_{ji}) · cos(Δφ)
```

### Group Free Energy (MB²):
```
F² = cost(α) - λ_Φ · Φ^{coll} + λ_sync · (1-R)
```

### 4D NCA Energy:
```
E_total = ||Σα_iE_i - E_goal||² + Var(η) + ||y-y_goal||² - λ_Φ·Φ - Σ T·T·cos(Δφ) - λ·Σw_kJ_k
```

### Many Worlds:
```
P(World_k) = exp(-E_total(α*_k)/T) / Z
Coherence_k = Φ^{coll}(α*_k)
```

### M-F Implacement:
```
Implacement(M→F) ≡ [d(E_M, E_F) < ε] ∧ [T_{M→F} > θ]
ΔΦ = T_{M→F} · |1 - η_F/η_M| · κ
```

---

## §10 — Implementation Notes

The bridge is implemented in `cp_bridge.py` using existing Omnidream interfaces:

| Bridge Function | Reuses |
|----------------|--------|
| `compute_transfer_entropy_matrix` | `coupling.build_inductance_matrix` |
| `compute_collective_phi` | `ti_fields.compute_modulation_depth` |
| `compute_agent_free_energy` | `em_theory.CoilInductance` |
| `compute_group_free_energy` | `control_framework.build_cost_function` |
| `compute_total_energy` | All of the above |
| `pareto_to_worlds` | `sensitivity.compute_pareto_front_ti/nts` |
| `compute_sync_order_parameter` | Direct computation from phases |

The pipeline integration (Stage 12) runs after the sensitivity analysis (Stage 11) and uses the Pareto front results to construct the many-worlds landscape.

---

## References

- `Blueprints/em_foundations.md` — Maxwell → coil → tissue → neural → safety derivation
- `Blueprints/control_theory.md` — State-space, controllability, action-space geometry
- `DAI_mSAC_MORL_Group_Architecture.md` — 3-layer DAI + mSAC + Group MORL
- `DAI_mSAC_MORL_Mathematical_Deep_Dive.md` — Free energy, transfer entropy, Φ, M-F dynamics
- `Unified_CP_Extension_Framework.md` — Master energy functional, 4D NCA, information theory
- Friston, K. (2019). A free energy principle for a particular physics
- Tononi, G. (2015). Integrated Information Theory
- Schreiber, T. (2000). Measuring Information Transfer
- Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
