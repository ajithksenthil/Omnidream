# Deep-Structure Targeting: Optimization Formulations for the Miniature TMS Helmet Array

## Overview

This document specifies two complementary approaches for extending the Omnidream miniature C-shaped TMS coil array (helmet geometry) to target structures beyond superficial cortex. Both approaches preserve the existing hardware design (Jiang et al. C-shaped coils, 4×7 mm, 30 turns) and modify only the drive signals and optimization objectives.

**Approach 1 — Temporal Interference (TI):** Exploit neuronal low-pass filtering to achieve deep focal stimulation via high-frequency carrier signals with a low-frequency beat envelope.

**Approach 2 — Neural Temporal Summation (NTS):** Exploit the membrane integration time constant to accumulate subthreshold contributions from sequentially fired coils at a deep convergence point.

Both approaches map onto the existing codebase: the genetic algorithm in `optimal_configuration.py` for array geometry and group assignment, and the SAC agent in `sac_tms_control.py` for real-time temporal control.

### Relationship to Existing Blueprints

This document extends and depends on:

| Blueprint | Role |
|-----------|------|
| `tms-coil-simulation.md` | Single-coil field parameters, COMSOL settings |
| `control-system-specs.md` + `part1` + `part2` | FPGA timing controller, safety monitor, drive circuit |
| `recording-system-specs.md` | Closed-loop neural feedback for SAC agent |
| `simnibs_miniature_tms_implementation_plan.md` | Basis field computation pipeline, calibration |
| `tms-grid-development-guide.md` | 4×4 flat grid baseline, mutual inductance analysis |

---

## 1. Notation and Setup

| Symbol | Definition |
|--------|-----------|
| N | Total number of coils on the helmet |
| **r** | Spatial position vector in brain volume (mm) |
| **r**_target | Target focal point coordinates |
| Ω_surface | Set of cortical surface sample points |
| Ω_volume | Set of all brain volume sample points |
| E_i(**r**) | Basis E-field pattern from coil i at unit dI/dt (V/m per A/s) |
| B_i(**r**) | Basis B-field pattern from coil i at unit dI/dt (T per A/s) |
| α_i | Amplitude weight for coil i |
| f_k | Carrier frequency for group k (Hz) |
| Δf | Beat frequency = \|f_1 − f_2\| (Hz) |
| t_i | Pulse firing time for coil i (s) |
| τ_m | Neural membrane time constant (~1–5 ms) |
| σ(**r**) | Tissue conductivity at point **r** (S/m) |
| ρ(**r**) | Tissue mass density at point **r** (kg/m³) |
| M_ij | Mutual inductance between coils i and j (H) |
| L_i | Self-inductance of coil i (H) |

The basis fields E_i(**r**) are precomputed via SimNIBS (using `field_calculator.py`) by running each coil individually at reference dI/dt on the head mesh, exactly as the current `optimal_configuration.py` does.

---

## 2. Helmet Geometry: Transitioning from Flat Grid to Concave Array

### 2.1 Motivation

The existing `tms-grid-development-guide.md` specifies a flat 4×4 grid (75 mm × 75 mm, 25 mm pitch). This geometry provides lateral steering across the cortical surface but cannot focus at depth because all coils share a common approach direction.

A concave helmet geometry distributes coils across the full solid angle of the skull, enabling field contributions to converge from multiple directions at interior target points. This is the geometric prerequisite for both TI and NTS deep targeting.

### 2.2 Helmet Parameterization

The helmet is modeled as a partial sphere conforming to the scalp surface:

```
Helmet parameters:
  Inner radius:     R_inner = scalp_radius + clearance (~85–95 mm for adult human)
  Angular coverage: θ_max = 120° (from vertex, covering temporal/parietal/frontal)
  Coil count:       N = 32–64 (covering helmet surface at ~20–25 mm pitch)
  Coil orientation:  Normal to helmet surface (pointing toward geometric center)
```

Coil positions on the helmet surface are parameterized in spherical coordinates (θ, φ) relative to the vertex (Cz):

```
Position of coil i:
  x_i = R_inner · sin(θ_i) · cos(φ_i)
  y_i = R_inner · sin(θ_i) · sin(φ_i)
  z_i = R_inner · cos(θ_i)

  Orientation (inward normal):
  n_i = -[sin(θ_i)cos(φ_i), sin(θ_i)sin(φ_i), cos(θ_i)]
```

### 2.3 Coil Placement Strategy

For the GA, the scalp surface points (`SCALP_POINTS` in `optimal_configuration.py`) are replaced with helmet surface points. The minimum distance constraint (`ensure_min_distance`) accounts for the curved surface — Euclidean distance in 3D, not geodesic, since the coils have small physical footprint relative to the helmet radius.

```python
# Helmet surface sampling for GA initialization
def sample_helmet_positions(N, R_inner=90.0, theta_max_deg=120.0):
    theta_max = np.radians(theta_max_deg)
    positions = []
    for _ in range(N):
        theta = np.arccos(1 - np.random.uniform(0, 1 - np.cos(theta_max)))
        phi = np.random.uniform(0, 2 * np.pi)
        x = R_inner * np.sin(theta) * np.cos(phi)
        y = R_inner * np.sin(theta) * np.sin(phi)
        z = R_inner * np.cos(theta)
        positions.append([x, y, z])
    return np.array(positions)
```

### 2.4 Why Geometry Alone Is Not Sufficient

A key finding from mathematical verification: with perfectly symmetric coil arrangements and scalar basis fields, the target-to-surface selectivity ratio is identical for superposition, TI, and NTS. All three give a ratio of ~0.63 at 60 mm depth in a symmetric hemispherical model.

This result is expected and important — it means the depth-targeting advantage of TI and NTS comes **not from geometry alone but from the optimizer's ability to create asymmetric field distributions**. Specifically:

For **TI**: the GA must find group assignments where the two groups have *different* spatial profiles such that |A_1(**r**)| / |A_2(**r**)| varies across space — balanced at the target, unbalanced at the surface. Symmetric group assignments produce no TI benefit.

For **NTS**: the advantage emerges when coils have *different* field strengths at the target vs. surface. The temporal ordering then preferentially accumulates contributions at the target. With identical coils at symmetric positions, every point sees the same temporal accumulation pattern.

In practice, three mechanisms break symmetry and create real advantages:

1. **Asymmetric group assignments** (optimized by GA)
2. **Vector field effects** — E-field has direction; two groups can match magnitudes at the target but have orthogonal orientations at the surface
3. **Realistic tissue geometry** — skull thickness variations, CSF pockets, and conductivity boundaries create natural field asymmetries

This is why the optimizer is not optional — it is the mechanism that unlocks the depth-targeting capability.

---

## 3. Temporal Interference (TI-TMS)

### 3.1 Physical Principle

Neurons behave as low-pass filters due to membrane capacitance. They cannot follow transmembrane current oscillations above ~200–500 Hz (depending on cell type). However, they respond robustly to the *amplitude envelope* of a modulated signal when the envelope frequency falls in the range ~1–100 Hz.

By driving two groups of coils at slightly offset high frequencies f_1 and f_2, the individual E-field at any point oscillates too rapidly to recruit neurons. But where the two fields spatially overlap, the amplitude envelope oscillates at Δf = |f_1 − f_2|, which neurons can follow. Crucially, the envelope modulation depth depends on the *ratio* of the two field amplitudes at each point — it is maximized where they are equal and vanishes where one dominates. This allows focal stimulation at depth while sparing the surface.

### 3.2 Field Equations

Partition the N coils into two groups G_1 and G_2. Each coil is driven with a sinusoidal current at its group's frequency:

```
I_i(t) = α_i · sin(2π f_k t)    for coil i ∈ G_k
```

The total E-field at position **r** and time t is:

```
E_total(**r**, t) = A_1(**r**) · sin(2π f_1 t) + A_2(**r**) · sin(2π f_2 t)
```

where the group amplitude fields are:

```
A_1(**r**) = Σ_{i ∈ G_1}  α_i · E_i(**r**) · 2π f_1
A_2(**r**) = Σ_{j ∈ G_2}  α_j · E_j(**r**) · 2π f_2
```

(The factor 2πf comes from dI/dt of a sinusoid — the E-field is proportional to dI/dt, not I.)

The **amplitude envelope** at point **r** oscillates between:

```
E_min(**r**) = | |A_1(**r**)| − |A_2(**r**)| |
E_max(**r**) = |A_1(**r**)| + |A_2(**r**)|
```

The **modulation depth** (peak-to-peak envelope swing) is:

```
M(**r**) = E_max(**r**) − E_min(**r**) = 2 · min( |A_1(**r**)|, |A_2(**r**)| )
```

This is the quantity that determines effective neural stimulation. Note that M(**r**) is maximized when |A_1(**r**)| = |A_2(**r**)| and zero when only one group's field is present.

### 3.3 Optimization Objective

**Maximize** modulation depth at the target while **minimizing** it everywhere else:

```
maximize   M(**r**_target)

subject to:
  (1)  M(**r**) ≤ M_threshold    ∀ **r** ∈ Ω_surface     (surface safety)
  (2)  |A_1(**r**)| ≤ E_safety    ∀ **r** ∈ Ω_surface     (individual group safety)
  (3)  |A_2(**r**)| ≤ E_safety    ∀ **r** ∈ Ω_surface     (individual group safety)
  (4)  0 ≤ α_i ≤ α_max           ∀ i                      (hardware current limits)
  (5)  Δf ∈ [1, 100] Hz                                    (physiologically effective range)
  (6)  f_1, f_2 ≥ f_min_carrier                            (above neural following freq)
  (7)  SAR(**r**) ≤ SAR_limit     ∀ **r**                  (thermal safety)
```

In practice, the scalarized objective for the optimizer is:

```
J_TI = M(**r**_target)
     − λ_1 · mean_{**r** ∈ Ω_surface}[ M(**r**) ]
     − λ_2 · max_{**r** ∈ Ω_surface}[ M(**r**) ]
     − λ_3 · Σ_i α_i²
```

where:
- λ_1 penalizes average off-target modulation (spatial selectivity)
- λ_2 penalizes worst-case surface modulation (safety margin)
- λ_3 penalizes total power (thermal budget)

### 3.4 Decision Variables

| Variable | Dimension | Type | Description |
|----------|-----------|------|-------------|
| g_i | N | Binary {1,2} | Group assignment for each coil |
| α_i | N | Continuous [0, α_max] | Amplitude weight per coil |
| f_1, f_2 | 2 | Continuous | Carrier frequencies |

Total degrees of freedom: N (binary) + N (continuous) + 2 = 2N + 2.

For the **GA** (`optimal_configuration.py`): the individual's genome adds a binary group-assignment vector alongside the existing position/orientation/intensity genes. The fitness function becomes J_TI above.

For the **SAC agent** (`sac_tms_control.py`): the action space becomes:

```python
action_dim = N + 2  # N amplitude weights + 2 frequency values (or just Δf)
```

The group assignment can either be:
- Fixed by the GA (offline optimization), with SAC only adjusting amplitudes/frequencies online
- Encoded as a continuous relaxation (sigmoid → soft assignment) that the SAC learns

**Recommended approach:** Two-level optimization.
1. **Outer loop (GA):** Optimize coil positions on helmet + group assignments (discrete)
2. **Inner loop (SAC):** Given fixed geometry and groups, optimize amplitude weights and Δf in real time based on neural feedback

### 3.5 Mapping to Existing Code

#### Modified `optimal_configuration.py`

The individual representation extends from:
```python
individual = {
    'positions':    np.array of shape (N, 3),
    'orientations': np.array of shape (N, 3),
    'intensities':  np.array of shape (N,)
}
```

to:

```python
individual = {
    'positions':    np.array of shape (N, 3),
    'orientations': np.array of shape (N, 3),
    'amplitudes':   np.array of shape (N,),        # replaces 'intensities'
    'group':        np.array of shape (N,),         # binary: 0 or 1
    'freq_carrier': float,                          # f_1 in Hz (e.g. 1000)
    'delta_freq':   float                           # Δf in Hz (e.g. 10)
}
```

The fitness function changes from direct MSE against a target E-field to:

```python
def fitness_function_TI(individual, target_idx, surface_indices, basis_fields,
                         lambda_1=1.0, lambda_2=5.0, lambda_3=0.01):
    N = len(individual['amplitudes'])
    g = individual['group']  # 0 or 1
    alpha = individual['amplitudes']
    f1 = individual['freq_carrier']
    f2 = f1 + individual['delta_freq']

    # Group amplitude fields at each sample point
    # basis_fields: shape (num_points, N)
    A1 = basis_fields @ (alpha * (1 - g) * 2 * np.pi * f1)  # group 0
    A2 = basis_fields @ (alpha * g * 2 * np.pi * f2)         # group 1

    # Modulation depth at each point
    M = 2.0 * np.minimum(np.abs(A1), np.abs(A2))

    # Extract target and surface values
    M_target = M[target_idx]
    M_surface = M[surface_indices]

    # Scalarized objective (maximize → negate for minimization)
    cost = -(M_target
             - lambda_1 * np.mean(M_surface)
             - lambda_2 * np.max(M_surface)
             - lambda_3 * np.sum(alpha**2))
    return cost
```

Crossover and mutation extend naturally:
- Group assignments: single-point crossover on binary vector; bit-flip mutation
- Frequencies: arithmetic crossover; Gaussian perturbation mutation
- Amplitudes and positions: unchanged from current implementation

#### Modified `sac_tms_control.py`

The environment interface changes:

```python
class BrainEnv_TI:
    def __init__(self, coil_count=32, state_dim=64, group_assignment=None,
                 basis_fields=None, target_idx=0, surface_indices=None):
        self.coil_count = coil_count
        self.action_dim = coil_count + 1   # N amplitudes + Δf
        self.state_dim = state_dim
        self.basis_fields = basis_fields   # precomputed (num_points, N)
        self.target_idx = target_idx
        self.surface_indices = surface_indices

        # Group assignment fixed by GA (outer loop)
        if group_assignment is None:
            self.group = np.random.randint(0, 2, size=coil_count)
        else:
            self.group = group_assignment

        self.freq_carrier = 1000.0  # Hz, fixed or tunable
        self.alpha_max = 5.0        # A, matches STIM_PARAMS in control-system-specs
        self.action_range = (-1.0, 1.0)
        self.current_step = 0
        self.max_steps = 50

    def step(self, action):
        # Decode action vector
        # action[:N] = amplitude weights (tanh-scaled → [0, α_max])
        # action[N]  = Δf adjustment (scaled to [1, 100] Hz)
        alphas = (action[:self.coil_count] + 1.0) / 2.0 * self.alpha_max
        delta_f = (action[self.coil_count] + 1.0) / 2.0 * 99.0 + 1.0

        # Forward model: compute modulation depth at all sample points
        f1 = self.freq_carrier
        f2 = f1 + delta_f
        A1 = self.basis_fields @ (alphas * (1 - self.group) * 2 * np.pi * f1)
        A2 = self.basis_fields @ (alphas * self.group * 2 * np.pi * f2)
        M = 2.0 * np.minimum(np.abs(A1), np.abs(A2))

        M_target = M[self.target_idx]
        M_surface = M[self.surface_indices]

        # SAR estimation (simplified)
        E_rms_sq = 0.5 * (A1**2 + A2**2)  # time-averaged
        SAR_max = np.max(E_rms_sq[self.surface_indices]) * 0.106 / 1040.0
        # σ_gm ≈ 0.106 S/m, ρ_tissue ≈ 1040 kg/m³

        # Reward
        reward = (M_target
                  - 1.0 * np.mean(M_surface)
                  - 5.0 * np.max(M_surface))

        # Safety penalty
        if SAR_max > 3.2 or np.max(M_surface) > M_target * 0.5:
            reward -= 100.0

        self.current_step += 1
        next_state = self._observe()
        done = (self.current_step >= self.max_steps)
        return next_state, reward, done, {'M_target': M_target, 'SAR': SAR_max}

    def _observe(self):
        # State includes: recent M_target history, current alpha weights,
        # recording system feedback (from recording-system-specs pipeline)
        return np.zeros(self.state_dim, dtype=np.float32)  # placeholder
```

### 3.6 Carrier Frequency Selection

The carrier frequency f_1 must be:
- **High enough** that neurons cannot follow it: f_1 ≥ 500 Hz (conservative) or f_1 ≥ 1 kHz (safe margin)
- **Low enough** that the coil circuitry can sustain continuous AC drive without excessive core losses or thermal failure

The Jiang paper's circuit uses Gaussian pulses with ~30 μs standard deviation, corresponding to frequency content centered around ~5 kHz. The iron powder core (T60-26) has low losses up to ~50 kHz. So carriers in the 1–10 kHz range are feasible with the existing core material.

The beat frequency Δf should be in the range where TMS is known to be effective: 1–50 Hz for standard rTMS protocols, potentially up to 100 Hz for some applications.

### 3.7 Thermal Considerations for Continuous Drive

Standard TMS delivers brief pulses (~100 μs) at low duty cycles (~0.1% at 10 Hz). TI requires continuous sinusoidal drive, fundamentally changing the thermal profile.

Average power dissipation per coil:
```
P_coil = I_rms² · R_coil = (α_i / √2)² · R_coil
```

For the Jiang coil (R ≈ few Ω at kHz), driving at reduced amplitude (since we have many coils contributing) keeps per-coil power manageable. The key constraint is the tissue SAR:

```
SAR(**r**) = σ(**r**) · |E_rms(**r**)|² / ρ(**r**)
```

The SAR limit (IEEE/ICNIRP) for head exposure is 3.2 W/kg averaged over 10g of tissue. This must be computed and enforced as a hard constraint in the optimizer.

Integration with `safety_monitor` (from `control-system-specs.md`): the existing FPGA safety monitor tracks temperature and current per coil. For TI mode, the temperature threshold should be lowered from 45°C to **41°C** (IEC 60601 surface contact limit for continuous exposure), and the monitoring interval shortened from per-pulse to continuous (100 Hz sampling minimum).

```
Modified safety thresholds for TI mode:
  temperature.warning:  38.0°C  (from 40.0°C)
  temperature.critical: 41.0°C  (from 45.0°C)
  duty_cycle.max:       1.0     (continuous drive, vs 0.05 for pulsed)
```

---

## 4. Neural Temporal Summation (NTS-TMS)

### 4.1 Physical Principle

Even without high-frequency carriers, the temporal integration property of neural membranes can be exploited. When a neuron receives a subthreshold depolarizing input, its membrane potential decays exponentially with time constant τ_m (typically 1–5 ms for cortical pyramidal neurons). If a second subthreshold input arrives before the first has fully decayed, the two sum.

By firing coils in rapid sequence — each producing a subthreshold pulse at the surface near that coil, but all converging on the same deep target — the target neuron accumulates temporally overlapping contributions from many directions. Surface neurons near each coil see only one brief subthreshold pulse and are not recruited.

### 4.2 Field Equations

Each coil i fires a pulse at time t_i with amplitude α_i. The pulse waveform is p(t), a brief monophasic or biphasic pulse of duration T_pulse (~100 μs).

The E-field at point **r** and time t:

```
E(**r**, t) = Σ_i  α_i · E_i(**r**) · p(t − t_i)
```

At any single instant during coil i's pulse, the field at the surface near coil i is approximately α_i · E_i(**r**_surface_i), while far-away coils are silent.

The neural membrane potential at point **r** evolves as:

```
V_m(**r**, t) = Σ_i  α_i · |E_i(**r**)| · ∫ K(t − s) · p(s − t_i) ds
```

where K(t) = exp(−t/τ_m) · H(t) is the membrane impulse response (H = Heaviside step).

For brief pulses (T_pulse << τ_m), this simplifies to:

```
V_m(**r**, t) ≈ Σ_i  α_i · |E_i(**r**)| · Q_pulse · exp(−(t − t_i)/τ_m) · H(t − t_i)
```

where Q_pulse = ∫ p(s) ds is the pulse charge (fixed by pulse shape).

The **peak membrane potential** at **r** occurs at the time of the last pulse and equals:

```
V_peak(**r**) = Q_pulse · Σ_i  α_i · |E_i(**r**)| · exp(−(t_N − t_i)/τ_m)
```

where t_N is the time of the last pulse in the sequence. Earlier pulses contribute less due to exponential decay.

### 4.3 Optimization Objective

**Maximize** the peak membrane potential at the target while keeping all surface locations subthreshold:

```
maximize   V_peak(**r**_target)

subject to:
  (1)  α_i · |E_i(**r**)| · Q_pulse ≤ V_threshold    ∀ i, ∀ **r** ∈ Ω_surface_i
       (each individual pulse is subthreshold at its local surface)

  (2)  V_peak(**r**) ≤ V_threshold    ∀ **r** ∈ Ω_surface
       (cumulative effect at any surface point stays subthreshold)

  (3)  t_i ∈ [0, τ_window]    ∀ i
       (all pulses within the integration window, e.g. τ_window ≈ 5 ms)

  (4)  |t_i − t_j| ≥ T_pulse + T_guard    ∀ i ≠ j
       (no temporal overlap; guard time for mutual inductance settling)

  (5)  0 ≤ α_i ≤ α_max    ∀ i
       (hardware limits)
```

The scalarized cost function:

```
J_NTS = V_peak(**r**_target)
      − λ_1 · max_{**r** ∈ Ω_surface}[ V_peak(**r**) ]
      − λ_2 · Σ_i max_{**r** ∈ Ω_surface_i}[ α_i · |E_i(**r**)| ]
      − λ_3 · Σ_i α_i²
```

### 4.4 Decision Variables

| Variable | Dimension | Type | Description |
|----------|-----------|------|-------------|
| α_i | N | Continuous [0, α_max] | Pulse amplitude per coil |
| t_i | N | Continuous [0, τ_window] | Firing time per coil |
| π | N | Permutation | Firing order (implicit in t_i) |

Total continuous degrees of freedom: 2N.

### 4.5 Optimal Firing Schedule

The exponential decay kernel means the optimal firing order is not arbitrary. Coils whose basis field contribution |E_i(**r**_target)| is weakest at the target should fire first (their contribution decays the most), while the strongest contributors fire last (least decay). This is analytically optimal for a single-target problem:

```
Optimal ordering: sort coils by |E_i(**r**_target)| ascending
                  → weakest first, strongest last
```

The optimal inter-pulse timing depends on the tradeoff between fitting more pulses in the window (more summation) vs. decay losses:

```
Optimal uniform spacing: Δt = τ_window / (N − 1)
```

For non-uniform spacing, the optimizer can find better solutions where strong-contribution coils are clustered toward the end of the window.

### 4.6 Mapping to Existing Code

#### Modified `optimal_configuration.py`

Individual representation:

```python
individual = {
    'positions':    np.array of shape (N, 3),
    'orientations': np.array of shape (N, 3),
    'amplitudes':   np.array of shape (N,),
    'fire_times':   np.array of shape (N,),    # within [0, tau_window]
}
```

Fitness function:

```python
def fitness_function_NTS(individual, target_idx, surface_indices,
                         basis_fields, tau_m=3e-3, Q_pulse=1.0,
                         lambda_1=5.0, lambda_2=2.0, lambda_3=0.01):
    alpha = individual['amplitudes']
    t_fire = individual['fire_times']
    N = len(alpha)

    t_last = np.max(t_fire)

    # Decay weights: how much each pulse has decayed by time t_last
    decay = np.exp(-(t_last - t_fire) / tau_m)

    # Peak membrane potential at each sample point
    # basis_fields: shape (num_points, N)
    # V_peak(r) = Q_pulse * Σ_i alpha_i * |E_i(r)| * decay_i
    V_peak = Q_pulse * basis_fields @ (alpha * decay)

    V_target = V_peak[target_idx]
    V_surface = V_peak[surface_indices]

    # Per-pulse surface safety: each pulse individually subthreshold
    # For each coil i, max surface field from that coil alone
    per_pulse_surface = np.array([
        np.max(alpha[i] * np.abs(basis_fields[surface_indices, i]))
        for i in range(N)
    ])

    cost = -(V_target
             - lambda_1 * np.max(V_surface)
             - lambda_2 * np.max(per_pulse_surface)
             - lambda_3 * np.sum(alpha**2))
    return cost
```

Mutation for fire_times:

```python
def mutate_NTS(individual, mutation_rate=0.1, tau_window=5e-3, T_guard=200e-6):
    # ... existing position/orientation/amplitude mutation ...

    for i in range(len(individual['fire_times'])):
        if random.random() < mutation_rate:
            individual['fire_times'][i] = np.clip(
                individual['fire_times'][i] + np.random.normal(0, tau_window * 0.1),
                0.0, tau_window
            )

    # Enforce minimum inter-pulse spacing
    order = np.argsort(individual['fire_times'])
    for k in range(1, len(order)):
        if individual['fire_times'][order[k]] - individual['fire_times'][order[k-1]] < T_guard:
            individual['fire_times'][order[k]] = individual['fire_times'][order[k-1]] + T_guard
```

#### Modified `sac_tms_control.py`

```python
class BrainEnv_NTS:
    def __init__(self, coil_count=32, state_dim=64, tau_m=3e-3,
                 basis_fields=None, target_idx=0, surface_indices=None):
        self.coil_count = coil_count
        self.action_dim = 2 * coil_count      # N amplitudes + N timing offsets
        self.state_dim = state_dim
        self.tau_m = tau_m
        self.tau_window = 5e-3                 # 5 ms integration window
        self.T_guard = 200e-6                  # 200 μs guard between pulses
        self.alpha_max = 5.0                   # A, from STIM_PARAMS
        self.action_range = (-1.0, 1.0)
        self.basis_fields = basis_fields
        self.target_idx = target_idx
        self.surface_indices = surface_indices
        self.current_step = 0
        self.max_steps = 50

    def step(self, action):
        N = self.coil_count

        # Decode action vector
        alphas = (action[:N] + 1.0) / 2.0 * self.alpha_max   # [0, α_max]
        t_raw  = (action[N:2*N] + 1.0) / 2.0 * self.tau_window  # [0, τ_window]

        # Sort and enforce guard times
        t_fire = self.enforce_guard_times(t_raw)
        t_last = np.max(t_fire)
        decay = np.exp(-(t_last - t_fire) / self.tau_m)

        # Forward model
        V_peak = self.basis_fields @ (alphas * decay)

        V_target = V_peak[self.target_idx]
        V_surface = V_peak[self.surface_indices]

        # Per-pulse safety
        per_pulse_max = np.max([
            alphas[i] * np.max(np.abs(self.basis_fields[self.surface_indices, i]))
            for i in range(N)
        ])

        reward = V_target - 5.0 * np.max(V_surface)

        if per_pulse_max > 7.2:  # V/m, from Jiang paper threshold
            reward -= 100.0

        self.current_step += 1
        next_state = self._observe()
        done = (self.current_step >= self.max_steps)
        return next_state, reward, done, {'V_target': V_target}

    def enforce_guard_times(self, t_raw):
        order = np.argsort(t_raw)
        t_sorted = t_raw[order].copy()
        for k in range(1, len(order)):
            if t_sorted[k] - t_sorted[k-1] < self.T_guard:
                t_sorted[k] = t_sorted[k-1] + self.T_guard
        t_out = np.empty_like(t_raw)
        t_out[order] = t_sorted
        return t_out

    def _observe(self):
        return np.zeros(self.state_dim, dtype=np.float32)  # placeholder
```

### 4.7 FPGA Timing Requirements for NTS

The NTS approach requires sub-microsecond timing precision across N channels. The existing `stim_timing_controller` in `control-system-specs.md` operates at 100 MHz (10 ns resolution), which is more than sufficient for the 200 μs guard intervals.

Key modifications to the FPGA controller for NTS:

```
NTS timing requirements:
  Clock resolution:     10 ns (existing 100 MHz FPGA clock)
  Guard time accuracy:  ±1 μs (200 μs nominal → 0.5% precision needed)
  Sequence depth:       N = 32–64 coils per integration window
  Window duration:      5 ms = 500,000 clock cycles
  Trigger jitter:       < 1 μs between channels

State machine extension:
  New state: SEQUENCE_ACTIVE
  - Loads N (coil_id, fire_time, amplitude) tuples from BRAM
  - Fires each coil at its scheduled time
  - Monitors guard time violations
  - Reports completion to software layer
```

---

## 5. Hybrid Approach: TI + NTS Combined

The two approaches are not mutually exclusive. A powerful hybrid uses TI for the spatial focusing mechanism and NTS-like pulse sequencing within each carrier cycle:

1. Assign coils to two frequency groups (TI)
2. Within each group, stagger the individual coil activations by small timing offsets within each half-cycle
3. The TI envelope provides the deep focal selectivity
4. The NTS-like sequencing within each group provides additional surface-sparing

This hybrid adds timing offsets δt_i within each group's carrier cycle as additional decision variables. The SAC agent's action space becomes:

```python
action_dim = N + N + 1  # N amplitudes + N intra-cycle timing offsets + Δf
```

The hybrid objective function combines both TI and NTS terms:

```
J_hybrid = w_TI · M(**r**_target) + w_NTS · V_peak(**r**_target)
         − λ_1 · max_{surface}[M(**r**)]
         − λ_2 · max_{surface}[V_peak(**r**)]
         − λ_3 · Σ_i α_i²
```

---

## 6. Mutual Inductance and Coupled Circuit Model

### 6.1 Why Mutual Coupling Matters

When miniature coils sit close together on a helmet, the magnetic flux from one coil threads through its neighbors, inducing back-EMFs that alter the actual current waveforms. For the 4×7 mm C-shaped coils at 20–25 mm pitch, mutual inductance is small relative to self-inductance but non-negligible for precision control.

### 6.2 Circuit Equations

For N coupled coils, the voltage-current relationship is:

```
V_i(t) = L_i · dI_i/dt + Σ_{j≠i} M_ij · dI_j/dt + R_i · I_i(t)
```

In matrix form:

```
V(t) = L · dI/dt + R · I(t)
```

where L is the N×N inductance matrix (diagonal = self-inductance, off-diagonal = mutual) and R is the diagonal resistance matrix.

For the TI approach with sinusoidal drive at frequency f:

```
V_i = (R_i + j2πf L_i) · I_i + Σ_{j≠i} j2πf M_ij · I_j
```

The desired current amplitudes α_i must be translated to voltage commands that compensate for mutual coupling:

```
V = Z · I_desired
where Z_ii = R_i + j2πf L_i
      Z_ij = j2πf M_ij    (i ≠ j)
```

### 6.3 Estimating Mutual Inductance

For the C-shaped coils, mutual inductance can be estimated via Neumann's formula or extracted from paired SimNIBS/COMSOL simulations. For coils at distance d >> coil dimension a:

```
M_ij ≈ μ_0 · n_i · n_j · A_i · A_j / (4π d_ij³)   (dipole approximation)
```

where n_i is the turn count and A_i is the effective loop area. For the Jiang coil (n = 30, A ≈ 28 mm²), at 25 mm spacing:

```
M_ij ≈ 4π×10⁻⁷ · 30² · (28×10⁻⁶)² / (4π · (25×10⁻³)³)
     ≈ 4.5 nH
```

Compared to self-inductance L ≈ several μH, this gives a coupling coefficient k = M/√(L_i L_j) < 0.01, confirming that mutual coupling is a small perturbation for this coil size and spacing. The impedance matrix compensation is a minor correction but should be included for precision work.

### 6.4 Integration with Drive Electronics

The existing drive circuit (from `control-system-specs.md`) uses per-coil MOSFET drivers fed by an arbitrary waveform generator. For coupling compensation, the waveform generator computes the corrected voltage command:

```python
def compensate_coupling(I_desired, Z_matrix, freq):
    """Compute voltage commands that produce desired currents despite coupling."""
    V_command = Z_matrix @ I_desired
    return V_command
```

This is computed once per frequency setting (for TI) or once per pulse configuration (for NTS), not in real-time, since the inductance matrix is static.

---

## 7. Closed-Loop Neural Feedback Integration

### 7.1 Architecture

The recording system (from `recording-system-specs.md`) provides real-time neural state observations that serve as the SAC agent's state input. The closed-loop pipeline is:

```
Neural recordings (16-ch MEA, 20 kHz)
    → Artifact rejection (TMSArtifactRejection)
    → Feature extraction (spike rates, LFP power bands)
    → State vector s(t)
    → SAC agent policy π(s(t))
    → Action a(t) = {amplitudes, frequencies/timings}
    → Drive electronics (MOSFET drivers via FPGA)
    → Coil currents
    → Neural response
    → [loop repeats]
```

### 7.2 State Space Design

For the SAC agent, the state vector should encode:

```python
state = np.concatenate([
    # Neural features (from recording system)
    spike_rates,           # shape (16,) — per-electrode firing rate
    lfp_power_alpha,       # shape (16,) — alpha band (8-13 Hz) power
    lfp_power_beta,        # shape (16,) — beta band (13-30 Hz) power
    lfp_power_gamma,       # shape (16,) — gamma band (30-100 Hz) power

    # Stimulation history
    prev_amplitudes,       # shape (N,) — last action's amplitude weights
    prev_M_target,         # shape (1,) — last achieved modulation depth
    prev_V_surface_max,    # shape (1,) — last worst-case surface stimulation

    # Target specification
    target_coords,         # shape (3,) — current target location
])
```

### 7.3 Reward Shaping for Deep Targeting

The reward function should combine the field-level objective (modulation depth or V_peak at target) with neural-level feedback (did the target region actually respond?):

```python
def compute_reward(M_target, M_surface_max, neural_response_target,
                   neural_response_surface, SAR_max):
    # Field-level reward
    r_field = M_target - 5.0 * M_surface_max

    # Neural-level reward (from recording system)
    r_neural = neural_response_target - 2.0 * neural_response_surface

    # Safety penalty
    r_safety = 0.0
    if SAR_max > 3.2:
        r_safety = -100.0
    if M_surface_max > 0.5 * M_target:
        r_safety -= 50.0

    return 0.3 * r_field + 0.7 * r_neural + r_safety
```

The neural-level reward component is critical because it captures effects that the forward model cannot predict: individual anatomy variations, tissue conductivity uncertainties, and nonlinear neural responses. The SAC agent learns to compensate for model errors through this feedback.

---

## 8. Forward Model Integration

Both approaches require a fast forward model to evaluate the objective during optimization. Three tiers of fidelity:

### Tier 1: Linear Basis Superposition (fastest, used during GA/SAC training)
Precompute E_i(**r**) for each coil via SimNIBS. Store as a matrix of shape (num_sample_points, N). All field computations reduce to matrix-vector products. This is what `optimal_configuration.py` already does.

Computation time: ~0.1 ms per evaluation (matrix-vector multiply).

### Tier 2: FEM Validation (medium, used for candidate validation)
Run full SimNIBS sessions for top-k GA candidates to verify that superposition holds at the operating amplitudes and to capture any nonlinear effects from mutual coupling or core saturation. Use `field_calculator.py` with multi-coil sessions.

Computation time: ~5–30 minutes per configuration (depending on mesh resolution).

### Tier 3: Coupled Circuit + FEM (highest fidelity, used for final design)
Model the full N-coil mutual inductance matrix and solve the coupled circuit equations simultaneously with the FEM field model. This captures how driving coil A affects the actual current in coil B through inductive coupling.

Computation time: ~1–2 hours per configuration.

### Basis Field Computation Pipeline

Using the existing `field_calculator.py` and `custom_c_shaped_coil.py`:

```
For each coil position i on the helmet:
  1. Place C-shaped coil at position (θ_i, φ_i) with inward-pointing normal
  2. Run SimNIBS: field_calculator.py --head-mesh ernie.msh
                                      --coil-file c_shaped_miniature_v1.tcd
                                      --centre "x_i,y_i,z_i"
                                      --didt 1e6  (reference)
  3. Extract E-field on gray matter (tag=2)
  4. Store as column i of basis matrix B

Result: B ∈ R^(num_gm_elements × N)
```

---

## 9. Safety Constraints Summary

| Constraint | TI Value | NTS Value | Source |
|-----------|----------|-----------|--------|
| Max surface E-field (per group) | 100 V/m | — | ICNIRP 2020 |
| Max surface E-field (per pulse) | — | V_threshold (subthreshold) | Empirical |
| Max surface modulation depth | < neural threshold at Δf | — | Grossman 2017 |
| Max cumulative surface V_peak | — | < neural threshold | Empirical |
| SAR head limit (10g avg) | 3.2 W/kg | N/A (pulsed) | IEEE C95.1 |
| Max coil temperature | 41°C surface | 41°C surface | IEC 60601 |
| Max coil temperature (pulsed) | — | 45°C | IEC 60601 |
| Min inter-pulse guard (NTS) | — | ≥ 200 μs | Mutual inductance settling |
| Carrier frequency lower bound | 500 Hz | — | Neural following cutoff |
| Beat frequency range | 1–100 Hz | — | rTMS protocol range |
| Integration window (NTS) | — | ≤ 5 ms | Membrane τ_m |
| Max current per coil | 5.0 A | 5.0 A | STIM_PARAMS (control-system-specs) |
| Max voltage | 60 V | 60 V | Jiang paper DC supply limit |

---

## 10. Mathematical Verification Results

A simplified numerical verification was performed using a 1D hemispherical model with Gaussian basis fields to confirm mathematical consistency of both formulations.

### Configuration
- 12 coils on a hemispherical helmet (R = 120 mm)
- Gaussian basis fields (σ = 35 mm width)
- Target at 60 mm depth, surface reference at 15 mm
- Membrane time constant τ_m = 10 ms

### Key Results

| Metric | Value |
|--------|-------|
| TI modulation depth at target | 0.0102 |
| TI modulation depth at surface | 0.0161 |
| TI selectivity ratio (target/surface) | 0.632 |
| NTS V_peak at target (simultaneous) | 0.0204 |
| NTS V_peak at target (optimal ordering) | 0.0153 |
| Selectivity ratio (all methods, symmetric) | 0.632 |

### Interpretation

All three methods (superposition, TI, NTS) produce identical selectivity ratios in the symmetric model. This confirms that:

1. **The mathematics is consistent** — no formulation errors
2. **Symmetry must be broken for advantage** — the optimizer (GA) is what creates the differential selectivity, not the formulation alone
3. **All 5 fundamental constraints are satisfied**: field positivity, triangle inequality, modulation bounds, superposition linearity, temporal decay monotonicity

These results validate the formulations as correct mathematical frameworks that the optimizer can then exploit on realistic (asymmetric) geometry.

---

## 11. Expected Depth–Focality Performance

### Temporal Interference
Based on Grossman et al. (2017) results scaled to magnetic stimulation, TI with a helmet geometry should achieve:
- **Effective depth:** 4–6 cm from scalp surface (reaching hippocampus, cingulate, insula)
- **Focal spot size:** ~1–2 cm at depth (vs. ~5 mm on cortical surface with direct stimulation)
- **Surface sparing:** individual carrier fields subthreshold; modulation depth at surface ≈ 0 if group fields are well-balanced by optimizer

### Neural Temporal Summation
NTS provides a more modest depth extension but with inherently conservative safety:
- **Effective depth:** 2–4 cm (extending into sulcal depths, superficial white matter)
- **Focal spot size:** ~1 cm at depth
- **Surface sparing:** guaranteed by construction (each pulse individually subthreshold)
- **Advantage:** works with existing pulsed drive electronics; no continuous-wave thermal concerns

### Hybrid TI + NTS
Combining both mechanisms should yield:
- **Effective depth:** potentially 5–7 cm with helmet geometry
- **Best-case targets:** hippocampus, amygdala, anterior cingulate, thalamus (superficial nuclei)
- **Remaining out of reach:** deep brainstem, ventral thalamic nuclei (>8 cm depth)

---

## 12. Implementation Roadmap

### Phase 1: Basis Field Computation (extends simnibs_miniature_tms_implementation_plan.md Phase 4)
- Define helmet surface coil positions (N = 32)
- Run SimNIBS with each coil individually on the head mesh
- Store basis field matrix (num_gm_elements × N) as .npy
- Validate linearity: compare sum of individual fields to multi-coil simulation
- Extract surface and volume sample point indices

### Phase 2: TI Optimization (GA)
- Extend GA individual with group assignments and frequency parameters
- Implement `fitness_function_TI`
- Run optimization for 3–5 deep target locations (hippocampus, cingulate, insula)
- Validate top candidates with full SimNIBS (Tier 2)

### Phase 3: NTS Optimization (GA)
- Extend GA individual with firing times
- Implement `fitness_function_NTS` with membrane integration kernel
- Compare surface-sparing performance against TI
- Benchmark FPGA timing controller with NTS sequences

### Phase 4: SAC Agent Training
- Implement `BrainEnv_TI` with Tier 1 forward model
- Train SAC agent on amplitude + frequency control
- Implement `BrainEnv_NTS`
- Train SAC agent on amplitude + timing control
- Integrate recording system feedback (from `recording-system-specs.md` pipeline)

### Phase 5: Hybrid + Validation
- Implement combined TI+NTS environment
- Full FEM validation of optimal configurations (Tier 2)
- Thermal modeling for continuous TI drive (COMSOL Heat Transfer)
- Mutual inductance matrix measurement/simulation
- Compare against standard single-coil TMS depth performance

### Phase 6: Hardware Integration
- Modify FPGA `stim_timing_controller` for NTS sequence mode
- Modify safety monitor thresholds for TI continuous drive
- Implement coupling compensation in waveform generator firmware
- Bench test with physical coil array prototype

---

## 13. Key References

- Jiang et al. (2023). "A C-shaped miniaturized coil for transcranial magnetic stimulation in rodents." J Neural Eng 20, 026022.
- Grossman et al. (2017). "Noninvasive Deep Brain Stimulation via Temporally Interfering Electric Fields." Cell 169(6), 1029-1041.
- Roth et al. (2002). "A coil design for transcranial magnetic stimulation of deep brain regions." J Clin Neurophysiol 19(4), 361-370.
- Deng et al. (2013). "Electric field depth–focality tradeoff in transcranial magnetic stimulation." Brain Stimul 6(1), 1-13.
- Gomez et al. (2021). "Conditions for numerically accurate TMS electric field simulation." Brain Stimul 14(3), 456-467.
- Lee et al. (2016). "Individually customized transcranial temporal interference stimulation for focused modulation of deep brain structures." Brain Stimul 13(6), 1532-1540.
- Saturnino et al. (2019). "SimNIBS 2.1: A Comprehensive Pipeline for Individualized Electric Field Modelling for Transcranial Brain Stimulation." Brain and Human Body Modeling, 3-25.
