# Control-Theoretic Framework for the Omnidream TMS Array

> Formal state-space formulation, controllability analysis, action space geometry,
> and abstraction layers for multi-modal neural stimulation control.

---

## §1  State-Space Formulation

### 1.1  System Variables

We model the Omnidream TMS array as a multi-input multi-output (MIMO) dynamical system.

**State vector** x ∈ ℝⁿ (n = 2N + 2 for N coils):

    x = [I₁, I₂, …, I_N,          # Coil currents [A]
         V_m(target),               # Membrane potential at target [V, normalized]
         T₁, T₂, …, T_N,          # Coil surface temperatures [°C]
         M_target]                  # TI modulation depth at target [V/m]

**Input vector** u ∈ ℝᵐ (m = N):

    u = [V₁, V₂, …, V_N]          # Voltage commands to coil drivers [V]

**Output vector** y ∈ ℝᵖ (p = 5):

    y = [M_target,                  # TI modulation depth at focal point
         M_surface_max,             # Maximum surface modulation (spillover)
         V_m_peak,                  # Peak membrane potential at target
         SAR_max,                   # Maximum SAR in tissue [W/kg]
         T_coil_max]               # Hottest coil temperature [°C]

**Parameter vector** θ (fixed per session):

    θ = {positions, orientations, group_assignment, freq_carrier, delta_freq, mode}

### 1.2  Nonlinear Dynamics

The full nonlinear system ẋ = f(x, u, θ) consists of three coupled subsystems:

**Circuit dynamics** (fast, τ ~ 3 μs):

    dI/dt = L⁻¹(V - RI)

where L ∈ ℝᴺˣᴺ is the inductance matrix (self + mutual), R = diag(R₁, …, R_N).

In the quasi-static TMS regime, coil currents settle within microseconds — effectively
instantaneous relative to neural dynamics (ms) and thermal dynamics (s).

**Neural dynamics** (medium, τ_m ~ 3 ms):

    τ_m dV_m(target)/dt = -V_m(target) + Q · |E(target)|

where Q is the neural gain factor and E(target) = B(target,:) @ (α ⊙ ω) is the
E-field magnitude at the target computed from the basis matrix.

For TI mode, the effective stimulus is the modulation depth M(target):

    τ_m dV_m/dt = -V_m + Q · M(target)

**Thermal dynamics** (slow, τ_th ~ 100 s):

    C_th dT_i/dt = I_i² R_i - h_conv (T_i - T_amb)

where C_th is the coil thermal capacitance [J/K] and h_conv is the convective
heat transfer coefficient [W/K].

### 1.3  Output Equations

The output y = h(x, u) maps state to observables:

    M_target = 2 · min(|B(target,:) @ (α ⊙ mask₁ ⊙ ω₁)|,
                       |B(target,:) @ (α ⊙ mask₂ ⊙ ω₂)|)

    M_surface_max = max_{r ∈ surface} M(r)

    V_m_peak = x[N]    (membrane potential state)

    SAR_max = max_r σ|E_rms(r)|² / ρ

    T_coil_max = max_i x[N+1+i]

### 1.4  Time-Scale Separation

The three subsystems operate at well-separated time scales:

| Subsystem | Time constant | Relative speed |
|-----------|--------------|----------------|
| Circuit   | τ_L ≈ 3 μs  | 1000× faster   |
| Neural    | τ_m ≈ 3 ms  | Reference       |
| Thermal   | τ_th ≈ 100 s| 33000× slower  |

This separation enables a **singular perturbation** approach:
- Circuit dynamics can be treated as algebraic (instantaneous I = I_ss(V))
- Neural dynamics are the primary control target
- Thermal dynamics provide a slow safety constraint

The effective reduced-order plant becomes:

    I_ss = R⁻¹ V    (for sinusoidal drive: I = Z⁻¹ V)
    τ_m dV_m/dt = -V_m + g(I_ss)

where g(·) is the nonlinear field-to-modulation mapping.

---

## §2  Linearization

### 2.1  Jacobian Matrices

Linearize around an operating point (x₀, u₀):

    Δẋ = A Δx + B Δu
    Δy = C Δx + D Δu

**A matrix** (∂f/∂x at x₀):

For the reduced-order model (circuit instantaneous):

    A = [-1/τ_m]    (scalar, for membrane dynamics only)

For the full model:

    A = [A_circuit  |  0      |  0     |  0    ]
        [A_neuro    | -1/τ_m  |  0     |  0    ]
        [A_therm    |  0      | A_cool |  0    ]
        [A_mod      |  0      |  0     |  0    ]

where:
- A_circuit = -L⁻¹R ∈ ℝᴺˣᴺ (stable, fast poles at -R_i/L_i)
- A_neuro = Q · ∂|E(target)|/∂I ∈ ℝ¹ˣᴺ
- A_therm = diag(2 I_i R_i / C_th) (linearized I²R heating)
- A_cool = diag(-h_conv / C_th) (convective cooling, stable)
- A_mod = ∂M_target/∂I (TI-specific, involves min() derivative)

**B matrix** (∂f/∂u at x₀):

    B = [L⁻¹     ]    # voltage → current
        [0        ]    # no direct voltage-to-membrane
        [0        ]    # no direct voltage-to-temperature
        [0        ]    # no direct voltage-to-modulation

**C matrix** (∂h/∂x):

    C = [∂M_target/∂I,    0,  0,  ∂M_target/∂M  ]
        [∂M_surf/∂I,      0,  0,  0              ]
        [0,                1,  0,  0              ]
        [∂SAR/∂I,         0,  0,  0              ]
        [0,                0,  I_N, 0             ]

where I_N selects max(T_i).

**D matrix** (∂h/∂u, direct feedthrough):

    D ≈ 0  (no instantaneous voltage-to-output path in continuous-time)

For the quasi-static approximation (I = Z⁻¹V), there IS effective feedthrough:

    D_eff = C · (0; B_circuit) · Z⁻¹

### 2.2  Key Jacobians for Optimization

The most important Jacobian for optimization is ∂y/∂α — how output metrics change
with amplitude parameters (not voltages, but the abstract optimization variables):

**TI mode:**

    ∂M(r)/∂αᵢ = 2 · ∂min(|A₁|, |A₂|)/∂αᵢ

For the min function:
- If |A₁(r)| < |A₂(r)|: ∂M/∂αᵢ = 2 · sign(A₁) · ∂A₁/∂αᵢ = 2 · sign(A₁) · Bᵢ(r) · 2πf₁ · (1-gᵢ)
- If |A₂(r)| < |A₁(r)|: ∂M/∂αᵢ = 2 · sign(A₂) · Bᵢ(r) · 2πf₂ · gᵢ

**NTS mode:**

    ∂V_peak(r)/∂αᵢ = Q · |Eᵢ(r)| · exp(-(t_N - tᵢ)/τ_m)
    ∂V_peak(r)/∂tᵢ = Q · αᵢ · |Eᵢ(r)| · (1/τ_m) · exp(-(t_N - tᵢ)/τ_m)

These are analytically computable (no finite differences needed).

---

## §3  Controllability & Observability

### 3.1  Controllability

**Definition:** The system (A, B) is controllable if any state can be reached from any
other state in finite time using admissible inputs.

**Controllability matrix:**

    𝒞 = [B, AB, A²B, …, Aⁿ⁻¹B] ∈ ℝⁿˣⁿᵐ

**Rank test:** System is controllable iff rank(𝒞) = n.

**Practical interpretation for Omnidream:**
- If rank(𝒞) = n: any target location can be stimulated (in principle)
- If rank(𝒞) < n: some directions in state space are unreachable
- The deficit dim(null(𝒞)) tells us how many "modes" are uncontrollable

**Controllability Gramian** (for stable A):

    W_c = ∫₀^∞ e^{At} B Bᵀ e^{Aᵀt} dt

Solution: W_c satisfies the Lyapunov equation A W_c + W_c Aᵀ + B Bᵀ = 0.

Energy-optimal control: The minimum energy to reach state x₁ from origin is:

    E_min = x₁ᵀ W_c⁻¹ x₁

Large eigenvalues of W_c correspond to easily-reachable directions; small eigenvalues
are expensive to reach.

### 3.2  Observability

**Definition:** The system (A, C) is observable if the initial state can be uniquely
determined from the output history.

**Observability matrix:**

    𝒪 = [C; CA; CA²; …; CAⁿ⁻¹] ∈ ℝⁿᵖˣⁿ

**Rank test:** System is observable iff rank(𝒪) = n.

**Practical interpretation:**
- With 16-channel recordings (p_recording = 16) and 32 coils (N = 32):
  rank(𝒪) can be at most min(n, n·p_recording)
- Full observability requires that recording electrodes span distinct coil influences

### 3.3  Controllability-Observability Duality

For the Omnidream system, controllability (can we reach the target?) and observability
(can we measure what happened?) are dual:
- (A, B) controllable ↔ (Aᵀ, Bᵀ) observable
- The recording system must observe the same modes we're trying to control

### 3.4  Practical Controllability Test

For Omnidream, the key question is not full state controllability but **output
controllability** — can we independently set y₁ = M_target high while keeping
y₂ = M_surface_max low?

Output controllability matrix:

    𝒞_out = C · 𝒞

rank(𝒞_out) = p iff all outputs are independently achievable.

If rank(𝒞_out) < p, some output trade-offs are fundamental (physics-limited),
not optimizer-limited.

---

## §4  Action Space Geometry

### 4.1  The Parameter Manifold

The optimizer's decision space is the parameter manifold:

    ℳ = {(α, g, t, f₁, f₂) : αᵢ ∈ [0, α_max], gᵢ ∈ {0,1},
         tᵢ ∈ [0, τ_window], f₁ ∈ [f_min, f_max], f₂ = f₁ + Δf}

This is a mixed continuous-discrete space:
- Continuous subspace: ℝᴺ (amplitudes) × ℝᴺ (timings) × ℝ² (frequencies)
- Discrete subspace: {0,1}ᴺ (group assignments)

The GA explores the full manifold; the SAC agent operates on the continuous subspace
with fixed discrete parameters.

### 4.2  Metric Tensor (Hessian of Objective)

The natural metric on the continuous subspace is the Hessian of the objective:

    g_ij = ∂²J/∂θᵢ∂θⱼ

where θ = (α₁, …, α_N, t₁, …, t_N, f₁, Δf).

The Hessian defines:
- **Curvature:** Eigenvalues of H indicate how quickly J changes in each direction
- **Condition number:** κ = λ_max/λ_min — ratio of steepest to shallowest direction
- **Anisotropy:** Eigenvectors show which parameter combinations matter most

### 4.3  Condition Number and Sensitivity

The condition number of the Jacobian J = ∂y/∂u characterizes the inverse problem:

    κ(J) = σ_max(J) / σ_min(J)

where σᵢ are singular values.

| κ value | Interpretation |
|---------|----------------|
| κ ≈ 1   | Well-conditioned: all parameters equally important |
| κ ≈ 10  | Mildly ill-conditioned: some directions are ~10× less sensitive |
| κ > 100 | Ill-conditioned: some parameters barely affect output |
| κ = ∞   | Singular: redundant parameters, rank-deficient |

For a well-designed 32-coil helmet, we expect κ ≈ 5–50, depending on target depth
and coil arrangement quality.

### 4.4  Reachable Set

The reachable set is the region in output space (y₁, y₂) achievable by varying
inputs within their constraints:

    ℛ = {h(x, u) : u ∈ [u_min, u_max]}

For the key trade-off (M_target vs M_surface_max), the reachable set boundary is
the Pareto front — the set of non-dominated solutions.

**Properties:**
- ℛ is convex if h is linear in u (true for basis field superposition)
- The Pareto front is the upper-left boundary of ℛ (high target, low surface)
- Points inside ℛ are suboptimal; points outside are unreachable

### 4.5  Pareto Front: The Fundamental Trade-Off

The depth-selectivity trade-off is quantified by the Pareto front:

    For each λ ∈ [0, 1]:
        α*(λ) = argmin_α { λ · (-M_target(α)) + (1-λ) · M_surface(α) }
        subject to: 0 ≤ αᵢ ≤ α_max, safety constraints

The resulting curve {(M_target(α*(λ)), M_surface(α*(λ)))} for λ ∈ [0, 1] traces
the achievable trade-off boundary.

---

## §5  Abstraction for Multi-Modal Goal Alignment

### 5.1  The Plant-Controller Interface

The control framework separates physics from control through abstract interfaces:

**PlantModel** (physics layer):
- forward(x, u) → y: compute output from state+input
- linearize(x₀, u₀) → (A, B, C, D): local linear model
- jacobian(x, u) → J: output sensitivity ∂y/∂u

**Controller** (control layer):
- select_action(state, goal) → u: compute control input

**GoalSpec** (objective layer):
- target_coords: where to stimulate
- weights: how to trade off competing objectives
- constraints: safety limits

### 5.2  Swapping Modalities

The PlantModel ABC encapsulates all modality-specific physics. Swapping from TMS to
another modality requires only a new PlantModel implementation:

| Modality | Basis Matrix | Forward Model | Action Space |
|----------|-------------|---------------|--------------|
| **TMS-TI** | E-field from SimNIBS | M = 2·min(\|A₁\|,\|A₂\|) | amplitudes + Δf |
| **TMS-NTS** | E-field from SimNIBS | V = Σ α·\|E\|·exp decay | amplitudes + timings |
| **FUS** | Pressure from k-Wave | P = Σ αᵢ pᵢ(r) | transducer amplitudes |
| **tDCS** | E-field from SimNIBS | J = σE (no dynamics) | electrode currents |
| **Optogenetics** | Light intensity from MC | I = Σ αᵢ lᵢ(r) | LED intensities |

The Controller, GoalSpec, and optimization infrastructure (GA, SAC) remain unchanged.

### 5.3  Goal Alignment Across Systems

A GoalSpec defines WHAT the system should achieve, independent of HOW:

    goal = GoalSpec(
        target_coords = [x, y, z],
        objectives = {
            SystemGoal.FOCAL_DEPTH: 1.0,      # maximize target metric
            SystemGoal.SPATIAL_SELECTIVITY: 0.5,# minimize off-target
            SystemGoal.POWER_EFFICIENCY: 0.1,   # minimize energy
        },
        constraints = {
            "SAR_max": 3.2,     # W/kg
            "T_max": 41.0,      # °C
            "I_max": 5.0,       # A per coil
        }
    )

The same goal can drive different plants: "stimulate point (x,y,z) with high selectivity"
works identically whether the plant is TMS, FUS, or optogenetics.

### 5.4  Control Hierarchy with Goal Alignment

The two-loop control architecture maps to goal alignment as follows:

    Level 0 (User):       Define GoalSpec (target, weights, constraints)
                              ↓
    Level 1 (GA):         Optimize STRUCTURE (positions, groups, frequencies)
                          to maximize reachable set extent toward goal
                              ↓
    Level 2 (SAC/MPC):    Optimize PARAMETERS (amplitudes, timings)
                          in real-time toward goal within fixed structure
                              ↓
    Level 3 (FPGA):       Execute hardware commands with μs precision

Each level operates on a different timescale and abstraction:
- Level 0: Human-interpretable goals (minutes to configure)
- Level 1: Structural optimization (hours to compute)
- Level 2: Continuous control (real-time, ms decisions)
- Level 3: Hardware execution (μs precision)

---

## §6  Summary: Control-Theoretic Objects

| Object | Symbol | Dimension | Computation |
|--------|--------|-----------|-------------|
| State | x | 2N+2 | Circuit + membrane + thermal |
| Input | u | N | Voltage commands |
| Output | y | 5 | M_target, M_surface, V_m, SAR, T |
| Plant matrix A | A | (2N+2)² | ∂f/∂x (linearized) |
| Input matrix B | B | (2N+2)×N | ∂f/∂u = L⁻¹ |
| Output matrix C | C | 5×(2N+2) | ∂h/∂x |
| Controllability | 𝒞 | (2N+2)×N(2N+2) | [B, AB, …] |
| Jacobian | J | 5×N | ∂y/∂α |
| Hessian | H | N×N | ∂²J/∂α² |
| Condition number | κ | scalar | σ_max/σ_min |
| Reachable set | ℛ | region in ℝ² | {(M_t, M_s) : α feasible} |
| Pareto front | 𝒫 | curve in ℝ² | Non-dominated boundary of ℛ |
