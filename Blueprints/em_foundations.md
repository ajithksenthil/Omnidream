# Electromagnetic Foundations of the Omnidream Miniature TMS Array

> A first-principles derivation of every link in the physics chain:
> voltage command → coil current → magnetic field → induced E-field → neural membrane response → safety limits.

---

## §1  Maxwell's Equations to Coil E-Fields

### 1.1  The Governing Equations

All electromagnetic phenomena in the Omnidream system are governed by Maxwell's equations in matter:

$$
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}  \quad \text{(Faraday's law)}
$$

$$
\nabla \times \mathbf{H} = \mathbf{J}_f + \frac{\partial \mathbf{D}}{\partial t}  \quad \text{(Ampère-Maxwell law)}
$$

$$
\nabla \cdot \mathbf{B} = 0 \quad \text{(no magnetic monopoles)}
$$

$$
\nabla \cdot \mathbf{D} = \rho_f  \quad \text{(Gauss's law)}
$$

with constitutive relations B = μH, D = εE, J_f = σE in linear, isotropic media.

### 1.2  The Quasi-Static Approximation

For TMS, the displacement current ∂D/∂t is negligible because the operating frequencies (f ~ 500–2500 Hz for TI carriers, up to 10 kHz for individual pulse rise times) produce wavelengths far larger than the head:

    λ = c/f = (3 × 10⁸ m/s) / (1 × 10³ Hz) = 300 km

Since λ ≈ 300 km ≫ 0.2 m (head diameter), the fields propagate instantaneously across the head. The quasi-static approximation is valid whenever:

    λ / (head diameter) > 100

This holds for all Omnidream operating frequencies up to at least 100 kHz. Under quasi-static conditions, Ampère-Maxwell reduces to Ampère's law:

    ∇ × H = J_f

and Faraday's law remains:

    ∇ × E = -∂B/∂t

### 1.3  Why E ∝ dI/dt (Not I)

The TMS coil creates a time-varying current I(t), which produces a time-varying magnetic field B(r, t). By Faraday's law, this changing B induces an electric field E in the tissue:

    ∇ × E = -∂B/∂t

Since B(r, t) is proportional to I(t) (linear media, no saturation), we can write:

    B(r, t) = b(r) · I(t)

where b(r) is the spatial field pattern per unit current. Then:

    ∂B/∂t = b(r) · dI/dt

and therefore:

    E(r, t) ∝ dI/dt

This is the fundamental reason why TMS E-fields scale with the rate of change of current, not the current itself. For sinusoidal drive I(t) = α sin(2πft):

    dI/dt = α · 2πf · cos(2πft)

The peak E-field is proportional to the product α · 2πf — both amplitude and frequency matter.

### 1.4  Biot-Savart Law for the C-Shaped Coil

For a coil carrying current I, the magnetic field is given by the Biot-Savart law:

    B(r) = (μ₀ / 4π) ∮ I dl' × (r - r') / |r - r'|³

For the Omnidream C-shaped miniature coil (Jiang et al. 2023):
- Core: carbonyl iron powder, μ_r = 75 (paper baseline) or ferrite μ_r = 5000 (variant)
- Dimensions: 7 mm width × 4 mm height × 5.2 mm depth (paper baseline)
- Winding: 30 turns of 0.2 mm copper wire
- Gap: 5 mm (paper) or 0.5 mm (variant)

The ferrite core amplifies the field by a factor related to the effective permeability:

    B_eff = μ_eff · H_coil

where μ_eff depends on the core geometry and demagnetization factor N_d:

    μ_eff = μ₀ μ_r / (1 + N_d(μ_r - 1))

For an elongated core (length ≫ diameter), N_d → 0 and μ_eff → μ₀ μ_r. For the C-shape, N_d ≈ 0.1–0.3 depending on gap/length ratio, giving a significant but not full enhancement.

### 1.5  Basis Field Decomposition — Proof of Linearity

**Theorem:** If the tissue conductivity σ(r) is independent of the applied field (linear medium), then the total E-field from N coils is the superposition of individual coil fields:

    E_total(r) = Σᵢ αᵢ · Eᵢ(r)

**Proof:** In the quasi-static limit, the induced E-field satisfies:

    ∇ · (σ(r) ∇φ(r)) = ∇ · (σ(r) A_s(r))

where φ is the scalar electric potential and A_s is the source term from the coil's vector potential A:

    E = -∇φ - ∂A/∂t

Since σ(r) is fixed (does not depend on E), this is a linear PDE. Let φᵢ be the solution for coil i alone with unit-strength drive. Then for N coils with amplitudes αᵢ:

    ∇ · (σ ∇(Σ αᵢ φᵢ)) = Σ αᵢ ∇ · (σ ∇φᵢ) = Σ αᵢ ∇ · (σ A_s,i)  = ∇ · (σ Σ αᵢ A_s,i)

By uniqueness of the Poisson solution (with appropriate boundary conditions), φ_total = Σ αᵢ φᵢ, and therefore:

    E_total = Σ αᵢ Eᵢ    ∎

**When linearity breaks:**
1. Core saturation: When B approaches B_sat (typically 0.3–0.5 T for ferrite, 1.5 T for iron powder), μ_r drops and the field-current relationship becomes nonlinear. For Omnidream at I < 5 A, B_core ≈ μ₀ μ_r N I / ℓ_core. With μ_r = 75, N = 30, I = 5, ℓ = 12 mm: B ≈ 0.012 T ≪ B_sat. Linearity is safe for the paper baseline. For μ_r = 5000 variant: B ≈ 0.79 T — near saturation; must verify.

2. Tissue nonlinearity: Conductivity σ becomes field-dependent for |E| > ~10 V/m due to electroporation effects. For Omnidream surface fields of ~7 V/m, this is marginal. Deep fields are ~0.1–1 V/m, safely linear.

### 1.6  From SimNIBS to Basis Fields

SimNIBS solves the quasi-static Poisson equation on a realistic head mesh using the finite element method (FEM). For each coil i placed at position pᵢ with orientation nᵢ:

1. The coil's magnetic vector potential A_s,i(r) is computed from the coil geometry and a reference dI/dt
2. The FEM solves ∇·(σ∇φᵢ) = ∇·(σ A_s,i) on the tetrahedral mesh
3. The induced E-field is Eᵢ = -∇φᵢ - ∂Aᵢ/∂t
4. We extract |Eᵢ(r)| on gray matter elements → basis vector eᵢ ∈ ℝ^P

The full basis matrix B ∈ ℝ^(P×N) has columns [e₁, e₂, …, e_N]. All downstream computations reduce to matrix operations on B.

**Units convention:** Basis fields are stored as V/m per A/s at a reference dI/dt (typically 10⁶ A/s). Actual fields are:

    E_actual(r) = B @ (α ⊙ ω) / dI_dt_ref

where ω = [2πf₁, …, 2πf_N] accounts for the frequency-dependent dI/dt scaling.

---

## §2  Tissue Interaction

### 2.1  Head Tissue Conductivities

The human head is a heterogeneous conductor with tissue-specific electrical properties:

| Tissue     | σ (S/m) at DC | σ (S/m) at 1 kHz | ε_r at 1 kHz | Thickness |
|------------|---------------|-------------------|--------------|-----------|
| Scalp      | 0.45          | 0.45              | ~10⁴         | 5–7 mm    |
| Skull      | 0.010         | 0.012             | ~10³         | 6–8 mm    |
| CSF        | 1.654         | 1.654             | ~100         | 1–3 mm    |
| Gray matter| 0.106         | 0.110             | ~10⁵         | 2–4 mm    |
| White matter| 0.065        | 0.068             | ~10⁵         | bulk      |

The skull acts as a high-impedance barrier (σ_skull ≈ 100× less than GM), while CSF is a low-impedance channel that can "short-circuit" fields along sulci.

### 2.2  Frequency-Dependent Conductivity: Cole-Cole Model

At TI carrier frequencies (500–2500 Hz), tissue conductivity has weak but non-negligible frequency dependence described by the Cole-Cole relaxation:

    σ(ω) = σ_∞ + (σ₀ - σ_∞) / (1 + (jωτ)^α)

Parameters for gray matter (Gabriel et al. 1996):
- σ₀ ≈ 0.020 S/m (DC component of this dispersion)
- σ_∞ ≈ 0.106 S/m (high-frequency limit for this band)
- τ ≈ 7.96 × 10⁻³ s (relaxation time)
- α ≈ 0.1 (broadening exponent)

For Omnidream's operating range (500–2500 Hz), the conductivity variation is small (~5% change from DC to 2.5 kHz for GM). The basis field approach uses a single conductivity value, introducing a ~5% systematic error that can be corrected if needed by recomputing basis fields at each carrier frequency.

### 2.3  Why FEM Is Necessary

Analytical solutions to ∇·(σ∇φ) = source exist only for simple geometries (concentric spheres, infinite half-spaces). The human head requires FEM because:

1. **Skull geometry**: Non-uniform thickness, sutures, foramina create complex current paths
2. **Sulcal folding**: CSF-filled sulci channel currents deep into cortex
3. **Conductivity discontinuities**: σ jumps by 100× at skull-CSF boundary, causing field focusing
4. **Anisotropy**: White matter fiber tracts have direction-dependent conductivity (σ_∥ ≈ 10× σ_⊥)

SimNIBS discretizes the head into ~500,000 tetrahedral elements with tissue-specific σ and solves the system using a sparse linear solver. Each coil simulation takes ~30–60 seconds on modern hardware.

### 2.4  Field Penetration and Depth-Focality Tradeoff

For a single small coil on the scalp, the E-field decays approximately as:

    |E(z)| ≈ |E₀| · (d / (d + z))³

where d is the effective coil diameter and z is the depth below the scalp. This means:
- At z = d (one coil-diameter depth): |E| ≈ |E₀|/8
- At z = 2d: |E| ≈ |E₀|/27

For Omnidream's 7 mm coils, significant field exists only to ~10–15 mm depth. Reaching deeper structures (e.g., thalamus at 50–60 mm) requires either:
- **TI**: Two groups at offset frequencies create constructive interference at depth
- **NTS**: Temporal summation accumulates sub-threshold pulses from multiple coils

The depth-focality tradeoff is fundamental: any linear combination of surface sources that focuses at depth necessarily has stronger fields on the surface. TI and NTS break this symmetry through temporal coding.

---

## §3  Coupled Circuit-Field Equations

### 3.1  Single Coil Circuit Model

Each coil is an RLC circuit (ignoring parasitic capacitance for f < 100 kHz):

    V_i(t) = L_i · dI_i/dt + R_i · I_i(t)

where:
- V_i(t) = applied voltage from MOSFET driver [V]
- L_i = self-inductance [H]
- R_i = total resistance (wire + core loss) [Ω]
- I_i(t) = coil current [A]

Transfer function (Laplace domain):

    H_i(s) = I_i(s) / V_i(s) = 1 / (L_i s + R_i)

This is a first-order low-pass with time constant τ_L = L_i / R_i.

### 3.2  Self-Inductance: Wheeler/Nagaoka Formulas

For a single-layer rectangular solenoid of N turns, width w, height h, depth d:

**Wheeler formula** (empirical, ±1% for common geometries):

    L = (μ₀ μ_r N² A) / ℓ_eff · K_N

where:
- A = w × h = cross-sectional area of the core
- ℓ_eff = effective magnetic path length ≈ d + gap for C-shape
- K_N = Nagaoka coefficient (correction for finite solenoid)

For the Omnidream paper baseline (N=30, w=7mm, h=4mm, d=5.2mm, gap=5mm, μ_r=75):

    A = 7e-3 × 4e-3 = 2.8e-5 m²
    ℓ_eff ≈ 10.2e-3 m (core depth + gap)
    L_approx = (4π×10⁻⁷ × 75 × 900 × 2.8e-5) / 10.2e-3
             ≈ 2.3e-4 H ≈ 230 μH

However, the C-shaped geometry with air gap reduces L significantly. Using magnetic circuit theory:

    L = μ₀ N² / (ℓ_core/(μ_r A_core) + ℓ_gap/(A_gap))

With ℓ_core ≈ 15 mm, ℓ_gap = 5 mm, and assuming A_core ≈ A_gap:

    L = 4π×10⁻⁷ × 900 / (0.015/(75 × 2.8e-5) + 0.005/2.8e-5)
      = 1.131e-3 / (7143 + 178571)
      ≈ 6.1 μH

The air gap dominates the reluctance, keeping L small (~5–10 μH typical). This is advantageous for fast current switching required by NTS timing.

### 3.3  Mutual Inductance: From Neumann to Dipole

**Exact (Neumann formula):**

    M_ij = (μ₀ / 4π) ∮_i ∮_j (dl_i · dl_j) / |r_i - r_j|

This double line integral is expensive to compute for arbitrary coil geometries.

**Dipole approximation** (valid when d_ij ≫ coil size):

For two magnetic dipoles with moments m_i = N_i I_i A_i n̂_i and m_j = N_j I_j A_j n̂_j separated by distance d_ij:

    M_ij ≈ (μ₀ / 4π) · (N_i N_j A_i A_j) / d_ij³ · [3(n̂_i · r̂)(n̂_j · r̂) - n̂_i · n̂_j]

For coils on a concave helmet surface with normals pointing inward (n̂_i · n̂_j ≈ 1 for nearby coils, n̂ · r̂ varies):

    M_ij ≈ (μ₀ n² A²) / (4π d³)    [simplified, aligned dipoles]

**Numerical check** (Omnidream: n=30, A=28mm², d=25mm):

    M_ij ≈ (4π×10⁻⁷ × 900 × (2.8e-5)²) / (4π × (0.025)³)
         ≈ (1.131e-3 × 7.84e-10) / (4π × 1.5625e-5)
         ≈ 8.87e-13 / 1.96e-4
         ≈ 4.5 nH

With L_self ≈ 6 μH:

    k = M / √(L_i L_j) ≈ 4.5e-9 / 6e-6 ≈ 7.5e-4

This confirms k < 0.01, validating the weak-coupling assumption. However, for densely packed arrays (d < 15 mm), the dipole approximation underestimates coupling and FEM-extracted inductances should be used.

### 3.4  N-Coil Coupled System

The complete circuit equations for N coupled coils:

    V(t) = L · dI/dt + R · I(t)

where:
- V = [V₁, …, V_N]ᵀ ∈ ℝᴺ (voltage commands)
- I = [I₁, …, I_N]ᵀ ∈ ℝᴺ (coil currents)
- L ∈ ℝᴺˣᴺ (full inductance matrix, L_ii = self, L_ij = M_ij)
- R = diag(R₁, …, R_N) ∈ ℝᴺˣᴺ (resistance matrix, assumed diagonal)

Rearranging:

    dI/dt = L⁻¹(V - RI)

In state-space form with state x = I:

    ẋ = -L⁻¹R · x + L⁻¹ · u
    ẋ = A_circuit · x + B_circuit · u

where A_circuit = -L⁻¹R and B_circuit = L⁻¹.

The eigenvalues of A_circuit are -R_i/L_ii (approximately, since M ≪ L), giving time constants τ_i = L_i/R_i ≈ 6μH/2Ω = 3 μs. This is much faster than the NTS integration window (5 ms), confirming that coil currents reach steady state within a single pulse.

### 3.5  Core Losses: Steinmetz Equation

For ferrite and iron powder cores, hysteresis and eddy current losses are modeled by:

    P_core = k · f^a · B_max^b    [W/m³]

Typical parameters for carbonyl iron powder (Omnidream baseline):
- k ≈ 6.5 (material constant)
- a ≈ 1.3 (frequency exponent)
- b ≈ 2.5 (flux density exponent)

At f = 1 kHz with B_max = 0.01 T:

    P_core ≈ 6.5 × 1000^1.3 × 0.01^2.5 ≈ 6.5 × 5012 × 3.16e-5 ≈ 1.03 W/m³

For a core volume of ~250 mm³:

    P_total_core ≈ 1.03 × 2.5e-7 ≈ 0.26 μW

This is negligible compared to I²R resistive losses (~5 W at 5 A), confirming that core losses can be ignored in the circuit model for Omnidream's operating regime.

### 3.6  Coupling Compensation

To achieve desired coil currents I_desired despite mutual coupling, we solve for the required voltage commands:

    V_command = Z · I_desired

where Z is the complex impedance matrix:

    Z_ii = R_i + j2πf L_ii
    Z_ij = j2πf M_ij    (i ≠ j)

This is computed once per frequency setting (static compensation). For real-time adjustment, the controller applies:

    V(t) = Z · I_desired(t)

The impedance matrix Z is well-conditioned when k ≪ 1 (as confirmed above), so the inversion is numerically stable.

---

## §4  Neural Membrane Response

### 4.1  The Cable Equation

A neuron's axon or dendrite is modeled as a leaky cable:

    τ_m ∂V_m/∂t = λ² ∂²V_m/∂x² - V_m + r_m · I_ext(x, t)

where:
- V_m(x, t) = transmembrane potential [V]
- τ_m = R_m C_m = membrane time constant [s]
  - Typical: τ_m ≈ 1–10 ms (3 ms default for cortical pyramidal cells)
- λ = √(a R_m / (4 R_i)) = electrotonic length constant [m]
  - Typical: λ ≈ 0.1–1 mm for cortical neurons
- R_m = membrane resistance per unit area [Ω·m²]
- C_m ≈ 0.01 F/m² (universal for biological membranes)
- R_i = intracellular resistivity [Ω·m]
- a = axon diameter [m]
- r_m = R_m / (2πa) = membrane resistance per unit length [Ω·m]
- I_ext = external current density induced by E-field [A/m]

### 4.2  The Activating Function (Rattay 1986)

The external current injected by the TMS-induced E-field is:

    I_ext(x) = (1/r_i) · ∂E_x/∂x

where r_i is the intracellular resistance per unit length and E_x is the component of the induced E-field along the axon direction x. The activating function is:

    f_act(x) = ∂E_x/∂x    [V/m²]

This means neurons are sensitive to the *spatial gradient* of the E-field along their length, not just the field magnitude. Depolarization occurs where f_act > 0 (E-field is diverging), and hyperpolarization where f_act < 0.

### 4.3  Why |E| Is Used as a Proxy

In practice, we use |E(r)| rather than f_act because:

1. **Orientation averaging:** Cortical neurons have diverse orientations. For a population of neurons at random angles θ to the E-field, the expected activating function scales as:
   ⟨|f_act|⟩ ∝ |∇E| ≈ |E| / L_char
   where L_char is a characteristic length scale.

2. **Simplified threshold:** For the integrate-and-fire approximation used in NTS, the membrane charge accumulated per pulse is proportional to the local E-field magnitude integrated over the pulse duration:
   ΔV_m ∝ ∫ |E(r, t)| dt = |E(r)| · Q_pulse

3. **SimNIBS convention:** The standard output metric from FEM simulations is |E| on tissue elements, which is what our basis matrix stores.

The |E| approximation introduces ~20–30% error in predicting exact threshold locations compared to the full activating function (Aberra et al. 2020), but captures the correct spatial distribution for optimization purposes.

### 4.4  TI: Why Neurons Respond to the Beat Envelope

When two coil groups produce E-fields oscillating at frequencies f₁ and f₂ (both > 500 Hz):

    E_total(r, t) = A₁(r) cos(2πf₁t) + A₂(r) cos(2πf₂t)

The neural membrane acts as a low-pass filter with cutoff frequency:

    f_c = 1/(2π τ_m) ≈ 1/(2π × 3ms) ≈ 53 Hz

The individual carriers f₁, f₂ ≫ f_c are filtered out, but the beat envelope at frequency Δf = |f₁ - f₂| (typically 1–100 Hz) passes through because Δf < f_c.

**Formal derivation:** The membrane transfer function is:

    H_m(ω) = 1 / (1 + jωτ_m)

The combined E-field has frequency components at f₁, f₂, and the sum/difference frequencies. The envelope modulation at Δf has amplitude:

    M(r) = 2 · min(|A₁(r)|, |A₂(r)|)

This modulation depth determines the effective stimulation at each point. Neurons at locations where M(r) > V_th will be activated, while regions with high individual group amplitudes but low modulation (e.g., directly under the coil groups) remain unstimulated.

**Key insight for action space:** The TI modulation depth is a nonlinear function of the amplitude partition between groups. The min() function creates a non-smooth landscape — the optimizer must balance group amplitudes rather than maximizing either one.

### 4.5  NTS: Temporal Summation in Integrate-and-Fire

For N coils firing sequentially at times t₁ < t₂ < … < t_N within a window τ_window ≤ 5 ms:

Each pulse i deposits a charge proportional to αᵢ |Eᵢ(r)| into the membrane capacitor. Between pulses, this charge decays exponentially with time constant τ_m:

    V_m(r, t) = Q_pulse Σᵢ αᵢ |Eᵢ(r)| exp(-(t - tᵢ)/τ_m) · H(t - tᵢ)

where H(·) is the Heaviside step function. At the time of the last pulse t_N:

    V_peak(r) = Q_pulse Σᵢ αᵢ |Eᵢ(r)| exp(-(t_N - tᵢ)/τ_m)

**Optimal firing order:** Coils should fire weakest-first (ascending |Eᵢ(target)|) so that the strongest contributors fire last and experience the least decay. This is proven by the rearrangement inequality: for sequences aᵢ (field strengths) and bᵢ (decay weights), Σ aᵢ bᵢ is maximized when both are sorted in the same order. Since decay weights decrease with time from last pulse (bᵢ = exp(-(t_N - tᵢ)/τ_m), largest for i = N), the strongest field should be paired with the largest weight (i = N).

**Key insight for action space:** The NTS action space has both continuous (amplitudes, timing) and ordering (which coil fires when) components. The ordering dimension is discrete but strongly affects the objective.

### 4.6  Hodgkin-Huxley vs. Integrate-and-Fire

The simplified exponential decay model used in NTS is an integrate-and-fire (IF) approximation of the full Hodgkin-Huxley (HH) dynamics:

**Hodgkin-Huxley (exact but expensive):**

    C_m dV_m/dt = -g_Na m³h(V_m - E_Na) - g_K n⁴(V_m - E_K) - g_L(V_m - E_L) + I_ext

with gating variables m, h, n evolving according to their own differential equations. Full HH simulation requires ~1 μs time steps and is not tractable for optimization over 5000+ spatial points.

**Integrate-and-fire (our model):**

    C_m dV_m/dt = -V_m/R_m + I_ext → τ_m dV_m/dt = -V_m + r_m I_ext

This keeps the subthreshold dynamics (linear leaky integration) but replaces the nonlinear threshold with a fixed V_th. The error is:
- Subthreshold: ~5% (leaky integration is accurate below threshold)
- Near threshold: ~20% (HH threshold depends on rate of approach, prior history)
- Suprathreshold: model breaks down (HH has refractoriness, IF does not)

For optimization purposes, the IF approximation is sufficient because we're maximizing V_peak at the target relative to surface — the relative ranking of configurations is preserved even if absolute threshold predictions have error.

---

## §5  Safety Physics

### 5.1  Specific Absorption Rate (SAR)

SAR quantifies the rate of electromagnetic energy absorption in tissue:

    SAR(r) = σ(r) |E_rms(r)|² / ρ(r)    [W/kg]

where:
- σ(r) = tissue conductivity [S/m]
- E_rms(r) = root-mean-square E-field [V/m]
- ρ(r) = tissue mass density [kg/m³]

**For TI stimulation** with two sinusoidal groups at different frequencies:

    E(r, t) = A₁(r) cos(2πf₁t) + A₂(r) cos(2πf₂t)

The time-averaged power is:

    ⟨|E|²⟩ = (1/2)(|A₁|² + |A₂|²)

because the cross-terms average to zero (f₁ ≠ f₂). Therefore:

    SAR_TI(r) = σ(r) · (|A₁(r)|² + |A₂(r)|²) / (2 ρ(r))

**For NTS stimulation** with brief pulses:

    SAR_NTS(r) = σ(r) · (Σᵢ αᵢ² |Eᵢ(r)|² · duty_i) / ρ(r)

where duty_i = pulse_width / repetition_period is the duty cycle per coil.

**Regulatory limits:**
- IEEE C95.1: 3.2 W/kg head average (10 g)
- IEC 60601: Implied by temperature limits
- ICNIRP 2020: 2 W/kg occupational, 10 W/kg localized

### 5.2  Pennes Bioheat Equation

The tissue temperature evolves according to:

    ρ c ∂T/∂t = k ∇²T + ρ_b c_b w_b (T_a - T) + σ|E|²

where:
- ρ c = volumetric heat capacity [J/(m³·K)]
  - GM: ρ ≈ 1040 kg/m³, c ≈ 3630 J/(kg·K), ρc ≈ 3.78 × 10⁶ J/(m³·K)
- k = thermal conductivity [W/(m·K)]
  - GM: k ≈ 0.51 W/(m·K)
- ρ_b c_b = blood volumetric heat capacity ≈ 3.6 × 10⁶ J/(m³·K)
- w_b = blood perfusion rate [1/s]
  - GM: w_b ≈ 0.008–0.012 s⁻¹ (8–12 mL/100g/min)
- T_a = arterial blood temperature ≈ 37°C
- σ|E|² = electromagnetic heat source [W/m³]

**Steady-state temperature rise** (ignoring spatial gradients):

    ΔT_ss = SAR · ρ / (ρ_b c_b w_b) = σ|E_rms|² / (ρ_b c_b w_b)

For GM with SAR = 3.2 W/kg, ρ_b c_b ≈ 3.6e6, w_b ≈ 0.01:

    ΔT_ss = (3.2 × 1040) / (3.6e6 × 0.01) ≈ 3328 / 36000 ≈ 0.09 °C

This is far below the 4°C limit (reaching 41°C from 37°C). The SAR limit is thus the binding safety constraint for Omnidream, not the thermal limit, unless coil-scalp contact heating is considered separately.

### 5.3  Thermal Time Constants

The characteristic time for temperature to reach steady state:

    τ_thermal = ρ c / (ρ_b c_b w_b) ≈ 3.78e6 / (3.6e6 × 0.01) ≈ 105 s

This means tissue temperature responds slowly (~2 minutes). Short-duration stimulation sessions (< 30 s) will see only ~25% of the steady-state temperature rise.

For the coil surface in contact with scalp:
- Coil heating: P_coil = I² R ≈ 25 × 2 = 50 W (at max current)
- But duty cycle for NTS is ~5%, so average P ≈ 2.5 W per coil
- 32 coils × 2.5 W = 80 W total — manageable with passive cooling but requires monitoring

### 5.4  Safety Limits Summary

| Constraint                    | Limit          | Source       | Binding? |
|-------------------------------|----------------|--------------|----------|
| Head SAR (10g average)        | 3.2 W/kg       | IEEE C95.1   | Yes      |
| Scalp temperature (continuous)| 41°C           | IEC 60601    | No*      |
| Scalp temperature (pulsed)    | 45°C           | IEC 60601    | Marginal |
| Surface E per pulse (NTS)     | 7.2 V/m        | Jiang 2023   | Yes      |
| Max current per coil          | 5.0 A          | Hardware     | Yes      |
| Max voltage                   | 60 V           | Hardware     | Marginal |
| Carrier frequency minimum     | 500 Hz         | Neural safety| Yes      |
| Beat frequency range          | 1–100 Hz       | rTMS norms   | Yes      |
| Inter-pulse guard time (NTS)  | 200 μs         | Coupling     | Yes      |

*Thermal limits become binding only for extended continuous stimulation (> 5 min).

---

## §6  Summary: Complete Physics Chain

```
Voltage Command V_i(t) ∈ ℝ^N
    │
    ▼  [Circuit dynamics: L dI/dt = V - RI]
Coil Current I_i(t) ∈ ℝ^N
    │
    ▼  [Biot-Savart + Faraday: E ∝ dI/dt]
Induced E-field E(r,t) = Σ αᵢ·Eᵢ(r)·2πfᵢ·cos(2πfᵢt) ∈ ℝ^P
    │
    ├── [TI path: M(r) = 2·min(|A₁|, |A₂|)]
    │   └── Modulation depth at beat frequency Δf
    │
    ├── [NTS path: V(r) = Q·Σ αᵢ|Eᵢ|·exp(-(t_N-tᵢ)/τ_m)]
    │   └── Peak membrane potential from temporal summation
    │
    └── [Safety: SAR = σ|E_rms|²/ρ, T via Pennes]
        └── Tissue heating constraint

Neural Response
    │
    ▼  [Activating function: f_act = ∂E_x/∂x, proxy: |E|]
Membrane Depolarization V_m(r,t)
    │
    ▼  [Threshold: V_m > V_th → action potential]
Neural Activation Pattern
    │
    ▼  [Recording: 16-ch MEA, 20 kHz]
Feedback to Controller
```

**Key mathematical objects for the optimizer:**
- Basis matrix B ∈ ℝ^(P×N): encodes spatial physics
- Amplitude vector α ∈ ℝ^N: continuous control variable
- Group vector g ∈ {0,1}^N: discrete TI assignment
- Timing vector t ∈ ℝ^N: NTS firing schedule
- Frequency pair (f₁, f₂) ∈ ℝ²: TI carrier/beat
- Impedance matrix Z ∈ ℂ^(N×N): coupling compensation
- Output metrics y = [M_target, M_surface, V_peak, SAR, T] ∈ ℝ⁵

All downstream optimization operates on these objects. The entire physics chain reduces to matrix operations on B once the basis fields are precomputed.
