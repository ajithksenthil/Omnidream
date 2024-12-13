# Recording System Integration Specifications

## 1. Electrode Configuration

### Physical Layout
```
Electrode specifications:
- Type: Pt-Ir coated microelectrodes
- Diameter: 25 μm
- Impedance: <500 kΩ at 1 kHz
- Spacing: 15 mm between electrodes

Array configuration:
- 16 recording sites (4x4)
- Offset from TMS coils by 12.5 mm diagonally
```

### Placement Optimization Algorithm
```python
import numpy as np
from scipy.optimize import minimize

def optimize_electrode_positions(coil_positions, field_map):
    def objective(positions):
        # Reshape positions into 4x4x2 array
        positions = positions.reshape(4, 4, 2)
        
        # Calculate interference score
        interference = calculate_interference(positions, coil_positions, field_map)
        
        # Calculate coverage score
        coverage = calculate_coverage(positions)
        
        return interference + 0.5 * coverage
    
    # Initial guess for electrode positions
    initial_positions = generate_initial_positions(coil_positions)
    
    # Optimization constraints
    constraints = [
        {'type': 'ineq', 'fun': minimum_distance_constraint},
        {'type': 'ineq', 'fun': boundary_constraint}
    ]
    
    result = minimize(objective, initial_positions, constraints=constraints)
    return result.x.reshape(4, 4, 2)

def calculate_interference(electrode_positions, coil_positions, field_map):
    total_interference = 0
    for ep in electrode_positions.reshape(-1, 2):
        for cp in coil_positions:
            distance = np.linalg.norm(ep - cp)
            field_strength = interpolate_field(field_map, distance)
            total_interference += field_strength / (distance ** 2)
    return total_interference

def calculate_coverage(positions):
    hull_area = ConvexHull(positions.reshape(-1, 2)).area
    target_area = 75 * 75  # Total array area in mm²
    return abs(hull_area - target_area) / target_area
```

## 2. Signal Processing Pipeline

### Artifact Rejection
```python
class TMSArtifactRejection:
    def __init__(self, sampling_rate=20000):
        self.fs = sampling_rate
        self.notch_freq = 2500  # TMS pulse frequency
        self.q_factor = 30
        
        # Design filters
        self.notch_b, self.notch_a = self.design_notch_filter()
        self.cleanup_b, self.cleanup_a = self.design_cleanup_filter()
    
    def design_notch_filter(self):
        w0 = self.notch_freq / (self.fs/2)
        q = self.q_factor
        b, a = signal.iirnotch(w0, q)
        return b, a
    
    def design_cleanup_filter(self):
        cutoff = 5000  # Hz
        nyquist = self.fs/2
        order = 4
        b, a = signal.butter(order, cutoff/nyquist, 'low')
        return b, a
    
    def process_signal(self, raw_signal):
        # Apply notch filter
        notch_filtered = signal.filtfilt(self.notch_b, self.notch_a, raw_signal)
        
        # Apply cleanup filter
        clean_signal = signal.filtfilt(self.cleanup_b, self.cleanup_a, 
                                     notch_filtered)
        
        return clean_signal
```

### Spike Detection
```python
class SpikeDetector:
    def __init__(self, threshold_std=4.0, dead_time=1.5e-3):
        self.threshold_std = threshold_std
        self.dead_time = dead_time
        self.fs = 20000  # Sampling rate in Hz
    
    def detect_spikes(self, signal):
        # Calculate threshold
        signal_std = np.std(signal)
        threshold = self.threshold_std * signal_std
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            signal,
            height=threshold,
            distance=int(self.dead_time * self.fs)
        )
        
        # Extract spike waveforms
        window = int(0.002 * self.fs)  # 2ms window
        waveforms = np.array([
            signal[max(0, p-window//2):min(len(signal), p+window//2)]
            for p in peaks
        ])
        
        return {
            'times': peaks / self.fs,
            'amplitudes': properties['peak_heights'],
            'waveforms': waveforms
        }
```
