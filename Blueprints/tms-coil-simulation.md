# TMS Coil Electromagnetic Simulation Specifications

## 1. Single C-Shaped Coil Parameters

### Physical Dimensions
```
Core dimensions:
- Length: 7.0 mm
- Width: 4.0 mm
- Height: 3.0 mm
- Core gap: 0.5 mm

Wire specifications:
- Wire diameter: 0.1 mm (including insulation)
- Number of turns: 30
- Winding layers: 2
- Turn spacing: 0.05 mm
```

### Material Properties
```
Iron powder core:
- Relative permeability (μr): 5000
- Electrical conductivity: 1.12×10^6 S/m
- Mass density: 7.874×10^3 kg/m³
- Heat capacity: 440 J/(kg·K)
- Thermal conductivity: 76.2 W/(m·K)

Copper wire:
- Electrical conductivity: 5.96×10^7 S/m
- Relative permeability: 0.999994
- Mass density: 8.96×10^3 kg/m³
- Heat capacity: 385 J/(kg·K)
- Thermal conductivity: 400 W/(m·K)
```

### COMSOL Implementation
```
1. Geometry Creation:
model.component("comp1").geom("geom1").create("c_core", "Block");
model.component("comp1").geom("geom1").feature("c_core").set("size", [0.007, 0.004, 0.003]);
model.component("comp1").geom("geom1").feature("c_core").set("gap", 0.0005);

2. Physics Settings:
model.component("comp1").physics("mf").create("coil1", "Coil");
model.component("comp1").physics("mf").feature("coil1").set("Nc", "30");
model.component("comp1").physics("mf").feature("coil1").set("Vc", "5[V]");

3. Mesh Settings:
model.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
model.component("comp1").mesh("mesh1").feature("ftri1").set("hmax", "0.0001");
model.component("comp1").mesh("mesh1").feature("ftri1").set("hgrad", "1.2");
```

## 2. Field Analysis Settings

### Simulation Parameters
```
Time-dependent analysis:
- Pulse duration: 100 μs
- Rise time: 10 μs
- Fall time: 30 μs
- Current amplitude: 0-5 A
- Frequency: 2.5 kHz

Study steps:
1. Stationary (initial conditions)
2. Time dependent (0-200 μs, 1 μs steps)
```

### Field Measurement Points
```
Measurement grid:
- X range: -10 to 10 mm, 0.5 mm steps
- Y range: -10 to 10 mm, 0.5 mm steps
- Z range: 0 to 20 mm, 1 mm steps

Key measurement locations:
1. Core surface (z = 0 mm)
2. Target depth (z = 10 mm)
3. Maximum field point (determined by simulation)
```

## 3. Grid Configuration

### Array Layout
```
Grid specifications:
- 4x4 array
- Center-to-center spacing: 25 mm
- Total array dimensions: 75 mm x 75 mm
- Coil orientation: Alternating 90° rotation
```

### Field Interaction Analysis
```matlab
% MATLAB code for field interaction analysis
function [total_field, interference] = analyze_grid_fields(coil_positions, individual_fields)
    total_field = zeros(size(individual_fields{1}));
    
    % Superpose fields
    for i = 1:length(coil_positions)
        translated_field = translate_field(individual_fields{i}, coil_positions(i,:));
        total_field = total_field + translated_field;
    end
    
    % Calculate interference metric
    interference = calculate_interference(total_field, individual_fields);
end

function interference = calculate_interference(total_field, individual_fields)
    ideal_sum = sum(cat(4, individual_fields{:}), 4);
    interference = norm(total_field - ideal_sum) / norm(ideal_sum);
end
```
