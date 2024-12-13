# Control System Specifications

## 1. Hardware Architecture

### FPGA Specifications
```
Device: Xilinx Artix-7 XC7A200T
- Speed grade: -1
- Package: FBG484
- Temperature grade: I
- Operating frequency: 100 MHz

Resource requirements:
- Flip-flops: ~20,000
- LUTs: ~15,000
- Block RAM: 50 blocks
- DSP slices: 20
```

### ADC Interface
```verilog
module adc_interface (
    input wire clk,
    input wire rst_n,
    input wire [15:0] adc_data,
    input wire adc_valid,
    output reg [15:0] processed_data,
    output reg data_valid
);

    // Timing parameters
    parameter SAMPLE_RATE = 20000;  // 20 kHz
    parameter CLK_DIV = 5000;       // 100 MHz / 20 kHz

    // Sample counter
    reg [12:0] sample_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_count <= 0;
            processed_data <= 0;
            data_valid <= 0;
        end else begin
            if (sample_count == CLK_DIV - 1) begin
                sample_count <= 0;
                processed_data <= adc_data;
                data_valid <= adc_valid;
            end else begin
                sample_count <= sample_count + 1;
                data_valid <= 0;
            end
        end
    end

endmodule
```

## 2. Stimulation Control

### Timing Controller
```verilog
module stim_timing_controller (
    input wire clk,
    input wire rst_n,
    input wire [7:0] coil_select,
    input wire [15:0] pulse_width,
    input wire [15:0] inter_pulse_interval,
    output reg [15:0] coil_drive
);

    // State definitions
    parameter IDLE = 2'b00;
    parameter PULSE_ACTIVE = 2'b01;
    parameter INTER_PULSE = 2'b10;

    reg [1:0] state;
    reg [15:0] timer;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            timer <= 0;
            coil_drive <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (coil_select != 0) begin
                        state <= PULSE_ACTIVE;
                        coil_drive <= coil_select;
                        timer <= pulse_width;
                    end
                end

                PULSE_ACTIVE: begin
                    if (timer == 0) begin
                        state <= INTER_PULSE;
                        coil_drive <= 0;
                        timer <= inter_pulse_interval;
                    end else begin
                        timer <= timer - 1;
                    end
                end

                INTER_PULSE: begin
                    if (timer == 0) begin
                        state <= IDLE;
                    end else begin
                        timer <= timer - 1;
                    end
                end
            endcase
        end
    end

endmodule
```

### Safety Monitor
```verilog
module safety_monitor (
    input wire clk,
    input wire rst_n,
    input wire [15:0] temperature [0:15],
    input wire [15:0] current [0:15],
    output reg safety_shutdown,
    output reg [15:0] status
);

    // Safety thresholds
    parameter TEMP_THRESHOLD = 16'd45;  // 45°C
    parameter CURRENT_THRESHOLD = 16'd5000;  // 5A

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            safety_shutdown <= 0;
            status <= 0;
        end else begin
            // Check temperature limits
            for (int i = 0; i < 16; i++) begin
                if (temperature[i] > TEMP_THRESHOLD) begin
                    safety_shutdown <= 1;
                    status[i] <= 1;
                end
            end

            // Check current limits
            for (int i = 0; i < 16; i++) begin
                if (current[i] > CURRENT_THRESHOLD) begin
                    safety_shutdown <= 1;
                    status[i] <= 1;
                end
            end
        end
    end

endmodule
```

## 3. Software Control Interface

### Main Control Class
```python
class TMSGridController:
    def __init__(self):
        self.fpga = FPGAInterface()
        self.recording = RecordingSystem()
        self.safety = SafetyMonitor()
        
        # Initialize coil array
        self.coils = [[Coil(x, y) for x in range(4)] for y in range(4)]
        
    def setup_stimulation_pattern(self, pattern):
        """
        pattern: dict with keys:
            'coils': list of (x,y) coordinates
            'timing': list of (start_time, duration) pairs
            'amplitude': list of amplitudes
        """
        # Validate pattern
        if not self._validate_pattern(pattern):
            raise ValueError("Invalid stimulation pattern")
            
        # Configure FPGA
        self.fpga.configure_pattern(pattern)
        
    def start_stimulation(self):
        """Start the stimulation sequence"""
        if not self.safety.check_status():
            raise SafetyError("Safety check failed")
            
        self.fpga.start_sequence()
        self.recording.start_recording()
        
    def stop_stimulation(self):
        """Emergency stop of stimulation"""
        self.fpga.emergency_stop()
        self.recording.stop_recording()
        
    def _validate_pattern(self, pattern):
        """Validate stimulation pattern parameters"""
        for coil in pattern['coils']:
            if not (0 <= coil[0] < 4 and 0 <= coil[1]