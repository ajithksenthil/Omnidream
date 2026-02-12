function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}
function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
}
function isInteger(value) {
    return typeof value === "number" && Number.isInteger(value);
}
function isBoolean(value) {
    return typeof value === "boolean";
}
function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function isNumberArray(value) {
    return Array.isArray(value) && value.every(isFiniteNumber);
}
function isBooleanArray(value) {
    return Array.isArray(value) && value.every(isBoolean);
}
function isNumberMatrix(value) {
    return Array.isArray(value) && value.every(isNumberArray);
}
function isBooleanMatrix(value) {
    return Array.isArray(value) && value.every(isBooleanArray);
}
function expectKeys(obj, keys, context) {
    for (const key of keys) {
        assert(key in obj, `${context} missing key: ${key}`);
    }
}
function parseGABest(input, context) {
    assert(isRecord(input), `${context} must be an object`);
    expectKeys(input, [
        "mode",
        "fitness",
        "num_coils",
        "generations",
        "positions",
        "orientations",
        "amplitudes",
        "group",
        "fire_times",
        "freq_carrier",
        "delta_freq",
    ], context);
    assert(typeof input.mode === "string", `${context}.mode must be string`);
    assert(isFiniteNumber(input.fitness), `${context}.fitness must be number`);
    assert(isInteger(input.num_coils), `${context}.num_coils must be integer`);
    assert(isInteger(input.generations), `${context}.generations must be integer`);
    assert(isNumberMatrix(input.positions), `${context}.positions must be matrix`);
    assert(isNumberMatrix(input.orientations), `${context}.orientations must be matrix`);
    assert(isNumberArray(input.amplitudes), `${context}.amplitudes must be number[]`);
    assert(isNumberArray(input.group), `${context}.group must be number[]`);
    assert(isNumberArray(input.fire_times), `${context}.fire_times must be number[]`);
    assert(isFiniteNumber(input.freq_carrier), `${context}.freq_carrier must be number`);
    assert(isFiniteNumber(input.delta_freq), `${context}.delta_freq must be number`);
    return {
        mode: input.mode,
        fitness: input.fitness,
        num_coils: input.num_coils,
        generations: input.generations,
        positions: input.positions,
        orientations: input.orientations,
        amplitudes: input.amplitudes,
        group: input.group,
        fire_times: input.fire_times,
        freq_carrier: input.freq_carrier,
        delta_freq: input.delta_freq,
    };
}
export function parsePipelineSummary(input) {
    const context = "pipeline_summary";
    assert(isRecord(input), `${context} must be an object`);
    expectKeys(input, ["mode", "synthetic", "num_coils", "stages_run", "elapsed_seconds", "ga_result"], context);
    assert(typeof input.mode === "string", `${context}.mode must be string`);
    assert(typeof input.synthetic === "boolean", `${context}.synthetic must be boolean`);
    assert(isInteger(input.num_coils), `${context}.num_coils must be integer`);
    assert(isNumberArray(input.stages_run), `${context}.stages_run must be number[]`);
    assert(isFiniteNumber(input.elapsed_seconds), `${context}.elapsed_seconds must be number`);
    const ga_result = parseGABest(input.ga_result, `${context}.ga_result`);
    return {
        mode: input.mode,
        synthetic: input.synthetic,
        num_coils: input.num_coils,
        stages_run: input.stages_run,
        elapsed_seconds: input.elapsed_seconds,
        ga_result,
    };
}
export function parseGaBest(input) {
    return parseGABest(input, "ga_best");
}
export function parseVoltageCommands(input) {
    const context = "voltage_commands";
    assert(isRecord(input), `${context} must be an object`);
    expectKeys(input, ["I_desired", "V_command_real", "V_command_imag", "freq_hz"], context);
    assert(isNumberArray(input.I_desired), `${context}.I_desired must be number[]`);
    assert(isNumberArray(input.V_command_real), `${context}.V_command_real must be number[]`);
    assert(isNumberArray(input.V_command_imag), `${context}.V_command_imag must be number[]`);
    assert(isFiniteNumber(input.freq_hz), `${context}.freq_hz must be number`);
    return {
        I_desired: input.I_desired,
        V_command_real: input.V_command_real,
        V_command_imag: input.V_command_imag,
        freq_hz: input.freq_hz,
    };
}
export function parseBasisData(input) {
    const context = "basis_data";
    assert(isRecord(input), `${context} must be an object`);
    expectKeys(input, ["basis_matrix", "target_idx", "surface_indices"], context);
    assert(isNumberMatrix(input.basis_matrix), `${context}.basis_matrix must be number[][]`);
    assert(isNumberArray(input.target_idx), `${context}.target_idx must be number[]`);
    assert(isNumberArray(input.surface_indices), `${context}.surface_indices must be number[]`);
    return {
        basis_matrix: input.basis_matrix,
        target_idx: input.target_idx,
        surface_indices: input.surface_indices,
    };
}
export function parseSensitivityAnalysis(input) {
    const context = "sensitivity_analysis";
    assert(isRecord(input), `${context} must be an object`);
    expectKeys(input, [
        "jacobian_nts__J_V_target_alpha",
        "jacobian_nts__J_V_target_time",
        "jacobian_nts__J_V_surface_alpha",
        "jacobian_nts__J_V_surface_time",
        "jacobian_nts__J_alpha",
        "jacobian_nts__J_time",
        "jacobian_nts_analytical__J_V_target_alpha",
        "jacobian_nts_analytical__J_V_target_time",
        "hessian_nts",
        "reachable_nts__V_target",
        "reachable_nts__V_surface_max",
        "reachable_nts__amplitudes",
        "reachable_nts__hull_vertices",
        "pareto_nts__lambdas",
        "pareto_nts__V_target",
        "pareto_nts__V_surface_max",
        "pareto_nts__amplitudes",
        "pareto_nts__is_dominated",
    ], context);
    assert(isNumberArray(input.jacobian_nts__J_V_target_alpha), `${context}.jacobian_nts__J_V_target_alpha must be number[]`);
    assert(isNumberArray(input.jacobian_nts__J_V_target_time), `${context}.jacobian_nts__J_V_target_time must be number[]`);
    assert(isNumberArray(input.jacobian_nts__J_V_surface_alpha), `${context}.jacobian_nts__J_V_surface_alpha must be number[]`);
    assert(isNumberArray(input.jacobian_nts__J_V_surface_time), `${context}.jacobian_nts__J_V_surface_time must be number[]`);
    assert(isNumberMatrix(input.jacobian_nts__J_alpha), `${context}.jacobian_nts__J_alpha must be number[][]`);
    assert(isNumberMatrix(input.jacobian_nts__J_time), `${context}.jacobian_nts__J_time must be number[][]`);
    assert(isNumberArray(input.jacobian_nts_analytical__J_V_target_alpha), `${context}.jacobian_nts_analytical__J_V_target_alpha must be number[]`);
    assert(isNumberArray(input.jacobian_nts_analytical__J_V_target_time), `${context}.jacobian_nts_analytical__J_V_target_time must be number[]`);
    assert(isNumberMatrix(input.hessian_nts), `${context}.hessian_nts must be number[][]`);
    assert(isNumberArray(input.reachable_nts__V_target), `${context}.reachable_nts__V_target must be number[]`);
    assert(isNumberArray(input.reachable_nts__V_surface_max), `${context}.reachable_nts__V_surface_max must be number[]`);
    assert(isNumberMatrix(input.reachable_nts__amplitudes), `${context}.reachable_nts__amplitudes must be number[][]`);
    assert(isNumberMatrix(input.reachable_nts__hull_vertices), `${context}.reachable_nts__hull_vertices must be number[][]`);
    assert(isNumberArray(input.pareto_nts__lambdas), `${context}.pareto_nts__lambdas must be number[]`);
    assert(isNumberArray(input.pareto_nts__V_target), `${context}.pareto_nts__V_target must be number[]`);
    assert(isNumberArray(input.pareto_nts__V_surface_max), `${context}.pareto_nts__V_surface_max must be number[]`);
    assert(isNumberMatrix(input.pareto_nts__amplitudes), `${context}.pareto_nts__amplitudes must be number[][]`);
    assert(isBooleanArray(input.pareto_nts__is_dominated), `${context}.pareto_nts__is_dominated must be boolean[]`);
    return {
        jacobian_nts__J_V_target_alpha: input.jacobian_nts__J_V_target_alpha,
        jacobian_nts__J_V_target_time: input.jacobian_nts__J_V_target_time,
        jacobian_nts__J_V_surface_alpha: input.jacobian_nts__J_V_surface_alpha,
        jacobian_nts__J_V_surface_time: input.jacobian_nts__J_V_surface_time,
        jacobian_nts__J_alpha: input.jacobian_nts__J_alpha,
        jacobian_nts__J_time: input.jacobian_nts__J_time,
        jacobian_nts_analytical__J_V_target_alpha: input.jacobian_nts_analytical__J_V_target_alpha,
        jacobian_nts_analytical__J_V_target_time: input.jacobian_nts_analytical__J_V_target_time,
        hessian_nts: input.hessian_nts,
        reachable_nts__V_target: input.reachable_nts__V_target,
        reachable_nts__V_surface_max: input.reachable_nts__V_surface_max,
        reachable_nts__amplitudes: input.reachable_nts__amplitudes,
        reachable_nts__hull_vertices: input.reachable_nts__hull_vertices,
        pareto_nts__lambdas: input.pareto_nts__lambdas,
        pareto_nts__V_target: input.pareto_nts__V_target,
        pareto_nts__V_surface_max: input.pareto_nts__V_surface_max,
        pareto_nts__amplitudes: input.pareto_nts__amplitudes,
        pareto_nts__is_dominated: input.pareto_nts__is_dominated,
    };
}
export function parseCPBridgeAnalysis(input) {
    const context = "cp_bridge_analysis";
    assert(isRecord(input), `${context} must be an object`);
    expectKeys(input, [
        "n_coils",
        "te_matrix",
        "attachment_matrix",
        "mutual_attachment",
        "agent_free_energies",
        "collective_phi",
        "sync_order_parameter",
        "group_free_energy",
        "morl_objectives__J_phi",
        "morl_objectives__J_arch",
        "morl_objectives__J_sync",
        "morl_objectives__J_task",
        "energy__E_nca",
        "energy__E_mf",
        "energy__E_arch",
        "energy__E_phi",
        "energy__E_couple",
        "energy__E_morl",
        "energy__E_total",
        "energy__phi_collective",
        "eta_field",
        "eta_field_post_implacement",
        "delta_phi_implacement",
        "n_worlds",
        "world_probabilities",
        "world_coherences",
        "world_energies",
        "world_amplitudes",
        "world_outputs",
        "group",
    ], context);
    assert(isNumberArray(input.n_coils), `${context}.n_coils must be number[]`);
    assert(isNumberMatrix(input.te_matrix), `${context}.te_matrix must be number[][]`);
    assert(isBooleanMatrix(input.attachment_matrix), `${context}.attachment_matrix must be boolean[][]`);
    assert(isNumberMatrix(input.mutual_attachment), `${context}.mutual_attachment must be number[][]`);
    assert(isNumberArray(input.agent_free_energies), `${context}.agent_free_energies must be number[]`);
    assert(isNumberArray(input.collective_phi), `${context}.collective_phi must be number[]`);
    assert(isNumberArray(input.sync_order_parameter), `${context}.sync_order_parameter must be number[]`);
    assert(isNumberArray(input.group_free_energy), `${context}.group_free_energy must be number[]`);
    assert(isNumberArray(input.morl_objectives__J_phi), `${context}.morl_objectives__J_phi must be number[]`);
    assert(isNumberArray(input.morl_objectives__J_arch), `${context}.morl_objectives__J_arch must be number[]`);
    assert(isNumberArray(input.morl_objectives__J_sync), `${context}.morl_objectives__J_sync must be number[]`);
    assert(isNumberArray(input.morl_objectives__J_task), `${context}.morl_objectives__J_task must be number[]`);
    assert(isNumberArray(input.energy__E_nca), `${context}.energy__E_nca must be number[]`);
    assert(isNumberArray(input.energy__E_mf), `${context}.energy__E_mf must be number[]`);
    assert(isNumberArray(input.energy__E_arch), `${context}.energy__E_arch must be number[]`);
    assert(isNumberArray(input.energy__E_phi), `${context}.energy__E_phi must be number[]`);
    assert(isNumberArray(input.energy__E_couple), `${context}.energy__E_couple must be number[]`);
    assert(isNumberArray(input.energy__E_morl), `${context}.energy__E_morl must be number[]`);
    assert(isNumberArray(input.energy__E_total), `${context}.energy__E_total must be number[]`);
    assert(isNumberArray(input.energy__phi_collective), `${context}.energy__phi_collective must be number[]`);
    assert(isNumberArray(input.eta_field), `${context}.eta_field must be number[]`);
    assert(isNumberArray(input.eta_field_post_implacement), `${context}.eta_field_post_implacement must be number[]`);
    assert(isNumberArray(input.delta_phi_implacement), `${context}.delta_phi_implacement must be number[]`);
    assert(isNumberArray(input.n_worlds), `${context}.n_worlds must be number[]`);
    assert(isNumberArray(input.world_probabilities), `${context}.world_probabilities must be number[]`);
    assert(isNumberArray(input.world_coherences), `${context}.world_coherences must be number[]`);
    assert(isNumberArray(input.world_energies), `${context}.world_energies must be number[]`);
    assert(isNumberMatrix(input.world_amplitudes), `${context}.world_amplitudes must be number[][]`);
    assert(isNumberMatrix(input.world_outputs), `${context}.world_outputs must be number[][]`);
    assert(isNumberArray(input.group), `${context}.group must be number[]`);
    if ("morl__J_phi" in input) {
        assert(isNumberArray(input.morl__J_phi), `${context}.morl__J_phi must be number[]`);
    }
    if ("morl__J_arch" in input) {
        assert(isNumberArray(input.morl__J_arch), `${context}.morl__J_arch must be number[]`);
    }
    if ("morl__J_sync" in input) {
        assert(isNumberArray(input.morl__J_sync), `${context}.morl__J_sync must be number[]`);
    }
    if ("morl__J_task" in input) {
        assert(isNumberArray(input.morl__J_task), `${context}.morl__J_task must be number[]`);
    }
    const morlJPhi = "morl__J_phi" in input ? input.morl__J_phi : undefined;
    const morlJArch = "morl__J_arch" in input ? input.morl__J_arch : undefined;
    const morlJSync = "morl__J_sync" in input ? input.morl__J_sync : undefined;
    const morlJTask = "morl__J_task" in input ? input.morl__J_task : undefined;
    return {
        n_coils: input.n_coils,
        te_matrix: input.te_matrix,
        attachment_matrix: input.attachment_matrix,
        mutual_attachment: input.mutual_attachment,
        agent_free_energies: input.agent_free_energies,
        collective_phi: input.collective_phi,
        sync_order_parameter: input.sync_order_parameter,
        group_free_energy: input.group_free_energy,
        morl_objectives__J_phi: input.morl_objectives__J_phi,
        morl_objectives__J_arch: input.morl_objectives__J_arch,
        morl_objectives__J_sync: input.morl_objectives__J_sync,
        morl_objectives__J_task: input.morl_objectives__J_task,
        energy__E_nca: input.energy__E_nca,
        energy__E_mf: input.energy__E_mf,
        energy__E_arch: input.energy__E_arch,
        energy__E_phi: input.energy__E_phi,
        energy__E_couple: input.energy__E_couple,
        energy__E_morl: input.energy__E_morl,
        energy__E_total: input.energy__E_total,
        energy__phi_collective: input.energy__phi_collective,
        eta_field: input.eta_field,
        eta_field_post_implacement: input.eta_field_post_implacement,
        delta_phi_implacement: input.delta_phi_implacement,
        n_worlds: input.n_worlds,
        world_probabilities: input.world_probabilities,
        world_coherences: input.world_coherences,
        world_energies: input.world_energies,
        world_amplitudes: input.world_amplitudes,
        world_outputs: input.world_outputs,
        group: input.group,
        morl__J_phi: morlJPhi,
        morl__J_arch: morlJArch,
        morl__J_sync: morlJSync,
        morl__J_task: morlJTask,
    };
}
export function parseTrajectoryResult(input) {
    const context = "trajectory_result";
    assert(isRecord(input), `${context} must be an object`);
    expectKeys(input, ["time", "amplitudes", "outputs", "energy", "phi", "sync", "waypoint_indices"], context);
    assert(isNumberArray(input.time), `${context}.time must be number[]`);
    assert(isNumberMatrix(input.amplitudes), `${context}.amplitudes must be number[][]`);
    assert(isNumberMatrix(input.outputs), `${context}.outputs must be number[][]`);
    assert(isNumberArray(input.energy), `${context}.energy must be number[]`);
    assert(isNumberArray(input.phi), `${context}.phi must be number[]`);
    assert(isNumberArray(input.sync), `${context}.sync must be number[]`);
    if ("h_cog" in input) {
        assert(isNumberArray(input.h_cog), `${context}.h_cog must be number[]`);
    }
    assert(isNumberArray(input.waypoint_indices), `${context}.waypoint_indices must be number[]`);
    const hCog = "h_cog" in input ? input.h_cog : undefined;
    return {
        time: input.time,
        amplitudes: input.amplitudes,
        outputs: input.outputs,
        energy: input.energy,
        phi: input.phi,
        sync: input.sync,
        h_cog: hCog,
        waypoint_indices: input.waypoint_indices,
    };
}
export function parseOmnidreamBundle(raw) {
    const bundle = {
        pipeline_summary: parsePipelineSummary(raw.pipeline_summary),
        ga_best: parseGaBest(raw.ga_best),
        voltage_commands: parseVoltageCommands(raw.voltage_commands),
        basis_data: parseBasisData(raw.basis_data),
        sensitivity_analysis: parseSensitivityAnalysis(raw.sensitivity_analysis),
        cp_bridge_analysis: parseCPBridgeAnalysis(raw.cp_bridge_analysis),
        trajectory_result: parseTrajectoryResult(raw.trajectory_result),
    };
    validateBundleConsistency(bundle);
    return bundle;
}
export function scalarFromSingleton(values, fieldName) {
    assert(values.length === 1, `${fieldName} must contain exactly one value`);
    return values[0];
}
export function validateBundleConsistency(bundle) {
    const nCoils = bundle.pipeline_summary.num_coils;
    assert(bundle.pipeline_summary.ga_result.num_coils === bundle.ga_best.num_coils, "pipeline_summary.ga_result.num_coils mismatch");
    assert(bundle.pipeline_summary.ga_result.mode === bundle.ga_best.mode, "pipeline_summary.ga_result.mode mismatch");
    assert(Math.abs(bundle.pipeline_summary.ga_result.fitness - bundle.ga_best.fitness) < 1e-9, "pipeline_summary.ga_result.fitness mismatch");
    assert(bundle.ga_best.num_coils === nCoils, "ga_best.num_coils mismatch");
    assert(bundle.ga_best.positions.length === nCoils, "ga_best.positions length mismatch");
    assert(bundle.ga_best.orientations.length === nCoils, "ga_best.orientations length mismatch");
    assert(bundle.ga_best.amplitudes.length === nCoils, "ga_best.amplitudes length mismatch");
    assert(bundle.ga_best.group.length === nCoils, "ga_best.group length mismatch");
    assert(bundle.ga_best.fire_times.length === nCoils, "ga_best.fire_times length mismatch");
    assert(bundle.voltage_commands.I_desired.length === nCoils, "voltage_commands.I_desired length mismatch");
    assert(bundle.voltage_commands.V_command_real.length === nCoils, "voltage_commands.V_command_real length mismatch");
    assert(bundle.voltage_commands.V_command_imag.length === nCoils, "voltage_commands.V_command_imag length mismatch");
    for (const row of bundle.basis_data.basis_matrix) {
        assert(row.length === nCoils, "basis_matrix row width mismatch");
    }
    for (const row of bundle.sensitivity_analysis.pareto_nts__amplitudes) {
        assert(row.length === nCoils, "pareto_nts__amplitudes row width mismatch");
    }
    for (const row of bundle.sensitivity_analysis.reachable_nts__amplitudes) {
        assert(row.length === nCoils, "reachable_nts__amplitudes row width mismatch");
    }
    const cpN = scalarFromSingleton(bundle.cp_bridge_analysis.n_coils, "n_coils");
    assert(cpN === nCoils, "cp_bridge_analysis.n_coils mismatch");
    const nWorlds = scalarFromSingleton(bundle.cp_bridge_analysis.n_worlds, "n_worlds");
    assert(bundle.cp_bridge_analysis.world_probabilities.length === nWorlds, "world_probabilities length mismatch");
    assert(bundle.cp_bridge_analysis.world_coherences.length === nWorlds, "world_coherences length mismatch");
    assert(bundle.cp_bridge_analysis.world_energies.length === nWorlds, "world_energies length mismatch");
    assert(bundle.cp_bridge_analysis.world_amplitudes.length === nWorlds, "world_amplitudes row count mismatch");
    assert(bundle.cp_bridge_analysis.world_outputs.length === nWorlds, "world_outputs row count mismatch");
    for (const row of bundle.cp_bridge_analysis.te_matrix) {
        assert(row.length === nCoils, "te_matrix width mismatch");
    }
    for (const row of bundle.cp_bridge_analysis.world_amplitudes) {
        assert(row.length === nCoils, "world_amplitudes row width mismatch");
    }
    for (const row of bundle.cp_bridge_analysis.world_outputs) {
        assert(row.length === 5, "world_outputs row width must be 5");
    }
    const tCount = bundle.trajectory_result.time.length;
    assert(bundle.trajectory_result.amplitudes.length === tCount, "trajectory amplitudes/time length mismatch");
    assert(bundle.trajectory_result.outputs.length === tCount, "trajectory outputs/time length mismatch");
    assert(bundle.trajectory_result.energy.length === tCount, "trajectory energy/time length mismatch");
    assert(bundle.trajectory_result.phi.length === tCount, "trajectory phi/time length mismatch");
    assert(bundle.trajectory_result.sync.length === tCount, "trajectory sync/time length mismatch");
    if (bundle.trajectory_result.h_cog) {
        assert(bundle.trajectory_result.h_cog.length === tCount, "trajectory h_cog/time length mismatch");
    }
    for (const row of bundle.trajectory_result.amplitudes) {
        assert(row.length === nCoils, "trajectory amplitudes row width mismatch");
    }
    for (const row of bundle.trajectory_result.outputs) {
        assert(row.length === 5, "trajectory outputs row width must be 5");
    }
}
