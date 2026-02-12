import { parseOmnidreamBundle, } from "./omnidreamDataSchema.js";
export const DEFAULT_OMNIDREAM_FILE_MAP = {
    pipeline_summary: "pipeline_summary.json",
    ga_best: "ga_best.json",
    voltage_commands: "voltage_commands.json",
    basis_data: "web/basis_data.json",
    sensitivity_analysis: "web/sensitivity_analysis.json",
    cp_bridge_analysis: "web/cp_bridge_analysis.json",
    trajectory_result: "web/trajectory_result.json",
};
function mergeFileMap(overrides) {
    return {
        ...DEFAULT_OMNIDREAM_FILE_MAP,
        ...overrides,
    };
}
function trimTrailingSlash(value) {
    return value.replace(/[\\/]+$/, "");
}
function joinUrl(baseUrl, relativePath) {
    const normalizedBase = trimTrailingSlash(baseUrl);
    const normalizedPath = relativePath.replace(/^[\\/]+/, "");
    return `${normalizedBase}/${normalizedPath}`;
}
async function loadJsonViaFetch(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load ${url} (${response.status})`);
    }
    return response.json();
}
export async function loadOmnidreamBundleFromHttp(baseUrl, fileMapOverride) {
    const fileMap = mergeFileMap(fileMapOverride);
    const urls = {
        pipeline_summary: joinUrl(baseUrl, fileMap.pipeline_summary),
        ga_best: joinUrl(baseUrl, fileMap.ga_best),
        voltage_commands: joinUrl(baseUrl, fileMap.voltage_commands),
        basis_data: joinUrl(baseUrl, fileMap.basis_data),
        sensitivity_analysis: joinUrl(baseUrl, fileMap.sensitivity_analysis),
        cp_bridge_analysis: joinUrl(baseUrl, fileMap.cp_bridge_analysis),
        trajectory_result: joinUrl(baseUrl, fileMap.trajectory_result),
    };
    const [pipeline_summary, ga_best, voltage_commands, basis_data, sensitivity_analysis, cp_bridge_analysis, trajectory_result,] = await Promise.all([
        loadJsonViaFetch(urls.pipeline_summary),
        loadJsonViaFetch(urls.ga_best),
        loadJsonViaFetch(urls.voltage_commands),
        loadJsonViaFetch(urls.basis_data),
        loadJsonViaFetch(urls.sensitivity_analysis),
        loadJsonViaFetch(urls.cp_bridge_analysis),
        loadJsonViaFetch(urls.trajectory_result),
    ]);
    const rawBundle = {
        pipeline_summary,
        ga_best,
        voltage_commands,
        basis_data,
        sensitivity_analysis,
        cp_bridge_analysis,
        trajectory_result,
    };
    return parseOmnidreamBundle(rawBundle);
}
async function loadJsonFromFs(path) {
    const dynamicImport = new Function("modulePath", "return import(modulePath);");
    const fs = await dynamicImport("node:fs/promises");
    const raw = await fs.readFile(path, "utf8");
    return JSON.parse(raw);
}
function joinFsPath(baseDir, relativePath) {
    const normalizedBase = trimTrailingSlash(baseDir);
    const normalizedPath = relativePath.replace(/^[\\/]+/, "");
    return `${normalizedBase}/${normalizedPath}`;
}
export async function loadOmnidreamBundleFromFs(baseDir, fileMapOverride) {
    const fileMap = mergeFileMap(fileMapOverride);
    const paths = {
        pipeline_summary: joinFsPath(baseDir, fileMap.pipeline_summary),
        ga_best: joinFsPath(baseDir, fileMap.ga_best),
        voltage_commands: joinFsPath(baseDir, fileMap.voltage_commands),
        basis_data: joinFsPath(baseDir, fileMap.basis_data),
        sensitivity_analysis: joinFsPath(baseDir, fileMap.sensitivity_analysis),
        cp_bridge_analysis: joinFsPath(baseDir, fileMap.cp_bridge_analysis),
        trajectory_result: joinFsPath(baseDir, fileMap.trajectory_result),
    };
    const [pipeline_summary, ga_best, voltage_commands, basis_data, sensitivity_analysis, cp_bridge_analysis, trajectory_result,] = await Promise.all([
        loadJsonFromFs(paths.pipeline_summary),
        loadJsonFromFs(paths.ga_best),
        loadJsonFromFs(paths.voltage_commands),
        loadJsonFromFs(paths.basis_data),
        loadJsonFromFs(paths.sensitivity_analysis),
        loadJsonFromFs(paths.cp_bridge_analysis),
        loadJsonFromFs(paths.trajectory_result),
    ]);
    const rawBundle = {
        pipeline_summary,
        ga_best,
        voltage_commands,
        basis_data,
        sensitivity_analysis,
        cp_bridge_analysis,
        trajectory_result,
    };
    return parseOmnidreamBundle(rawBundle);
}
