import { createContext, useContext, useState, useMemo, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, ScatterChart, Scatter, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, BarChart, Bar,
  ResponsiveContainer, ReferenceLine, ZAxis
} from "recharts";
import { Activity, Layers, GitBranch, Network, Zap, Brain } from "lucide-react";
import { loadOmnidreamBundleFromHttp } from "./monitoring/frontend/omnidreamDataLoader.js";

// ============================================================
// §1  DATA CONSTANTS — Real pipeline output from Omnidream
// ============================================================

const COLORS = {
  bg: "#0a0e27", card: "#1a1f3a", border: "#2a2f4a",
  text: "#e0e0ff", textDim: "#a0a0c0",
  cyan: "#00d9ff", magenta: "#ff00ff", gold: "#ffd700",
  lime: "#00ff41", purple: "#9d4edd", orange: "#ff9500",
  red: "#ff4444", teal: "#00bfa5",
};

const GA_HISTORY = [
  { gen: 0, best: 44.878, median: 54.074 },
  { gen: 1, best: 43.339, median: 49.322 },
  { gen: 2, best: 38.426, median: 43.261 },
  { gen: 3, best: 38.072, median: 40.082 },
  { gen: 4, best: 37.159, median: 38.466 },
];

const GA_AMPLITUDES = [0.487,1.633,1.145,1.691,1.561,0.788,0.515,0.529,0.906,1.159,0.504,0.870,0.769,0.521,1.408,0.945];
const GROUP = [1,1,1,1,0,0,1,0,0,1,1,0,1,0,1,1]; // 1=M, 0=F from ga_best
const CP_GROUP = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]; // from cp_bridge analysis

const POSITIONS = [
  [26.49,-73.47,-44.73],[56.87,66.21,21.97],[53.07,-72.67,1.70],[-82.46,34.86,9.27],
  [45.72,-67.30,38.48],[-11.98,89.09,-4.44],[20.41,51.31,71.07],[82.54,25.04,25.71],
  [-83.40,-12.36,-31.49],[70.15,24.95,50.57],[89.23,-7.94,8.70],[-64.26,-47.64,-41.25],
  [-55.31,-5.97,70.75],[16.50,88.47,-0.33],[44.71,-71.01,-32.54],[84.34,1.89,31.36]
];

const CP_METRICS = {
  phi: 1.0000142, sync: 1.0, groupF2: 0.2592,
  morl: { J_phi: 1.0, J_arch: 12.376, J_sync: 1.0, J_task: 0.0898 },
  energy: { E_nca: 0.966, E_mf: 0.101, E_arch: 12.376, E_phi: -2.0, E_couple: 0, E_morl: 10.286, E_total: 21.729 },
};

const AGENT_FE = [4.041e-5,4.041e-5,4.041e-5,4.041e-5,4.042e-5,4.041e-5,4.042e-5,4.042e-5,4.041e-5,4.041e-5,4.041e-5,4.042e-5,4.042e-5,4.041e-5,4.042e-5,4.042e-5];
const ETA_BEFORE = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0];
const ETA_AFTER = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0];

// Trajectory: 59 timesteps through 3 waypoints (worlds 0, 24, 49)
const TRAJ_ENERGY = [2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,3.089,3.928,4.846,5.845,6.924,8.083,9.322,10.642,12.044,13.527,15.090,16.733,18.458,20.265,22.153,24.121,26.170,28.300,30.511,32.802,35.173,37.625,40.158,42.772,45.466,48.241,51.097,54.033,57.049];
const TRAJ_PHI = new Array(59).fill(1.0000142);
const TRAJ_SYNC = new Array(59).fill(1.0);
const TRAJ_VTARGET = [0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0343,0.0383,0.0422,0.0461,0.0500,0.0539,0.0578,0.0617,0.0656,0.0695,0.0734,0.0773,0.0813,0.0852,0.0891,0.0930,0.0969,0.1008,0.1047,0.1086,0.1125,0.1164,0.1204,0.1243,0.1282,0.1321,0.1360,0.1399,0.1438];
const TRAJ_VSURF = [1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.946,2.088,2.229,2.371,2.512,2.653,2.795,2.936,3.078,3.220,3.362,3.503,3.645,3.787,3.929,4.071,4.213,4.355,4.497,4.639,4.780,4.922,5.064,5.206,5.348,5.490,5.632,5.774,5.916];

// Amplitude heatmap: 59 timesteps x 16 coils (key frames, interpolated)
const TRAJ_AMPS_START = [0.178,0.613,0.014,0.928,0.265,0.040,0.108,0.240,0.274,0.033,0.279,0.193,0.379,0.385,0.339,0.328];
const TRAJ_AMPS_END = [0.717,0.760,0.575,0.863,0.291,0.882,0.729,0.799,0.665,0.846,0.993,0.420,0.814,0.745,0.453,0.927];

// Many worlds: 50 worlds
const WORLD_ENERGIES = [2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,2.331,11.097,57.049];
const WORLD_PROBS = [0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,0.0208,3.25e-6,3.59e-26];

// Sensitivity: Jacobians
const J_ALPHA = [[0.0215,0.0,0.0077,0.0047,0.0013,0.0481,0.0035,0.0150,0.0127,0.0263,0.0108,0.0063,0.0000,0.0021,0.0027,0.0170],[0.7942,0.1826,0.4563,0.3640,0.2342,1.0151,0.3128,0.6337,0.5399,0.8590,0.5064,0.4028,0.2125,0.2582,0.2930,0.7317]];
const J_TIME = [[3.587,0.0,1.275,0.790,0.211,-21.934,0.582,2.501,2.123,4.378,1.793,1.056,0.006,0.347,0.448,2.836],[132.364,30.431,76.043,60.666,39.025,-1130.213,52.138,105.615,89.989,143.164,84.397,67.130,35.418,43.040,48.837,121.957]];

const PARETO_VT = [0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0304,0.0872,0.1438];
const PARETO_VS = [1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,1.805,2.996,5.916];
const PARETO_LAMBDAS = [0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99];

const REACHABLE = [[0.118,4.752],[0.096,4.453],[0.090,3.766],[0.102,4.078],[0.083,3.522],[0.069,3.267],[0.105,4.249],[0.091,3.612],[0.075,3.587],[0.071,3.600],[0.076,3.396],[0.109,4.576],[0.107,3.846],[0.100,4.185],[0.068,3.424],[0.115,4.599],[0.087,2.996],[0.066,2.949],[0.097,4.608],[0.076,3.590],[0.105,4.512],[0.113,4.383],[0.100,4.146],[0.077,3.736],[0.101,4.126],[0.115,4.517],[0.066,3.444],[0.091,3.995],[0.063,3.238],[0.075,3.382],[0.086,3.737],[0.120,4.725],[0.100,3.904],[0.098,4.287],[0.071,2.986],[0.088,3.936],[0.103,4.194],[0.098,4.385],[0.116,4.637],[0.112,4.596],[0.095,3.849],[0.119,4.464],[0.086,3.777],[0.100,4.337],[0.115,5.030],[0.056,2.562],[0.051,2.675],[0.079,2.882],[0.101,4.157],[0.089,3.633]];

const TE_MATRIX = [
  [0,2.77e-10,6.16e-9,4.39e-11,2.33e-10,7.32e-11,1.65e-9,8.05e-10,4.53e-11,3.12e-11,1.20e-10,1.21e-8,6.51e-10,5.26e-7,7.45e-10,4.49e-9],
  [2.77e-10,0,2.82e-10,5.31e-11,7.05e-9,7.45e-8,1.28e-10,3.71e-10,4.56e-9,1.28e-10,5.92e-10,9.12e-10,3.72e-9,3.85e-10,2.19e-10,3.57e-10],
  [6.16e-9,2.82e-10,0,1.08e-10,6.49e-11,1.03e-10,1.17e-6,7.36e-11,9.34e-11,7.44e-11,3.69e-11,2.69e-10,9.32e-11,8.12e-10,4.42e-7,1.50e-10],
  [4.39e-11,5.31e-11,1.08e-10,0,5.55e-11,1.41e-10,2.59e-10,5.79e-11,4.32e-10,1.52e-7,1.21e-10,3.35e-11,4.12e-11,3.66e-11,2.79e-10,4.39e-11],
  [2.33e-10,7.05e-9,6.49e-11,5.55e-11,0,4.25e-9,4.10e-11,1.88e-8,8.48e-10,1.10e-10,1.08e-7,5.91e-9,2.28e-6,8.10e-10,4.58e-11,4.70e-9],
  [7.32e-11,7.45e-8,1.03e-10,1.41e-10,4.25e-9,0,7.41e-11,2.18e-10,6.79e-7,6.70e-10,1.03e-9,2.06e-10,1.09e-9,9.81e-11,1.31e-10,1.39e-10],
  [1.65e-9,1.28e-10,1.17e-6,2.59e-10,4.10e-11,7.41e-11,0,5.16e-11,8.70e-11,1.29e-10,3.06e-11,1.24e-10,5.27e-11,3.27e-10,3.82e-6,8.99e-11],
  [8.05e-10,3.71e-10,7.36e-11,5.79e-11,1.88e-8,2.18e-10,5.16e-11,0,1.07e-10,6.15e-11,3.75e-8,6.46e-8,1.22e-7,6.46e-9,4.15e-11,1.42e-6],
  [4.53e-11,4.56e-9,9.34e-11,4.32e-10,8.48e-10,6.79e-7,8.70e-11,1.07e-10,0,4.17e-9,5.19e-10,8.47e-11,2.93e-10,5.13e-11,1.66e-10,6.87e-11],
  [3.12e-11,1.28e-10,7.44e-11,1.52e-7,1.10e-10,6.70e-10,1.29e-10,6.15e-11,4.17e-9,0,2.17e-10,3.44e-11,6.25e-11,2.98e-11,1.86e-10,4.04e-11],
  [1.20e-10,5.92e-10,3.69e-11,1.21e-10,1.08e-7,1.03e-9,3.06e-11,3.75e-8,5.19e-10,2.17e-10,0,1.52e-9,2.65e-8,3.69e-10,3.07e-11,3.36e-9],
  [1.21e-8,9.12e-10,2.69e-10,3.35e-11,5.91e-9,2.06e-10,1.24e-10,6.46e-8,8.47e-11,3.44e-11,1.52e-9,0,7.74e-8,6.93e-7,9.60e-11,2.29e-6],
  [6.51e-10,3.72e-9,9.32e-11,4.12e-11,2.28e-6,1.09e-9,5.27e-11,1.22e-7,2.93e-10,6.25e-11,2.65e-8,7.74e-8,0,3.84e-9,5.24e-11,4.20e-8],
  [5.26e-7,3.85e-10,8.12e-10,3.66e-11,8.10e-10,9.81e-11,3.27e-10,6.46e-9,5.13e-11,2.98e-11,3.69e-10,6.93e-7,3.84e-9,0,1.98e-10,1.17e-7],
  [7.45e-10,2.19e-10,4.42e-7,2.79e-10,4.58e-11,1.31e-10,3.82e-6,4.15e-11,1.66e-10,1.86e-10,3.07e-11,9.60e-11,5.24e-11,1.98e-10,0,6.47e-11],
  [4.49e-9,3.57e-10,1.50e-10,4.39e-11,4.70e-9,1.39e-10,8.99e-11,1.42e-6,6.87e-11,4.04e-11,3.36e-9,2.29e-6,4.20e-8,1.17e-7,6.47e-11,0]
];

const PIPELINE_STAGES = [
  { id: 1, name: "Config", done: true },
  { id: 2, name: "Coil", done: true },
  { id: 3, name: "Helmet", done: true },
  { id: 4, name: "Basis", done: true },
  { id: 5, name: "Inductance", done: true },
  { id: 6, name: "GA Opt", done: true },
  { id: 7, name: "Validate", done: true },
  { id: 8, name: "Voltages", done: true },
  { id: 9, name: "SAC", done: false },
  { id: 10, name: "Report", done: true },
  { id: 11, name: "Sensitivity", done: true },
  { id: 12, name: "CP Bridge", done: true },
  { id: 13, name: "Trajectory", done: true },
];

// ============================================================
// §2  UTILITY FUNCTIONS
// ============================================================

const lerp = (a, b, t) => a + (b - a) * t;
const fmt = (n, d = 4) => typeof n === "number" ? n.toFixed(d) : "—";
const fmtSci = (n) => n < 0.001 || n > 9999 ? n.toExponential(2) : n.toFixed(4);

const ampColor = (val, min, max) => {
  const t = Math.max(0, Math.min(1, (val - min) / (max - min + 1e-12)));
  const r = Math.round(lerp(10, 0, t));
  const g = Math.round(lerp(14, 217, t));
  const b = Math.round(lerp(39, 255, t));
  return `rgb(${r},${g},${b})`;
};

const divColor = (val, absMax) => {
  const t = Math.max(-1, Math.min(1, val / (absMax + 1e-12)));
  if (t < 0) {
    const s = -t;
    return `rgb(${Math.round(lerp(255, 40, s))},${Math.round(lerp(255, 80, s))},${Math.round(255)})`;
  }
  const s = t;
  return `rgb(${Math.round(255)},${Math.round(lerp(255, 60, s))},${Math.round(lerp(255, 60, s))})`;
};

const getAmpsAtStep = (step) => {
  if (step < 30) return TRAJ_AMPS_START;
  const t = (step - 29) / 29;
  return TRAJ_AMPS_START.map((a, i) => lerp(a, TRAJ_AMPS_END[i], t));
};

const scalarFromSingleton = (arr, fallback = 0) => {
  if (!Array.isArray(arr) || arr.length === 0 || !Number.isFinite(arr[0])) return fallback;
  return arr[0];
};

const buildAmplitudeSeriesFromEndpoints = (start, end, stepCount) => {
  if (!Array.isArray(start) || !Array.isArray(end) || stepCount <= 0) return [];
  if (stepCount === 1) return [start];
  return Array.from({ length: stepCount }, (_, step) => {
    const t = step / (stepCount - 1);
    return start.map((a, i) => lerp(a, end[i] ?? a, t));
  });
};

const toStageStatus = (stagesRun) => {
  const runSet = new Set(Array.isArray(stagesRun) ? stagesRun : []);
  return PIPELINE_STAGES.map((stage) => ({ ...stage, done: runSet.has(stage.id) }));
};

const conditionNumber2xN = (matrix) => {
  if (!Array.isArray(matrix) || matrix.length !== 2) return null;
  const rowA = matrix[0];
  const rowB = matrix[1];
  if (!Array.isArray(rowA) || !Array.isArray(rowB) || rowA.length === 0 || rowA.length !== rowB.length) {
    return null;
  }

  let aa = 0;
  let bb = 0;
  let ab = 0;
  for (let i = 0; i < rowA.length; i += 1) {
    const a = Number(rowA[i]) || 0;
    const b = Number(rowB[i]) || 0;
    aa += a * a;
    bb += b * b;
    ab += a * b;
  }
  const trace = aa + bb;
  const det = aa * bb - ab * ab;
  const disc = Math.max(trace * trace - 4 * det, 0);
  const lambdaMax = (trace + Math.sqrt(disc)) / 2;
  const lambdaMin = (trace - Math.sqrt(disc)) / 2;
  if (lambdaMin <= 0) return null;
  return Math.sqrt(lambdaMax / lambdaMin);
};

const FALLBACK_VIZ_DATA = {
  GA_HISTORY,
  GA_AMPLITUDES,
  GROUP,
  CP_GROUP,
  POSITIONS,
  CP_METRICS,
  AGENT_FE,
  ETA_BEFORE,
  ETA_AFTER,
  TRAJ_ENERGY,
  TRAJ_PHI,
  TRAJ_SYNC,
  TRAJ_VTARGET,
  TRAJ_VSURF,
  TRAJ_AMPS_START,
  TRAJ_AMPS_END,
  TRAJ_AMPLITUDES: buildAmplitudeSeriesFromEndpoints(TRAJ_AMPS_START, TRAJ_AMPS_END, TRAJ_ENERGY.length),
  WAYPOINTS: [0, 29, 58],
  WORLD_ENERGIES,
  WORLD_PROBS,
  J_ALPHA,
  J_TIME,
  PARETO_VT,
  PARETO_VS,
  PARETO_LAMBDAS,
  REACHABLE,
  TE_MATRIX,
  PIPELINE_STAGES,
  SUMMARY: {
    mode: "NTS",
    num_coils: 16,
    stages_run: PIPELINE_STAGES.filter((s) => s.done).map((s) => s.id),
    ga_result: { fitness: 36.07859529401943 },
  },
};

const VizDataContext = createContext(null);
const useVizData = () => useContext(VizDataContext) || FALLBACK_VIZ_DATA;

const buildVizDataFromBundle = (bundle) => {
  const outputs = Array.isArray(bundle?.trajectory_result?.outputs) ? bundle.trajectory_result.outputs : [];
  const trajectoryAmplitudes = Array.isArray(bundle?.trajectory_result?.amplitudes)
    ? bundle.trajectory_result.amplitudes
    : [];
  const stepCount = Array.isArray(bundle?.trajectory_result?.time)
    ? bundle.trajectory_result.time.length
    : trajectoryAmplitudes.length;
  const vtSeries = outputs.map((row) => (Array.isArray(row) ? (Number(row[0]) || 0) : 0));
  const vsSeries = outputs.map((row) => (Array.isArray(row) ? (Number(row[1]) || 0) : 0));
  const startAmps = trajectoryAmplitudes[0] || bundle?.ga_best?.amplitudes || FALLBACK_VIZ_DATA.TRAJ_AMPS_START;
  const endAmps =
    trajectoryAmplitudes[Math.max(trajectoryAmplitudes.length - 1, 0)] ||
    bundle?.ga_best?.amplitudes ||
    FALLBACK_VIZ_DATA.TRAJ_AMPS_END;
  const amplitudesTimeline =
    trajectoryAmplitudes.length > 0
      ? trajectoryAmplitudes
      : buildAmplitudeSeriesFromEndpoints(startAmps, endAmps, Math.max(stepCount, 1));

  const cp = bundle?.cp_bridge_analysis;
  const sensitivity = bundle?.sensitivity_analysis;
  const summary = bundle?.pipeline_summary || FALLBACK_VIZ_DATA.SUMMARY;
  const worldEnergy = Array.isArray(cp?.world_energies) ? cp.world_energies : FALLBACK_VIZ_DATA.WORLD_ENERGIES;
  const worldProb = Array.isArray(cp?.world_probabilities)
    ? cp.world_probabilities
    : FALLBACK_VIZ_DATA.WORLD_PROBS;

  const reachableVt = Array.isArray(sensitivity?.reachable_nts__V_target)
    ? sensitivity.reachable_nts__V_target
    : [];
  const reachableVs = Array.isArray(sensitivity?.reachable_nts__V_surface_max)
    ? sensitivity.reachable_nts__V_surface_max
    : [];
  const reachableCount = Math.min(reachableVt.length, reachableVs.length);
  const reachablePairs = Array.from({ length: reachableCount }, (_, i) => [
    reachableVt[i],
    reachableVs[i],
  ]);

  return {
    ...FALLBACK_VIZ_DATA,
    GA_AMPLITUDES: bundle?.ga_best?.amplitudes || FALLBACK_VIZ_DATA.GA_AMPLITUDES,
    GROUP: bundle?.ga_best?.group || FALLBACK_VIZ_DATA.GROUP,
    CP_GROUP: cp?.group || FALLBACK_VIZ_DATA.CP_GROUP,
    POSITIONS: bundle?.ga_best?.positions || FALLBACK_VIZ_DATA.POSITIONS,
    CP_METRICS: {
      phi: scalarFromSingleton(cp?.collective_phi, FALLBACK_VIZ_DATA.CP_METRICS.phi),
      sync: scalarFromSingleton(cp?.sync_order_parameter, FALLBACK_VIZ_DATA.CP_METRICS.sync),
      groupF2: scalarFromSingleton(cp?.group_free_energy, FALLBACK_VIZ_DATA.CP_METRICS.groupF2),
      morl: {
        J_phi: scalarFromSingleton(cp?.morl_objectives__J_phi, FALLBACK_VIZ_DATA.CP_METRICS.morl.J_phi),
        J_arch: scalarFromSingleton(cp?.morl_objectives__J_arch, FALLBACK_VIZ_DATA.CP_METRICS.morl.J_arch),
        J_sync: scalarFromSingleton(cp?.morl_objectives__J_sync, FALLBACK_VIZ_DATA.CP_METRICS.morl.J_sync),
        J_task: scalarFromSingleton(cp?.morl_objectives__J_task, FALLBACK_VIZ_DATA.CP_METRICS.morl.J_task),
      },
      energy: {
        E_nca: scalarFromSingleton(cp?.energy__E_nca, FALLBACK_VIZ_DATA.CP_METRICS.energy.E_nca),
        E_mf: scalarFromSingleton(cp?.energy__E_mf, FALLBACK_VIZ_DATA.CP_METRICS.energy.E_mf),
        E_arch: scalarFromSingleton(cp?.energy__E_arch, FALLBACK_VIZ_DATA.CP_METRICS.energy.E_arch),
        E_phi: scalarFromSingleton(cp?.energy__E_phi, FALLBACK_VIZ_DATA.CP_METRICS.energy.E_phi),
        E_couple: scalarFromSingleton(cp?.energy__E_couple, FALLBACK_VIZ_DATA.CP_METRICS.energy.E_couple),
        E_morl: scalarFromSingleton(cp?.energy__E_morl, FALLBACK_VIZ_DATA.CP_METRICS.energy.E_morl),
        E_total: scalarFromSingleton(cp?.energy__E_total, FALLBACK_VIZ_DATA.CP_METRICS.energy.E_total),
      },
    },
    AGENT_FE: cp?.agent_free_energies || FALLBACK_VIZ_DATA.AGENT_FE,
    ETA_BEFORE: cp?.eta_field || FALLBACK_VIZ_DATA.ETA_BEFORE,
    ETA_AFTER: cp?.eta_field_post_implacement || FALLBACK_VIZ_DATA.ETA_AFTER,
    TRAJ_ENERGY: bundle?.trajectory_result?.energy || FALLBACK_VIZ_DATA.TRAJ_ENERGY,
    TRAJ_PHI: bundle?.trajectory_result?.phi || FALLBACK_VIZ_DATA.TRAJ_PHI,
    TRAJ_SYNC: bundle?.trajectory_result?.sync || FALLBACK_VIZ_DATA.TRAJ_SYNC,
    TRAJ_VTARGET: vtSeries.length > 0 ? vtSeries : FALLBACK_VIZ_DATA.TRAJ_VTARGET,
    TRAJ_VSURF: vsSeries.length > 0 ? vsSeries : FALLBACK_VIZ_DATA.TRAJ_VSURF,
    TRAJ_AMPS_START: startAmps,
    TRAJ_AMPS_END: endAmps,
    TRAJ_AMPLITUDES: amplitudesTimeline,
    WAYPOINTS:
      bundle?.trajectory_result?.waypoint_indices ||
      [0, Math.floor(Math.max(stepCount - 1, 0) / 2), Math.max(stepCount - 1, 0)],
    WORLD_ENERGIES: worldEnergy,
    WORLD_PROBS: worldProb,
    J_ALPHA: sensitivity?.jacobian_nts__J_alpha || FALLBACK_VIZ_DATA.J_ALPHA,
    J_TIME: sensitivity?.jacobian_nts__J_time || FALLBACK_VIZ_DATA.J_TIME,
    PARETO_VT: sensitivity?.pareto_nts__V_target || FALLBACK_VIZ_DATA.PARETO_VT,
    PARETO_VS: sensitivity?.pareto_nts__V_surface_max || FALLBACK_VIZ_DATA.PARETO_VS,
    PARETO_LAMBDAS: sensitivity?.pareto_nts__lambdas || FALLBACK_VIZ_DATA.PARETO_LAMBDAS,
    REACHABLE: reachablePairs.length > 0 ? reachablePairs : FALLBACK_VIZ_DATA.REACHABLE,
    TE_MATRIX: cp?.te_matrix || FALLBACK_VIZ_DATA.TE_MATRIX,
    PIPELINE_STAGES: toStageStatus(summary?.stages_run),
    SUMMARY: summary,
  };
};

// ============================================================
// §3  CUSTOM TOOLTIP
// ============================================================

const DarkTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.cyan}`, borderRadius: 6, padding: "8px 12px" }}>
      {label !== undefined && <div style={{ color: COLORS.textDim, fontSize: 11, marginBottom: 4 }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || COLORS.text, fontSize: 12 }}>
          {p.name}: <strong>{typeof p.value === "number" ? fmtSci(p.value) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

// ============================================================
// §4  SYSTEM OVERVIEW VIEW
// ============================================================

const SystemOverview = () => {
  const {
    CP_METRICS,
    PIPELINE_STAGES,
    GA_HISTORY,
    SUMMARY,
    WORLD_ENERGIES,
    TRAJ_ENERGY,
    PARETO_VT,
    J_ALPHA,
    J_TIME,
  } = useVizData();
  const kappaAlpha = conditionNumber2xN(J_ALPHA);
  const kappaTime = conditionNumber2xN(J_TIME);

  const energyData = useMemo(() => [
    { name: "E_nca", value: CP_METRICS.energy.E_nca, color: COLORS.cyan },
    { name: "E_mf", value: CP_METRICS.energy.E_mf, color: COLORS.teal },
    { name: "E_arch", value: CP_METRICS.energy.E_arch, color: COLORS.gold },
    { name: "|E_phi|", value: Math.abs(CP_METRICS.energy.E_phi), color: COLORS.purple },
    { name: "E_morl", value: CP_METRICS.energy.E_morl, color: COLORS.orange },
  ], [CP_METRICS.energy]);

  const morlData = useMemo(() => {
    const raw = [
      { axis: "J_phi", raw: CP_METRICS.morl.J_phi },
      { axis: "J_arch", raw: CP_METRICS.morl.J_arch },
      { axis: "J_sync", raw: CP_METRICS.morl.J_sync },
      { axis: "J_task", raw: CP_METRICS.morl.J_task },
    ];
    const denom = Math.max(...raw.map((d) => Math.abs(d.raw)), 1e-9);
    return raw.map((d) => ({ ...d, value: d.raw / denom }));
  }, [CP_METRICS.morl]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* Pipeline Flow */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16, gridColumn: "1 / -1" }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Pipeline Flow (13 Stages)</div>
        <svg width="100%" height={60} viewBox="0 0 780 50">
          {PIPELINE_STAGES.map((s, i) => {
            const x = 10 + i * 60;
            return (
              <g key={s.id}>
                <rect x={x} y={10} width={50} height={28} rx={4}
                  fill={s.done ? COLORS.cyan + "30" : "#333"} stroke={s.done ? COLORS.cyan : "#555"} strokeWidth={1.5} />
                <text x={x + 25} y={28} textAnchor="middle" fill={s.done ? COLORS.cyan : "#888"} fontSize={8} fontFamily="monospace">{s.name}</text>
                {i < 12 && <line x1={x + 52} y1={24} x2={x + 58} y2={24} stroke="#555" strokeWidth={1} />}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Key Stats */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Key Metrics</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {[
            ["Collective \u03a6", fmt(CP_METRICS.phi, 6), COLORS.cyan],
            ["Sync R", fmt(CP_METRICS.sync, 4), COLORS.cyan],
            ["Group F\u00b2", fmt(CP_METRICS.groupF2, 4), COLORS.magenta],
            ["E_total", fmt(CP_METRICS.energy.E_total, 2) + " mJ", COLORS.orange],
            ["\u03ba_\u03b1", kappaAlpha ? fmt(kappaAlpha, 2) : "—", COLORS.gold],
            ["\u03ba_t", kappaTime ? fmt(kappaTime, 2) : "—", COLORS.red],
            ["Worlds", String(WORLD_ENERGIES.length), COLORS.text],
            ["Trajectory", `${TRAJ_ENERGY.length} steps`, COLORS.text],
            ["Mode", SUMMARY.mode || "NTS", COLORS.lime],
            ["Coils", String(SUMMARY.num_coils || 0), COLORS.text],
            ["Fitness", fmt(SUMMARY?.ga_result?.fitness, 2), COLORS.cyan],
            ["Pareto Pts", String(PARETO_VT.length), COLORS.gold],
          ].map(([label, val, color], i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: `1px solid ${COLORS.border}` }}>
              <span style={{ color: COLORS.textDim, fontSize: 11 }}>{label}</span>
              <span style={{ color, fontSize: 12, fontFamily: "monospace", fontWeight: 600 }}>{val}</span>
            </div>
          ))}
        </div>
      </div>

      {/* GA Convergence */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>GA Convergence</div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={GA_HISTORY}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="gen" stroke={COLORS.textDim} fontSize={10} label={{ value: "Generation", position: "bottom", fill: COLORS.textDim, fontSize: 10 }} />
            <YAxis domain={[34, 56]} stroke={COLORS.textDim} fontSize={10} />
            <Tooltip content={<DarkTooltip />} />
            <Line type="monotone" dataKey="best" stroke={COLORS.cyan} strokeWidth={2} dot={{ fill: COLORS.cyan, r: 4 }} name="Best" />
            <Line type="monotone" dataKey="median" stroke={COLORS.magenta} strokeWidth={2} dot={{ fill: COLORS.magenta, r: 3 }} name="Median" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Energy Decomposition Donut */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>4D NCA Energy Decomposition</div>
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie data={energyData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={2} label={({ name, value }) => `${name}`} labelLine={{ stroke: COLORS.textDim }}>
              {energyData.map((e, i) => <Cell key={i} fill={e.color} />)}
            </Pie>
            <Tooltip content={<DarkTooltip />} />
          </PieChart>
        </ResponsiveContainer>
        <div style={{ textAlign: "center", color: COLORS.gold, fontSize: 12, fontFamily: "monospace", marginTop: -4 }}>E_total = {fmt(CP_METRICS.energy.E_total, 2)} mJ</div>
      </div>

      {/* MORL Radar */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16, gridColumn: "1 / -1" }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>MORL Objectives</div>
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <ResponsiveContainer width="50%" height={200}>
            <RadarChart data={morlData}>
              <PolarGrid stroke="#333" />
              <PolarAngleAxis dataKey="axis" tick={{ fill: COLORS.text, fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 1.1]} tick={false} axisLine={false} />
              <Radar dataKey="value" stroke={COLORS.magenta} fill={COLORS.magenta} fillOpacity={0.25} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
          <div style={{ flex: 1 }}>
            {[
              ["J_phi (consciousness)", CP_METRICS.morl.J_phi, COLORS.cyan],
              ["J_arch (architecture)", CP_METRICS.morl.J_arch, COLORS.gold],
              ["J_sync (synchrony)", CP_METRICS.morl.J_sync, COLORS.magenta],
              ["J_task (performance)", CP_METRICS.morl.J_task, COLORS.lime],
            ].map(([label, val, color], i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, padding: "4px 8px", background: COLORS.bg, borderRadius: 4 }}>
                <span style={{ color: COLORS.textDim, fontSize: 11 }}>{label}</span>
                <span style={{ color, fontSize: 13, fontFamily: "monospace", fontWeight: 700 }}>{fmt(val, 4)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================
// §5  MANY-WORLDS & TRAJECTORY VIEW
// ============================================================

const WorldsTrajectoryView = ({ timeStep, setTimeStep }) => {
  const {
    WORLD_ENERGIES,
    WORLD_PROBS,
    TRAJ_ENERGY,
    TRAJ_PHI,
    TRAJ_VTARGET,
    TRAJ_VSURF,
    TRAJ_AMPLITUDES,
    TRAJ_AMPS_START,
    TRAJ_AMPS_END,
    CP_GROUP,
    WAYPOINTS,
  } = useVizData();
  const stepCount = Math.max(1, TRAJ_ENERGY.length);
  const maxStep = stepCount - 1;
  const clampedStep = Math.max(0, Math.min(timeStep, maxStep));
  const waypointList = Array.isArray(WAYPOINTS) && WAYPOINTS.length > 0
    ? WAYPOINTS.map((w) => Math.max(0, Math.min(Math.floor(w), maxStep)))
    : [0, Math.floor(maxStep / 2), maxStep];
  const amplitudeTimeline = TRAJ_AMPLITUDES?.length
    ? TRAJ_AMPLITUDES
    : buildAmplitudeSeriesFromEndpoints(TRAJ_AMPS_START, TRAJ_AMPS_END, stepCount);

  const worldData = useMemo(() =>
    WORLD_ENERGIES.map((e, i) => ({
      id: i, energy: e, prob: WORLD_PROBS[i],
      y: i < 48 ? 5 + (i % 8) * 2.5 + Math.sin(i * 0.7) * 3 : (i === 48 ? 25 : 45),
    })), [WORLD_ENERGIES, WORLD_PROBS]);

  const trajLine = useMemo(() => TRAJ_ENERGY.map((e, i) => ({
    step: i, energy: e, vTarget: TRAJ_VTARGET[i], vSurf: TRAJ_VSURF[i],
  })), [TRAJ_ENERGY, TRAJ_VTARGET, TRAJ_VSURF]);

  const currentAmps = useMemo(
    () => amplitudeTimeline[Math.min(clampedStep, amplitudeTimeline.length - 1)] || [],
    [amplitudeTimeline, clampedStep],
  );
  const ampValues = amplitudeTimeline.flat();
  const ampMin = Math.min(...ampValues, 0);
  const ampMax = Math.max(...ampValues, 1);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Many-Worlds Scatter */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Many-Worlds Landscape (50 Pareto-Optimal Configurations)</div>
        <ResponsiveContainer width="100%" height={200}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="energy" name="Energy" domain={[0, 60]} stroke={COLORS.textDim} fontSize={10}
              label={{ value: "Energy (mJ)", position: "bottom", fill: COLORS.textDim, fontSize: 10, offset: 0 }} />
            <YAxis dataKey="y" name="World spread" stroke={COLORS.textDim} fontSize={10} hide />
            <ZAxis dataKey="prob" range={[20, 200]} />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]?.payload;
              return (
                <div style={{ background: COLORS.card, border: `1px solid ${COLORS.cyan}`, borderRadius: 6, padding: "8px 12px" }}>
                  <div style={{ color: COLORS.cyan, fontSize: 12 }}>World {d.id}</div>
                  <div style={{ color: COLORS.text, fontSize: 11 }}>Energy: {fmt(d.energy, 3)} mJ</div>
                  <div style={{ color: COLORS.text, fontSize: 11 }}>Prob: {fmtSci(d.prob)}</div>
                </div>
              );
            }} />
            <Scatter data={worldData} fill={COLORS.cyan} fillOpacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Timeline Scrubber */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: "12px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{ color: COLORS.textDim, fontSize: 11, minWidth: 80 }}>Timestep: <span style={{ color: COLORS.cyan, fontFamily: "monospace" }}>{clampedStep}</span>/{maxStep}</span>
          <input type="range" min={0} max={maxStep} value={clampedStep} onChange={e => setTimeStep(+e.target.value)}
            style={{ flex: 1, accentColor: COLORS.cyan }} />
          <div style={{ display: "flex", gap: 12 }}>
            <span style={{ color: COLORS.textDim, fontSize: 10 }}>E: <span style={{ color: COLORS.cyan, fontFamily: "monospace" }}>{fmt(TRAJ_ENERGY[clampedStep], 2)}</span></span>
            <span style={{ color: COLORS.textDim, fontSize: 10 }}>\u03a6: <span style={{ color: COLORS.magenta, fontFamily: "monospace" }}>{fmt(TRAJ_PHI[clampedStep], 6)}</span></span>
            <span style={{ color: COLORS.textDim, fontSize: 10 }}>V_t: <span style={{ color: COLORS.gold, fontFamily: "monospace" }}>{fmt(TRAJ_VTARGET[clampedStep], 4)}</span></span>
          </div>
        </div>
        {/* Waypoint markers */}
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
          {waypointList.map((wp, idx) => (
            <span key={`${wp}-${idx}`} style={{ color: COLORS.gold, fontSize: 9, cursor: "pointer" }} onClick={() => setTimeStep(wp)}>
              {idx === 0 ? "\u2605 Start" : idx === waypointList.length - 1 ? "\u2605 End" : "\u2605 Mid"}
            </span>
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Amplitude Heatmap */}
        <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16, overflow: "auto" }}>
          <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Coil Amplitudes Over Time</div>
          <div style={{ display: "flex", gap: 4 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
              {Array.from({ length: currentAmps.length || CP_GROUP.length }, (_, c) => (
                <div key={c} style={{ height: 14, display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: 4 }}>
                  <span style={{ color: CP_GROUP[c] === 0 ? COLORS.cyan : COLORS.magenta, fontSize: 8, fontFamily: "monospace" }}>
                    {c}{CP_GROUP[c] === 0 ? "M" : "F"}
                  </span>
                </div>
              ))}
            </div>
            <svg width={stepCount * 8} height={(currentAmps.length || CP_GROUP.length) * 14} style={{ display: "block" }}>
              {Array.from({ length: currentAmps.length || CP_GROUP.length }, (_, coil) =>
                Array.from({ length: stepCount }, (_, step) => {
                  const amps = amplitudeTimeline[step] || [];
                  return (
                    <rect key={`${coil}-${step}`} x={step * 8} y={coil * 14} width={8} height={13}
                      fill={ampColor(amps[coil] ?? 0, ampMin, ampMax)} rx={1}
                      style={{ cursor: "pointer" }} onClick={() => setTimeStep(step)} />
                  );
                })
              )}
              {/* Current timestep cursor */}
              <rect x={clampedStep * 8} y={0} width={8} height={(currentAmps.length || CP_GROUP.length) * 14} fill="none" stroke={COLORS.gold} strokeWidth={2} rx={1} />
            </svg>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, marginLeft: 30 }}>
            <span style={{ color: COLORS.textDim, fontSize: 8 }}>t=0</span>
            <span style={{ color: COLORS.textDim, fontSize: 8 }}>t=mid</span>
            <span style={{ color: COLORS.textDim, fontSize: 8 }}>t=end</span>
          </div>
        </div>

        {/* Time Series */}
        <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
          <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Trajectory Time Series</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trajLine}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="step" stroke={COLORS.textDim} fontSize={10} />
              <YAxis yAxisId="left" stroke={COLORS.cyan} fontSize={10} domain={[0, 60]} />
              <YAxis yAxisId="right" orientation="right" stroke={COLORS.gold} fontSize={10} domain={[0, 6.5]} />
              <Tooltip content={<DarkTooltip />} />
              <Legend wrapperStyle={{ fontSize: 10, color: COLORS.textDim }} />
              {waypointList.map((wp, idx) => (
                <ReferenceLine key={`wp-${idx}`} yAxisId="left" x={wp} stroke={COLORS.gold} strokeDasharray="5 5" strokeWidth={1} />
              ))}
              <ReferenceLine yAxisId="left" x={clampedStep} stroke={COLORS.gold} strokeWidth={2} />
              <Line yAxisId="left" type="monotone" dataKey="energy" stroke={COLORS.cyan} strokeWidth={2} dot={false} name="Energy (mJ)" />
              <Line yAxisId="right" type="monotone" dataKey="vTarget" stroke={COLORS.magenta} strokeWidth={1.5} dot={false} name="V_target" />
              <Line yAxisId="right" type="monotone" dataKey="vSurf" stroke={COLORS.gold} strokeWidth={1.5} dot={false} name="V_surface" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Current Amplitude Bar */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Current Amplitude Vector (step {clampedStep})</div>
        <ResponsiveContainer width="100%" height={100}>
          <BarChart data={currentAmps.map((a, i) => ({ coil: i, amp: +a.toFixed(3), group: CP_GROUP[i] === 0 ? "M" : "F" }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="coil" stroke={COLORS.textDim} fontSize={9} />
            <YAxis domain={[0, 1]} stroke={COLORS.textDim} fontSize={9} />
            <Tooltip content={<DarkTooltip />} />
            <Bar dataKey="amp" name="Amplitude">
              {currentAmps.map((_, i) => <Cell key={i} fill={CP_GROUP[i] === 0 ? COLORS.cyan : COLORS.magenta} fillOpacity={0.8} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// ============================================================
// §6  CONTROL & SENSITIVITY VIEW
// ============================================================

const ControlSensitivityView = () => {
  const {
    PARETO_VT,
    PARETO_VS,
    PARETO_LAMBDAS,
    REACHABLE,
    J_ALPHA,
    J_TIME,
  } = useVizData();
  const [hovered, setHovered] = useState(null);

  const paretoData = useMemo(() =>
    PARETO_VT.map((vt, i) => ({
      vt, vs: PARETO_VS[i], lambda: PARETO_LAMBDAS[i],
      dominated: i < 48, // first 48 cluster, last 2 are distinct
    })), [PARETO_VT, PARETO_VS, PARETO_LAMBDAS]);

  const reachableData = useMemo(() =>
    REACHABLE.map(([vt, vs], i) => ({ vt, vs, id: i })), [REACHABLE]);

  const absMaxAlpha = Math.max(...J_ALPHA.flat().map(Math.abs), 1e-9);
  const absMaxTime = Math.max(...J_TIME.flat().map(Math.abs), 1e-9);
  const kappaAlpha = conditionNumber2xN(J_ALPHA);
  const kappaTime = conditionNumber2xN(J_TIME);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* Jacobian Heatmaps */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Jacobian J_\u03b1 (\u2202output/\u2202amplitude)</div>
        <svg width="100%" height={70} viewBox="0 0 400 60">
          {["V_target", "V_surface"].map((label, row) => (
            <g key={row}>
              <text x={0} y={row * 25 + 17} fill={COLORS.textDim} fontSize={8}>{label}</text>
              {J_ALPHA[row].map((val, col) => (
                <g key={col}>
                  <rect x={60 + col * 20} y={row * 25} width={19} height={22} rx={2}
                    fill={divColor(val, absMaxAlpha)} stroke={hovered === `a${row}${col}` ? COLORS.gold : "none"} strokeWidth={1.5}
                    onMouseEnter={() => setHovered(`a${row}${col}`)} onMouseLeave={() => setHovered(null)} />
                  <text x={60 + col * 20 + 10} y={row * 25 + 14} textAnchor="middle" fill={COLORS.bg} fontSize={6}>{val > 0.01 ? val.toFixed(2) : ""}</text>
                </g>
              ))}
            </g>
          ))}
        </svg>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginTop: 12, marginBottom: 8 }}>Jacobian J_t (\u2202output/\u2202timing)</div>
        <svg width="100%" height={70} viewBox="0 0 400 60">
          {["V_target", "V_surface"].map((label, row) => (
            <g key={row}>
              <text x={0} y={row * 25 + 17} fill={COLORS.textDim} fontSize={8}>{label}</text>
              {J_TIME[row].map((val, col) => (
                <g key={col}>
                  <rect x={60 + col * 20} y={row * 25} width={19} height={22} rx={2}
                    fill={divColor(val, absMaxTime)} stroke={hovered === `t${row}${col}` ? COLORS.gold : "none"} strokeWidth={1.5}
                    onMouseEnter={() => setHovered(`t${row}${col}`)} onMouseLeave={() => setHovered(null)} />
                </g>
              ))}
            </g>
          ))}
        </svg>
        {hovered && <div style={{ color: COLORS.gold, fontSize: 11, marginTop: 4, fontFamily: "monospace" }}>
          {hovered.startsWith("a") ? `J_\u03b1[${hovered[1]}][${hovered.slice(2)}] = ${J_ALPHA[+hovered[1]][+hovered.slice(2)].toFixed(4)}` : `J_t[${hovered[1]}][${hovered.slice(2)}] = ${J_TIME[+hovered[1]][+hovered.slice(2)].toFixed(3)}`}
        </div>}
      </div>

      {/* Pareto Front */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Pareto Front (V_target vs V_surface)</div>
        <ResponsiveContainer width="100%" height={250}>
          <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="vt" name="V_target" stroke={COLORS.textDim} fontSize={10} domain={[0, 0.16]}
              label={{ value: "V_target (mV)", position: "bottom", fill: COLORS.textDim, fontSize: 10 }} />
            <YAxis dataKey="vs" name="V_surface" stroke={COLORS.textDim} fontSize={10} domain={[0, 7]}
              label={{ value: "V_surface (mV)", angle: -90, position: "left", fill: COLORS.textDim, fontSize: 10 }} />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]?.payload;
              return (
                <div style={{ background: COLORS.card, border: `1px solid ${COLORS.gold}`, borderRadius: 6, padding: "8px 12px" }}>
                  <div style={{ color: COLORS.gold, fontSize: 11 }}>\u03bb = {d.lambda?.toFixed(2)}</div>
                  <div style={{ color: COLORS.text, fontSize: 11 }}>V_t: {d.vt?.toFixed(4)} mV</div>
                  <div style={{ color: COLORS.text, fontSize: 11 }}>V_s: {d.vs?.toFixed(3)} mV</div>
                </div>
              );
            }} />
            <Scatter data={paretoData} fill={COLORS.gold} fillOpacity={0.7} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Reachable Set */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Reachable Set (50 Samples)</div>
        <ResponsiveContainer width="100%" height={220}>
          <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="vt" name="V_target" stroke={COLORS.textDim} fontSize={10} domain={[0.04, 0.13]}
              label={{ value: "V_target (mV)", position: "bottom", fill: COLORS.textDim, fontSize: 10 }} />
            <YAxis dataKey="vs" name="V_surface" stroke={COLORS.textDim} fontSize={10} domain={[2, 5.5]}
              label={{ value: "V_surface (mV)", angle: -90, position: "left", fill: COLORS.textDim, fontSize: 10 }} />
            <Tooltip content={<DarkTooltip />} />
            <Scatter data={reachableData} fill={COLORS.cyan} fillOpacity={0.5} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Condition Numbers */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Condition Numbers</div>
        {[
          { label: "\u03ba_\u03b1 (amplitude)", value: kappaAlpha || 0, max: 500 },
          { label: "\u03ba_t (timing)", value: kappaTime || 0, max: 500 },
        ].map(({ label, value, max }, i) => {
          const pct = Math.min(100, (value / max) * 100);
          const color = value < 10 ? COLORS.lime : value < 100 ? COLORS.gold : COLORS.red;
          return (
            <div key={i} style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ color: COLORS.textDim, fontSize: 11 }}>{label}</span>
                <span style={{ color, fontSize: 13, fontFamily: "monospace", fontWeight: 700 }}>{value}</span>
              </div>
              <div style={{ height: 12, background: COLORS.bg, borderRadius: 6, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 6, transition: "width 0.3s" }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2 }}>
                <span style={{ color: COLORS.lime, fontSize: 8 }}>Good (&lt;10)</span>
                <span style={{ color: COLORS.gold, fontSize: 8 }}>Moderate</span>
                <span style={{ color: COLORS.red, fontSize: 8 }}>Poor (&gt;100)</span>
              </div>
            </div>
          );
        })}

        <div style={{ color: COLORS.text, fontSize: 13, fontWeight: 600, marginTop: 8, marginBottom: 8 }}>Controllability</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {[
            ["State rank", "33/34", 33/34],
            ["Observability", "18/34", 18/34],
            ["Output ctrl", "5/5", 1.0],
          ].map(([label, text, pct], i) => (
            <div key={i}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                <span style={{ color: COLORS.textDim, fontSize: 10 }}>{label}</span>
                <span style={{ color: COLORS.cyan, fontSize: 10, fontFamily: "monospace" }}>{text}</span>
              </div>
              <div style={{ height: 6, background: COLORS.bg, borderRadius: 3 }}>
                <div style={{ height: "100%", width: `${pct * 100}%`, background: COLORS.cyan, borderRadius: 3 }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ============================================================
// §7  CP BRIDGE DEEP DIVE VIEW
// ============================================================

const CPBridgeView = () => {
  const {
    CP_GROUP,
    TE_MATRIX,
    CP_METRICS,
    AGENT_FE,
    ETA_BEFORE,
  } = useVizData();
  const [hoveredNode, setHoveredNode] = useState(null);
  const [hoveredEdge, setHoveredEdge] = useState(null);
  const nCoils = Math.max(CP_GROUP.length, 1);

  // TE network: circular layout
  const teNodes = useMemo(() =>
    Array.from({ length: nCoils }, (_, i) => {
      const angle = (i / nCoils) * 2 * Math.PI - Math.PI / 2;
      return { id: i, x: 140 + 110 * Math.cos(angle), y: 140 + 110 * Math.sin(angle), group: CP_GROUP[i] };
    }), [CP_GROUP, nCoils]);

  const teEdges = useMemo(() => {
    const edges = [];
    for (let i = 0; i < nCoils; i++) {
      for (let j = i + 1; j < nCoils; j++) {
        const te = Math.max(TE_MATRIX?.[i]?.[j] || 0, TE_MATRIX?.[j]?.[i] || 0);
        if (te > 1e-8) edges.push({ src: i, dst: j, te, logTe: Math.log10(te + 1e-20) });
      }
    }
    return edges.sort((a, b) => b.te - a.te).slice(0, 25);
  }, [TE_MATRIX, nCoils]);

  const teMax = Math.max(...teEdges.map(e => e.te), 1e-12);

  const morlRadarData = useMemo(() => [
    { axis: "J_\u03a6", value: CP_METRICS.morl.J_phi / 12.376, raw: CP_METRICS.morl.J_phi },
    { axis: "J_arch", value: CP_METRICS.morl.J_arch / 12.376, raw: CP_METRICS.morl.J_arch },
    { axis: "J_sync", value: CP_METRICS.morl.J_sync / 12.376, raw: CP_METRICS.morl.J_sync },
    { axis: "J_task", value: CP_METRICS.morl.J_task / 12.376, raw: CP_METRICS.morl.J_task },
  ], []);

  const agentData = useMemo(() =>
    AGENT_FE.map((fe, i) => ({
      coil: `C${i} (${CP_GROUP[i] === 0 ? "M" : "F"})`,
      fe: fe * 1e5, // scale for visibility
      eta: ETA_BEFORE[i],
      group: CP_GROUP[i],
    })), []);

  const energyStack = useMemo(() => [
    { name: "E_nca", value: CP_METRICS.energy.E_nca, color: COLORS.cyan },
    { name: "E_mf", value: CP_METRICS.energy.E_mf, color: COLORS.teal },
    { name: "E_arch", value: CP_METRICS.energy.E_arch, color: COLORS.gold },
    { name: "E_phi", value: Math.abs(CP_METRICS.energy.E_phi), color: COLORS.purple, negative: true },
    { name: "E_morl", value: CP_METRICS.energy.E_morl, color: COLORS.orange },
  ], []);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* Transfer Entropy Network */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Transfer Entropy Network</div>
        <svg width={280} height={280} viewBox="0 0 280 280">
          {/* Edges */}
          {teEdges.map((e, i) => {
            const src = teNodes[e.src], dst = teNodes[e.dst];
            const w = Math.max(0.5, (e.te / teMax) * 4);
            const isHov = hoveredEdge === i;
            return (
              <line key={`e${i}`} x1={src.x} y1={src.y} x2={dst.x} y2={dst.y}
                stroke={isHov ? COLORS.gold : COLORS.cyan} strokeWidth={w} strokeOpacity={isHov ? 0.9 : 0.25}
                onMouseEnter={() => setHoveredEdge(i)} onMouseLeave={() => setHoveredEdge(null)} style={{ cursor: "pointer" }} />
            );
          })}
          {/* Nodes */}
          {teNodes.map(n => {
            const isHov = hoveredNode === n.id;
            return (
              <g key={`n${n.id}`} onMouseEnter={() => setHoveredNode(n.id)} onMouseLeave={() => setHoveredNode(null)} style={{ cursor: "pointer" }}>
                <circle cx={n.x} cy={n.y} r={isHov ? 12 : 9}
                  fill={n.group === 0 ? COLORS.cyan : COLORS.magenta} fillOpacity={isHov ? 1 : 0.7}
                  stroke={isHov ? COLORS.gold : "none"} strokeWidth={2} />
                <text x={n.x} y={n.y + 3} textAnchor="middle" fill={COLORS.bg} fontSize={8} fontWeight={700}>{n.id}</text>
              </g>
            );
          })}
        </svg>
        <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
          <span style={{ color: COLORS.cyan, fontSize: 10 }}>\u25cf Group M (0)</span>
          <span style={{ color: COLORS.magenta, fontSize: 10 }}>\u25cf Group F (1)</span>
        </div>
        {hoveredNode !== null && (
          <div style={{ color: COLORS.gold, fontSize: 11, marginTop: 4, fontFamily: "monospace" }}>
            Coil {hoveredNode} | Group: {CP_GROUP[hoveredNode] === 0 ? "M" : "F"} | FE: {fmtSci(AGENT_FE[hoveredNode])} | \u03b7: {ETA_BEFORE[hoveredNode]}
          </div>
        )}
        {hoveredEdge !== null && (
          <div style={{ color: COLORS.gold, fontSize: 11, marginTop: 4, fontFamily: "monospace" }}>
            Edge {teEdges[hoveredEdge].src}\u2194{teEdges[hoveredEdge].dst} | TE: {teEdges[hoveredEdge].te.toExponential(2)}
          </div>
        )}
      </div>

      {/* MORL Radar */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>MORL Objective Space</div>
        <ResponsiveContainer width="100%" height={200}>
          <RadarChart data={morlRadarData}>
            <PolarGrid stroke="#444" />
            <PolarAngleAxis dataKey="axis" tick={{ fill: COLORS.text, fontSize: 11 }} />
            <PolarRadiusAxis domain={[0, 1.1]} tick={false} axisLine={false} />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]?.payload;
              return (
                <div style={{ background: COLORS.card, border: `1px solid ${COLORS.magenta}`, borderRadius: 6, padding: "6px 10px" }}>
                  <div style={{ color: COLORS.magenta, fontSize: 12 }}>{d.axis}</div>
                  <div style={{ color: COLORS.text, fontSize: 11 }}>Raw: {fmt(d.raw, 4)}</div>
                  <div style={{ color: COLORS.textDim, fontSize: 10 }}>Normalized: {fmt(d.value, 4)}</div>
                </div>
              );
            }} />
            <Radar dataKey="value" stroke={COLORS.magenta} fill={COLORS.magenta} fillOpacity={0.2} strokeWidth={2} dot={{ fill: COLORS.magenta, r: 3 }} />
          </RadarChart>
        </ResponsiveContainer>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginTop: 8 }}>
          {Object.entries(CP_METRICS.morl).map(([k, v]) => (
            <div key={k} style={{ background: COLORS.bg, borderRadius: 4, padding: "4px 8px", display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: COLORS.textDim, fontSize: 10 }}>{k}</span>
              <span style={{ color: COLORS.magenta, fontSize: 11, fontFamily: "monospace" }}>{fmt(v, 4)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Agent Dashboard */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Agent Free Energies & \u03b7-Field</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={agentData} layout="vertical" margin={{ left: 50 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis type="number" stroke={COLORS.textDim} fontSize={9} label={{ value: "FE (\u00d710\u207b\u2075)", position: "bottom", fill: COLORS.textDim, fontSize: 9 }} />
            <YAxis type="category" dataKey="coil" stroke={COLORS.textDim} fontSize={8} width={50} />
            <Tooltip content={<DarkTooltip />} />
            <Bar dataKey="fe" name="Free Energy (\u00d710\u207b\u2075)">
              {agentData.map((d, i) => <Cell key={i} fill={d.group === 0 ? COLORS.cyan : COLORS.magenta} fillOpacity={0.7} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Energy Functional */}
      <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
        <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 600, marginBottom: 12 }}>4D NCA Energy Functional Breakdown</div>
        {energyStack.map((comp, i) => {
          const pct = Math.abs(comp.value) / CP_METRICS.energy.E_total * 100;
          return (
            <div key={i} style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                <span style={{ color: COLORS.textDim, fontSize: 11 }}>{comp.name}{comp.negative ? " (neg)" : ""}</span>
                <span style={{ color: comp.color, fontSize: 12, fontFamily: "monospace" }}>{comp.negative ? "-" : ""}{comp.value.toFixed(3)}</span>
              </div>
              <div style={{ height: 10, background: COLORS.bg, borderRadius: 5, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${Math.min(100, pct)}%`, background: comp.color, borderRadius: 5, opacity: comp.negative ? 0.5 : 0.8 }} />
              </div>
            </div>
          );
        })}
        <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 8, marginTop: 8, display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: COLORS.text, fontSize: 13, fontWeight: 600 }}>E_total</span>
          <span style={{ color: COLORS.gold, fontSize: 14, fontFamily: "monospace", fontWeight: 700 }}>{CP_METRICS.energy.E_total.toFixed(3)} mJ</span>
        </div>
        <div style={{ marginTop: 12, padding: 8, background: COLORS.bg, borderRadius: 6 }}>
          <div style={{ color: COLORS.textDim, fontSize: 10, marginBottom: 4 }}>Implacement Status</div>
          <div style={{ color: COLORS.text, fontSize: 11 }}>Pairs detected: <span style={{ color: COLORS.lime }}>0</span></div>
          <div style={{ color: COLORS.text, fontSize: 11 }}>\u0394\u03a6 from implacement: <span style={{ color: COLORS.lime }}>0.0000</span></div>
          <div style={{ color: COLORS.text, fontSize: 11 }}>\u03b7-field unchanged (no M\u2192F locking)</div>
        </div>
      </div>
    </div>
  );
};

// ============================================================
// §8  ROOT COMPONENT
// ============================================================

const TABS = [
  { id: "overview", label: "System Overview", icon: Activity },
  { id: "worlds", label: "Worlds & Trajectory", icon: GitBranch },
  { id: "control", label: "Control & Sensitivity", icon: Zap },
  { id: "cp", label: "CP Bridge", icon: Brain },
];

export default function OmnidreamVisualization() {
  const [activeView, setActiveView] = useState("overview");
  const [timeStep, setTimeStep] = useState(0);
  const [vizData, setVizData] = useState(FALLBACK_VIZ_DATA);
  const [dataState, setDataState] = useState({
    status: "loading",
    source: "fallback",
    baseUrl: "./pipeline_output",
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    const configuredBase =
      typeof window !== "undefined" && window.OMNIDREAM_DATA_BASE_URL
        ? window.OMNIDREAM_DATA_BASE_URL
        : "./pipeline_output";

    setDataState((prev) => ({ ...prev, status: "loading", baseUrl: configuredBase }));
    (async () => {
      try {
        const bundle = await loadOmnidreamBundleFromHttp(configuredBase);
        if (cancelled) return;
        setVizData(buildVizDataFromBundle(bundle));
        setDataState({
          status: "ready",
          source: "live",
          baseUrl: configuredBase,
          error: null,
        });
      } catch (err) {
        if (cancelled) return;
        setDataState({
          status: "fallback",
          source: "fallback",
          baseUrl: configuredBase,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const maxStep = Math.max(0, (vizData.TRAJ_ENERGY?.length || 1) - 1);
    setTimeStep((prev) => Math.max(0, Math.min(prev, maxStep)));
  }, [vizData]);

  const statusColor = dataState.source === "live" ? COLORS.lime : COLORS.gold;
  const summaryMode = vizData?.SUMMARY?.mode || "NTS";
  const summaryCoils = vizData?.SUMMARY?.num_coils || vizData?.CP_GROUP?.length || 0;
  const summaryStages = vizData?.PIPELINE_STAGES?.filter((s) => s.done).length || 0;

  return (
    <VizDataContext.Provider value={vizData}>
      <div style={{ minHeight: "100vh", background: COLORS.bg, color: COLORS.text, fontFamily: "'Inter', system-ui, sans-serif" }}>
      {/* Header */}
      <div style={{ background: COLORS.card, borderBottom: `1px solid ${COLORS.border}`, padding: "12px 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <Network size={20} color={COLORS.cyan} />
          <span style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.02em" }}>
            <span style={{ color: COLORS.cyan }}>Omni</span><span style={{ color: COLORS.magenta }}>dream</span>
          </span>
          <span style={{ color: COLORS.textDim, fontSize: 12, marginLeft: 8 }}>TMS Coil Array Visualization</span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <span style={{ color: COLORS.lime, fontSize: 10, padding: "2px 8px", background: COLORS.lime + "20", borderRadius: 4 }}>{summaryMode} Mode</span>
          <span style={{ color: COLORS.cyan, fontSize: 10, padding: "2px 8px", background: COLORS.cyan + "20", borderRadius: 4 }}>{summaryCoils} Coils</span>
          <span style={{ color: COLORS.gold, fontSize: 10, padding: "2px 8px", background: COLORS.gold + "20", borderRadius: 4 }}>{summaryStages} Stages</span>
          <span style={{ color: statusColor, fontSize: 10, padding: "2px 8px", background: statusColor + "20", borderRadius: 4 }}>
            {dataState.status === "loading" ? "Loading data" : dataState.source === "live" ? "Live data" : "Fallback data"}
          </span>
        </div>
      </div>
      {dataState.error && (
        <div style={{ padding: "6px 24px", background: COLORS.card, borderBottom: `1px solid ${COLORS.border}` }}>
          <span style={{ color: COLORS.gold, fontSize: 10 }}>
            Data loader fallback: {dataState.error}
          </span>
        </div>
      )}

      {/* Tab Bar */}
      <div style={{ background: COLORS.card, borderBottom: `1px solid ${COLORS.border}`, padding: "0 24px", display: "flex", gap: 0 }}>
        {TABS.map(tab => {
          const Icon = tab.icon;
          const active = activeView === tab.id;
          return (
            <button key={tab.id} onClick={() => setActiveView(tab.id)}
              style={{
                display: "flex", alignItems: "center", gap: 6, padding: "10px 16px",
                background: "transparent", border: "none", cursor: "pointer",
                color: active ? COLORS.cyan : COLORS.textDim,
                borderBottom: active ? `2px solid ${COLORS.cyan}` : "2px solid transparent",
                fontSize: 12, fontWeight: active ? 600 : 400,
                transition: "all 0.2s",
              }}>
              <Icon size={14} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Main Content */}
      <div style={{ padding: 24, maxWidth: 1200, margin: "0 auto" }}>
        {activeView === "overview" && <SystemOverview />}
        {activeView === "worlds" && <WorldsTrajectoryView timeStep={timeStep} setTimeStep={setTimeStep} />}
        {activeView === "control" && <ControlSensitivityView />}
        {activeView === "cp" && <CPBridgeView />}
      </div>
    </div>
    </VizDataContext.Provider>
  );
}
