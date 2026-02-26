const TAU = Math.PI * 2;

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const wrapTau = (a) => {
  let v = a % TAU;
  if (v < 0) v += TAU;
  return v;
};
const randRange = (min, max) => min + Math.random() * (max - min);
const signedNoise = () => (Math.random() * 2 - 1);
const tanh = Math.tanh;
const fmtSeconds = (v) => `${Math.max(0, v).toFixed(1)}s`;

const MODE_INFO = {
  mlp: {
    title: "MLP Coil Brain",
    description:
      "A feed-forward controller maps plasma sensors to toroidal and poloidal coil corrections.",
  },
  cnn: {
    title: "CNN Flux Vision",
    description:
      "Circular convolution scans the torus for unstable hotspots and issues localized magnetic pulses.",
  },
  snn: {
    title: "SNN Spike Grid",
    description:
      "Spiking neurons fire around the torus as event-based coil triggers, producing rhythmic confinement pulses.",
  },
};

const MISSION_PHASE_TEMPLATES = [
  {
    id: "stabilize-edge",
    title: "Phase 1: Edge Stabilization",
    mode: "mlp",
    hint: "Use MLP mode. Raise gain and keep instability below 35% while maintaining confinement above 65%.",
    duration: 42,
    progressNeeded: 18,
  },
  {
    id: "quench-hotspots",
    title: "Phase 2: Hotspot Quench",
    mode: "cnn",
    hint: "Use CNN mode. Identify and smooth hotspots: keep coherence high and instability moderate.",
    duration: 40,
    progressNeeded: 16,
  },
  {
    id: "spike-burn",
    title: "Phase 3: Spike Burn Window",
    mode: "snn",
    hint: "Use SNN mode. Time manual pulses to sustain yield above 45% without destabilizing the core.",
    duration: 36,
    progressNeeded: 14,
  },
];

class NeuroFusionReactor {
  constructor(canvas, ui) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true });
    this.ui = ui;

    this.config = {
      mode: "mlp",
      paused: false,
      heating: 0.72,
      gain: 0.61,
      adapt: 0.38,
      turbulence: 0.24,
      masterVolume: 0.42,
      zoom: 1.0,
      yaw: 0.76,
      pitch: -0.34,
      autoOrbit: 1,
      userOrbitImpulse: 0,
    };

    this.viewport = { width: 0, height: 0, dpr: 1, cx: 0, cy: 0 };
    this.time = 0;
    this.lastTime = 0;
    this.frameCount = 0;
    this.metrics = {
      temperature: 0.72,
      density: 0.52,
      confinement: 0.54,
      instability: 0.26,
      yield: 0.12,
      coherence: 0.58,
      fusionPotential: 0,
    };
    this.coils = { toroidal: 0, poloidal: 0, pulse: 0 };

    this.R = 1.34;
    this.r = 0.42;
    this.maxRadial = 0.19;
    this.sensorCount = 36;
    this.trailDepth = 14;

    this.sensors = new Float32Array(this.sensorCount);
    this.sensorBlur = new Float32Array(this.sensorCount);
    this.sensorDelta = new Float32Array(this.sensorCount);
    this.sensorTrail = Array.from({ length: this.trailDepth }, () => new Float32Array(this.sensorCount));
    this.sensorTrailAge = 0;

    this.outputs = [0, 0, 0];
    this.outputsTarget = [0, 0, 0];
    this.manualPulseBoost = 0;
    this.audioTelemetry = {
      beat: 0,
      burstImpulse: 0,
      spikeImpulse: 0,
      lastBurstCount: 0,
    };
    this.audioEngine = null;
    this.challenge = this._createChallengeState();

    this.particles = [];
    this.bursts = [];
    this.stars = Array.from({ length: 160 }, () => ({
      x: Math.random(),
      y: Math.random(),
      z: Math.random(),
      twinkle: Math.random() * TAU,
      amp: Math.random() * 0.8 + 0.2,
    }));

    this.pointer = {
      down: false,
      id: null,
      x: 0,
      y: 0,
      lastX: 0,
      lastY: 0,
    };

    this.mlp = this._createMLP();
    this.cnn = this._createCNN();
    this.snn = this._createSNN();
    this._seedParticles();
  }

  _createChallengeState() {
    return {
      active: false,
      state: "sandbox", // sandbox | running | won | lost
      timeLeft: 118,
      score: 0,
      multiplier: 1,
      streak: 0,
      integrity: 1,
      phaseIndex: 0,
      phaseProgress: 0,
      phases: MISSION_PHASE_TEMPLATES.map((p) => ({ ...p })),
      phaseStartedAt: 0,
      manualPulseCharge: 1,
      manualPulseCooldown: 0,
      statusText: "Press Enter or Start Challenge to begin mission mode.",
      statusTone: "warn",
      objectiveText: "Objective: Start the challenge. Stabilize the reactor while adapting the neural controller.",
      hintText: "Shortcuts: 1/2/3 modes, arrows tune heating/gain, A/B tune adapt/turbulence, Space pulse, F fullscreen.",
      bridgeExportCount: 0,
      flash: 0,
      modeUsage: { mlp: 0, cnn: 0, snn: 0 },
      modeSwitches: 0,
      lastMode: "mlp",
      manualPulseCount: 0,
      scoreBreakdown: {
        base: 0,
        modeBonus: 0,
        modePenalty: 0,
        phaseClear: 0,
        pulseBonus: 0,
      },
      phaseClearTimes: [],
      summary: null,
      reportVisible: false,
      lastImportedProfile: null,
    };
  }

  _createMLP() {
    const inCount = 5;
    const hiddenCount = 8;
    const outCount = 3;
    const w1 = Array.from({ length: hiddenCount }, () =>
      Array.from({ length: inCount }, () => randRange(-1.0, 1.0))
    );
    const b1 = Array.from({ length: hiddenCount }, () => randRange(-0.4, 0.4));
    const w2 = Array.from({ length: outCount }, () =>
      Array.from({ length: hiddenCount }, () => randRange(-1.0, 1.0))
    );
    const b2 = Array.from({ length: outCount }, () => randRange(-0.3, 0.3));

    return {
      inCount,
      hiddenCount,
      outCount,
      w1,
      b1,
      w2,
      b2,
      inputs: new Float32Array(inCount),
      hidden: new Float32Array(hiddenCount),
      outputs: new Float32Array(outCount),
      desired: new Float32Array(outCount),
    };
  }

  _createCNN() {
    const channels = 3;
    const kernelSize = 5;
    return {
      channels,
      kernelSize,
      kernels: Array.from({ length: channels }, () =>
        Array.from({ length: kernelSize }, () => randRange(-1.0, 1.0))
      ),
      bias: Array.from({ length: channels }, () => randRange(-0.35, 0.35)),
      mix: Array.from({ length: 3 }, () =>
        Array.from({ length: channels }, () => randRange(-1.1, 1.1))
      ),
      mixBias: Array.from({ length: 3 }, () => randRange(-0.25, 0.25)),
      features: Array.from({ length: channels }, () => new Float32Array(this.sensorCount)),
      pooled: new Float32Array(channels),
      peaks: new Float32Array(channels),
      hotIndex: 0,
    };
  }

  _createSNN() {
    const count = 24;
    return {
      count,
      neurons: Array.from({ length: count }, () => ({
        v: randRange(-0.1, 0.5),
        threshold: randRange(0.95, 1.15),
        refractory: 0,
        spikeGlow: 0,
      })),
      prevSpikes: new Float32Array(count),
      spikes: new Float32Array(count),
      spikeRate: 0,
      directionality: 0,
      pulseMemory: 0,
    };
  }

  reseed() {
    this.bursts.length = 0;
    this._seedParticles();
    this.mlp = this._createMLP();
    this.cnn = this._createCNN();
    this.snn = this._createSNN();
    this.outputs = [0, 0, 0];
    this.outputsTarget = [0, 0, 0];
    this.manualPulseBoost = 0;
    this.audioTelemetry.beat = 0;
    this.audioTelemetry.burstImpulse = 0;
    this.audioTelemetry.spikeImpulse = 0;
    for (const arr of this.sensorTrail) arr.fill(0);
    this.sensorTrailAge = 0;
    if (this.challenge.state === "running") {
      this.startChallenge(true);
    } else {
      this.challenge = this._createChallengeState();
    }
  }

  setMode(mode) {
    if (!MODE_INFO[mode]) return;
    const prev = this.config.mode;
    this.config.mode = mode;
    if (this.challenge?.state === "running" && prev && prev !== mode) {
      this.challenge.modeSwitches += 1;
      this.challenge.lastMode = mode;
      this.challenge.flash = Math.max(this.challenge.flash, 0.35);
    }
  }

  setControl(name, raw) {
    this.config[name] = clamp(raw, 0, 1);
  }

  setPaused(v) {
    this.config.paused = Boolean(v);
  }

  setAudioEngine(engine) {
    this.audioEngine = engine;
  }

  resize(widthCss, heightCss, dpr = window.devicePixelRatio || 1) {
    const safeDpr = Math.min(2, Math.max(1, dpr));
    const width = Math.max(1, Math.floor(widthCss * safeDpr));
    const height = Math.max(1, Math.floor(heightCss * safeDpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.viewport.width = width;
    this.viewport.height = height;
    this.viewport.dpr = safeDpr;
    this.viewport.cx = width * 0.52;
    this.viewport.cy = height * 0.57;
  }

  _getCurrentPhase() {
    return this.challenge.phases[this.challenge.phaseIndex] || null;
  }

  startChallenge(fromReseed = false) {
    const prevExports = this.challenge?.bridgeExportCount || 0;
    const prevImported = this.challenge?.lastImportedProfile || null;
    this.challenge = this._createChallengeState();
    this.challenge.active = true;
    this.challenge.state = "running";
    this.challenge.timeLeft = this.challenge.phases.reduce((sum, p) => sum + p.duration, 0);
    this.challenge.score = 0;
    this.challenge.phaseIndex = 0;
    this.challenge.phaseProgress = 0;
    this.challenge.integrity = 1;
    this.challenge.multiplier = 1;
    this.challenge.streak = 0;
    this.challenge.phaseStartedAt = this.time;
    this.challenge.bridgeExportCount = prevExports;
    this.challenge.lastImportedProfile = prevImported;
    this.challenge.statusTone = "live";
    this.challenge.flash = 0;
    this.challenge.reportVisible = false;
    this.manualPulseBoost = 0;
    if (!fromReseed) {
      this.reseed();
      return;
    }
    this._setModeForMissionPhase(this._getCurrentPhase());
    this._syncMissionCopy();
  }

  restartChallenge() {
    this.startChallenge();
  }

  setReportVisible(visible) {
    this.challenge.reportVisible = Boolean(visible);
  }

  _addMissionScore(amount, bucket = "base") {
    const mission = this.challenge;
    if (!Number.isFinite(amount) || amount === 0) return;
    mission.score = Math.max(0, mission.score + amount);
    if (mission.scoreBreakdown[bucket] === undefined) mission.scoreBreakdown[bucket] = 0;
    mission.scoreBreakdown[bucket] += amount;
  }

  _captureMissionSummary(outcome, reason = "") {
    const mission = this.challenge;
    const totalScore = Math.round(mission.score);
    const modeUsage = {
      mlp: Number(mission.modeUsage.mlp.toFixed(1)),
      cnn: Number(mission.modeUsage.cnn.toFixed(1)),
      snn: Number(mission.modeUsage.snn.toFixed(1)),
    };
    mission.summary = {
      outcome,
      reason,
      totalScore,
      timeLeft: Number(mission.timeLeft.toFixed(1)),
      integrityPct: Math.round(mission.integrity * 100),
      phaseCleared: Math.min(mission.phases.length, mission.phaseIndex + (outcome === "won" ? 0 : 0)),
      modeSwitches: mission.modeSwitches,
      manualPulseCount: mission.manualPulseCount,
      modeUsage,
      scoreBreakdown: {
        base: Math.round(mission.scoreBreakdown.base),
        modeBonus: Math.round(mission.scoreBreakdown.modeBonus),
        modePenalty: Math.round(mission.scoreBreakdown.modePenalty),
        phaseClear: Math.round(mission.scoreBreakdown.phaseClear),
        pulseBonus: Math.round(mission.scoreBreakdown.pulseBonus),
      },
      phaseClearTimes: mission.phaseClearTimes.map((v) => Number(v.toFixed(1))),
    };
    mission.reportVisible = true;
  }

  _refreshBridgePreviewUI(kind = "cpp") {
    if (!this.ui?.bridgePreview) return;
    this.ui.bridgePreview.textContent = kind === "json"
      ? JSON.stringify(this.getBridgeProfile(), null, 2).slice(0, 2200)
      : this.getBridgeCppSnippet();
  }

  applyBridgeProfile(profile) {
    if (!profile || typeof profile !== "object") {
      throw new Error("Invalid profile: expected object");
    }
    if (profile.controls) {
      if (Number.isFinite(profile.controls.heatingDrive)) this.setControl("heating", profile.controls.heatingDrive);
      if (Number.isFinite(profile.controls.magneticGain)) this.setControl("gain", profile.controls.magneticGain);
      if (Number.isFinite(profile.controls.adaptiveLearning)) this.setControl("adapt", profile.controls.adaptiveLearning);
      if (Number.isFinite(profile.controls.turbulenceBias)) this.setControl("turbulence", profile.controls.turbulenceBias);
      if (Number.isFinite(profile.controls.masterVolume)) this.config.masterVolume = clamp(profile.controls.masterVolume, 0, 1);
    }
    if (typeof profile.mode === "string") this.setMode(profile.mode.toLowerCase());
    if (profile.camera) {
      if (Number.isFinite(profile.camera.yaw)) this.config.yaw = profile.camera.yaw;
      if (Number.isFinite(profile.camera.pitch)) this.config.pitch = clamp(profile.camera.pitch, -1.15, 1.1);
      if (Number.isFinite(profile.camera.zoom)) this.config.zoom = clamp(profile.camera.zoom, 0.72, 1.55);
    }
    this.challenge.lastImportedProfile = profile;
    if (this.audioEngine) this.audioEngine.setVolume(this.config.masterVolume);
    if (this.challenge.state !== "running") this.challenge.statusText = "Bridge profile applied to sandbox controls.";
    this._refreshBridgePreviewUI("json");
    return profile;
  }

  importBridgeProfileText(text) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (err) {
      throw new Error(`JSON parse failed: ${err.message}`);
    }
    return this.applyBridgeProfile(parsed);
  }

  saveBridgePresetLocal() {
    const key = "neurofusion_bridge_preset_v1";
    const text = JSON.stringify(this.getBridgeProfile());
    localStorage.setItem(key, text);
    return text;
  }

  loadBridgePresetLocal() {
    const key = "neurofusion_bridge_preset_v1";
    const text = localStorage.getItem(key);
    if (!text) throw new Error("No local preset found.");
    return this.importBridgeProfileText(text);
  }

  _syncMissionCopy() {
    const mission = this.challenge;
    const phase = this._getCurrentPhase();
    if (mission.state === "sandbox") {
      mission.statusText = "Press Enter or Start Challenge to begin mission mode.";
      mission.statusTone = "warn";
      mission.objectiveText =
        "Objective: Start the challenge. Stabilize the reactor while adapting the neural controller.";
      mission.hintText =
        "Shortcuts: 1/2/3 modes, arrows tune heating/gain, A/B tune adapt/turbulence, Space pulse, F fullscreen.";
      return;
    }
    if (mission.state === "won") {
      mission.statusText = "Mission complete. Neural tokamak achieved controlled burn.";
      mission.statusTone = "live";
      mission.objectiveText = "Objective complete. Press Enter to run a new mission seed.";
      mission.hintText = "Export a bridge preset for FusionCpp or replay with a different strategy.";
      return;
    }
    if (mission.state === "lost") {
      mission.statusText = "Containment failed. Press Enter / Start to restart the mission.";
      mission.statusTone = "danger";
      mission.objectiveText = "Objective failed. Restore integrity and complete all three phases.";
      mission.hintText = "Try lowering turbulence and raising magnetic gain before pushing yield.";
      return;
    }
    if (!phase) return;
    const pPct = Math.round(
      clamp(mission.phaseProgress / Math.max(phase.progressNeeded, 0.01), 0, 1) * 100
    );
    mission.statusText = `${phase.title} (${pPct}%): ${phase.hint}`;
    mission.statusTone = "live";
    mission.objectiveText = `Objective: ${phase.title}. Required mode: ${phase.mode.toUpperCase()}.`;
    mission.hintText = phase.hint;
  }

  triggerManualPulse(intensity = 1) {
    const mission = this.challenge;
    if (mission.state === "running") {
      if (mission.manualPulseCooldown > 0 || mission.manualPulseCharge < 0.18) return false;
      mission.manualPulseCharge = Math.max(0, mission.manualPulseCharge - 0.25);
      mission.manualPulseCooldown = 0.32;
      mission.flash = Math.max(mission.flash, 0.85);
      mission.statusText = "Manual pulse injected.";
      mission.statusTone = "warn";
      mission.manualPulseCount += 1;
      if (this._getCurrentPhase()?.id === "spike-burn") {
        this._addMissionScore(24, "pulseBonus");
      }
    }
    this.manualPulseBoost = clamp(this.manualPulseBoost + 0.8 * intensity, 0, 1.4);
    this.audioTelemetry.beat = Math.max(this.audioTelemetry.beat, 0.8);
    this.audioTelemetry.spikeImpulse += 0.25 * intensity;
    return true;
  }

  _setModeForMissionPhase(phase) {
    if (!phase) return;
    this.setMode(phase.mode);
  }

  _advanceMissionPhase() {
    const mission = this.challenge;
    const phase = this._getCurrentPhase();
    if (!phase) return;
    const clearBonus = Math.round(800 + mission.timeLeft * 5 + mission.integrity * 350);
    this._addMissionScore(clearBonus, "phaseClear");
    mission.phaseClearTimes.push(this.time - mission.phaseStartedAt);
    mission.phaseIndex += 1;
    mission.phaseProgress = 0;
    mission.streak = 0;
    mission.multiplier = Math.min(5, mission.multiplier + 0.35);
    mission.flash = 1;
    mission.phaseStartedAt = this.time;

    if (mission.phaseIndex >= mission.phases.length) {
      mission.state = "won";
      mission.active = false;
      mission.statusTone = "live";
      mission.statusText = "Mission complete. Controlled burn locked in.";
      mission.objectiveText = "All phases cleared. Press Enter to run another mission.";
      mission.hintText = "Review the mission report, then export/import a bridge preset or replay.";
      this._captureMissionSummary("won", "Controlled burn achieved across all phases.");
      return;
    }

    const nextPhase = this._getCurrentPhase();
    this._setModeForMissionPhase(nextPhase);
    this._syncMissionCopy();
  }

  _failMission(reason) {
    const mission = this.challenge;
    mission.state = "lost";
    mission.active = false;
    mission.statusTone = "danger";
    mission.statusText = reason;
    mission.objectiveText = "Press Enter or Start Challenge to restart the mission.";
    mission.flash = 1;
    mission.hintText = "Review the report for penalties and mode usage, then retry.";
    this._captureMissionSummary("lost", reason);
  }

  _updateMission(dt, plasmaSnapshot) {
    const mission = this.challenge;
    mission.flash = Math.max(0, mission.flash - dt * 1.6);
    mission.manualPulseCooldown = Math.max(0, mission.manualPulseCooldown - dt);
    mission.manualPulseCharge = clamp(
      mission.manualPulseCharge + dt * (0.18 + (1 - this.metrics.instability) * 0.22),
      0,
      1
    );

    if (mission.state !== "running") {
      this._syncMissionCopy();
      return;
    }

    mission.timeLeft = Math.max(0, mission.timeLeft - dt);
    const phase = this._getCurrentPhase();
    if (!phase) {
      this._syncMissionCopy();
      return;
    }

    const modeMatch = this.config.mode === phase.mode;
    const m = this.metrics;
    if (mission.modeUsage[this.config.mode] !== undefined) {
      mission.modeUsage[this.config.mode] += dt;
    }

    let stableScore = 0;
    if (phase.id === "stabilize-edge") {
      stableScore =
        (m.confinement - 0.62) * 2.3 +
        (0.38 - m.instability) * 1.8 +
        (m.coherence - 0.55) * 1.2;
    } else if (phase.id === "quench-hotspots") {
      const hotspotContrast = Math.max(0, plasmaSnapshot.peak - plasmaSnapshot.avg);
      stableScore =
        (0.72 - hotspotContrast * 1.8) * 1.25 +
        (m.coherence - 0.58) * 1.4 +
        (0.48 - m.instability) * 1.3;
    } else if (phase.id === "spike-burn") {
      stableScore =
        (m.yield - 0.42) * 1.8 +
        (m.coherence - 0.52) * 1.1 +
        (0.58 - m.instability) * 1.2 +
        (this.snn.spikeRate - 0.08) * 1.4;
    }

    if (modeMatch) stableScore += 0.25;
    else stableScore -= 0.35;

    const positive = stableScore > 0;
    if (positive) {
      mission.phaseProgress = clamp(
        mission.phaseProgress + dt * (0.65 + stableScore * 0.65),
        0,
        phase.progressNeeded
      );
      mission.streak = clamp(mission.streak + dt, 0, 16);
    } else {
      mission.phaseProgress = Math.max(0, mission.phaseProgress - dt * (0.22 + Math.abs(stableScore) * 0.18));
      mission.streak = Math.max(0, mission.streak - dt * 2.0);
    }

    mission.multiplier = clamp(1 + mission.streak / 5, 1, 4.5);
    const baseRate =
      (m.yield * 280 + m.confinement * 120 + m.coherence * 70 - m.instability * 90) *
      mission.multiplier;

    let modeBonusRate = 0;
    let modePenaltyRate = 0;
    if (modeMatch) {
      if (phase.id === "stabilize-edge") {
        modeBonusRate = Math.max(0, (m.confinement - 0.7) * 180 + (0.28 - m.instability) * 120);
      } else if (phase.id === "quench-hotspots") {
        const hotspotContrast = Math.max(0, plasmaSnapshot.peak - plasmaSnapshot.avg);
        modeBonusRate = Math.max(0, (0.22 - hotspotContrast) * 220 + (m.coherence - 0.78) * 140);
      } else if (phase.id === "spike-burn") {
        modeBonusRate = Math.max(0, (m.yield - 0.46) * 260 + (this.snn.spikeRate - 0.12) * 160);
      }
    } else {
      modePenaltyRate = 34 + Math.max(0, m.instability - 0.45) * 85;
    }

    if (baseRate !== 0) this._addMissionScore(baseRate * dt, "base");
    if (modeBonusRate > 0) this._addMissionScore(modeBonusRate * dt, "modeBonus");
    if (modePenaltyRate > 0) this._addMissionScore(-modePenaltyRate * dt, "modePenalty");

    const damage =
      Math.max(0, m.instability - 0.72) * 0.55 +
      Math.max(0, m.temperature - 0.93) * 0.25 +
      (modeMatch ? 0 : 0.035);
    const repair = Math.max(0, m.confinement - 0.68) * 0.12 + Math.max(0, m.coherence - 0.7) * 0.08;
    mission.integrity = clamp(mission.integrity + (repair - damage) * dt, 0, 1);

    if (mission.phaseProgress >= phase.progressNeeded) {
      this._advanceMissionPhase();
    } else if (mission.integrity <= 0) {
      this._failMission("Containment failed: vessel integrity reached 0%.");
    } else if (mission.timeLeft <= 0) {
      this._failMission("Mission timer expired before all phases were completed.");
    } else {
      this._syncMissionCopy();
      if (!modeMatch) {
        mission.statusTone = "warn";
        mission.statusText = `Switch to ${phase.mode.toUpperCase()} mode for ${phase.title}.`;
      }
    }
  }

  getBridgeProfile() {
    const phase = this._getCurrentPhase();
    return {
      schema: "neurofusion-bridge-v2",
      exportedAt: new Date().toISOString(),
      concept: "NeuroFusion Reactor",
      mode: this.config.mode,
      controls: {
        heatingDrive: Number(this.config.heating.toFixed(4)),
        magneticGain: Number(this.config.gain.toFixed(4)),
        adaptiveLearning: Number(this.config.adapt.toFixed(4)),
        turbulenceBias: Number(this.config.turbulence.toFixed(4)),
        masterVolume: Number(this.config.masterVolume.toFixed(4)),
      },
      coils: {
        toroidal: Number(this.coils.toroidal.toFixed(4)),
        poloidal: Number(this.coils.poloidal.toFixed(4)),
        pulse: Number(this.coils.pulse.toFixed(4)),
      },
      metrics: {
        temperature: Number(this.metrics.temperature.toFixed(4)),
        density: Number(this.metrics.density.toFixed(4)),
        confinement: Number(this.metrics.confinement.toFixed(4)),
        instability: Number(this.metrics.instability.toFixed(4)),
        yield: Number(this.metrics.yield.toFixed(4)),
        coherence: Number(this.metrics.coherence.toFixed(4)),
      },
      mission: {
        state: this.challenge.state,
        timeLeft: Number(this.challenge.timeLeft.toFixed(2)),
        score: Math.round(this.challenge.score),
        integrity: Number(this.challenge.integrity.toFixed(4)),
        modeSwitches: this.challenge.modeSwitches,
        modeUsage: {
          mlp: Number(this.challenge.modeUsage.mlp.toFixed(2)),
          cnn: Number(this.challenge.modeUsage.cnn.toFixed(2)),
          snn: Number(this.challenge.modeUsage.snn.toFixed(2)),
        },
        phase: phase
          ? {
              id: phase.id,
              title: phase.title,
              mode: phase.mode,
              progress: Number(this.challenge.phaseProgress.toFixed(3)),
              progressNeeded: phase.progressNeeded,
            }
          : null,
        scoreBreakdown: Object.fromEntries(
          Object.entries(this.challenge.scoreBreakdown).map(([k, v]) => [k, Math.round(v)])
        ),
      },
      camera: {
        yaw: Number(this.config.yaw.toFixed(4)),
        pitch: Number(this.config.pitch.toFixed(4)),
        zoom: Number(this.config.zoom.toFixed(4)),
      },
      fusionCppMapping: {
        notes: [
          "Map controls.* to ImGui sliders / simulation coefficients in FusionCpp.",
          "Map coils.* to toroidal/poloidal field gain and pulse injection terms.",
          "Map metrics.* to HUD telemetry and adaptive feedback targets.",
          "Mission analytics fields are optional and can be ignored by FusionCpp.",
        ],
        suggestedPresetName: `neurofusion_${this.config.mode}_${this.challenge.state}`,
      },
    };
  }

  getBridgeCppSnippet() {
    const p = this.getBridgeProfile();
    return [
      "// NeuroFusion Reactor bridge preset (generated)",
      "struct NeuroFusionPreset {",
      "  float heatingDrive, magneticGain, adaptiveLearning, turbulenceBias;",
      "  float toroidalOut, poloidalOut, pulseOut;",
      "};",
      "",
      `NeuroFusionPreset preset_${p.mode} = {`,
      `  ${p.controls.heatingDrive}f, ${p.controls.magneticGain}f, ${p.controls.adaptiveLearning}f, ${p.controls.turbulenceBias}f,`,
      `  ${p.coils.toroidal}f, ${p.coils.poloidal}f, ${p.coils.pulse}f`,
      "};",
      "",
      "// Suggested runtime targets:",
      `// confinement=${p.metrics.confinement}, instability=${p.metrics.instability}, yield=${p.metrics.yield}`,
    ].join("\n");
  }

  exportBridgeProfile() {
    const profile = this.getBridgeProfile();
    const text = JSON.stringify(profile, null, 2);
    const blob = new Blob([text], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `neurofusion-bridge-${profile.mode}-${Date.now()}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    this.challenge.bridgeExportCount += 1;
    return text;
  }

  buildTextState() {
    const phase = this._getCurrentPhase();
    const payload = {
      mode: this.config.mode,
      coordinate_system: {
        note: "Torus parameter space; u wraps around major ring [0,2pi), v wraps around minor ring [0,2pi). Screen origin is top-left with +x right, +y down.",
      },
      mission: {
        state: this.challenge.state,
        active: this.challenge.active,
        time_left_s: Number(this.challenge.timeLeft.toFixed(2)),
        score: Math.round(this.challenge.score),
        integrity_pct: Math.round(this.challenge.integrity * 100),
        phase_index: this.challenge.phaseIndex,
        phase_total: this.challenge.phases.length,
        phase: phase
          ? {
              id: phase.id,
              mode: phase.mode,
              progress: Number(this.challenge.phaseProgress.toFixed(2)),
              progress_needed: phase.progressNeeded,
            }
          : null,
        report_visible: this.challenge.reportVisible,
        summary: this.challenge.summary,
      },
      controls: {
        heating: Number(this.config.heating.toFixed(3)),
        gain: Number(this.config.gain.toFixed(3)),
        adapt: Number(this.config.adapt.toFixed(3)),
        turbulence: Number(this.config.turbulence.toFixed(3)),
      },
      metrics: {
        temperature: Number(this.metrics.temperature.toFixed(3)),
        density: Number(this.metrics.density.toFixed(3)),
        confinement: Number(this.metrics.confinement.toFixed(3)),
        instability: Number(this.metrics.instability.toFixed(3)),
        yield: Number(this.metrics.yield.toFixed(3)),
        coherence: Number(this.metrics.coherence.toFixed(3)),
      },
      nn: {
        coils: {
          toroidal: Number(this.coils.toroidal.toFixed(3)),
          poloidal: Number(this.coils.poloidal.toFixed(3)),
          pulse: Number(this.coils.pulse.toFixed(3)),
        },
        cnn_hot_index: this.cnn.hotIndex,
        snn_spike_rate: Number(this.snn.spikeRate.toFixed(3)),
      },
      gameplay: {
        manual_pulse_charge: Number(this.challenge.manualPulseCharge.toFixed(3)),
        manual_pulse_cooldown_s: Number(this.challenge.manualPulseCooldown.toFixed(3)),
        manual_pulse_count: this.challenge.manualPulseCount,
        mode_switches: this.challenge.modeSwitches,
        mode_usage_s: {
          mlp: Number(this.challenge.modeUsage.mlp.toFixed(2)),
          cnn: Number(this.challenge.modeUsage.cnn.toFixed(2)),
          snn: Number(this.challenge.modeUsage.snn.toFixed(2)),
        },
        score_breakdown: Object.fromEntries(
          Object.entries(this.challenge.scoreBreakdown).map(([k, v]) => [k, Math.round(v)])
        ),
      },
    };
    return JSON.stringify(payload, null, 2);
  }

  _seedParticles() {
    this.particles.length = 0;
    const isSmall = window.matchMedia("(max-width: 720px)").matches;
    const count = isSmall ? 360 : 620;
    for (let i = 0; i < count; i++) {
      this.particles.push({
        u: Math.random() * TAU,
        v: Math.random() * TAU,
        du: randRange(0.6, 1.2) * (Math.random() < 0.5 ? -1 : 1),
        dv: randRange(0.4, 0.9) * (Math.random() < 0.5 ? -1 : 1),
        radial: randRange(-this.maxRadial * 0.6, this.maxRadial * 0.6),
        heat: Math.random(),
        charge: Math.random() < 0.5 ? -1 : 1,
        phase: Math.random() * TAU,
      });
    }
  }

  tick(timeMs) {
    const t = timeMs * 0.001;
    if (!this.lastTime) this.lastTime = t;
    let dt = t - this.lastTime;
    this.lastTime = t;
    dt = clamp(dt, 1 / 240, 1 / 20);

    this.step(dt);
    this.renderFrame();
  }

  step(dt) {
    if (!this.config.paused) {
      this.time += dt;
      this._update(dt);
    }
    if (this.audioEngine) {
      this.audioEngine.update(dt);
    }
  }

  renderFrame() {
    this._render();
    this._updateUI();
  }

  _update(dt) {
    const m = this.metrics;
    const c = this.config;
    const t = this.time;

    const orbitDrift = 0.07 + c.autoOrbit * 0.05 + c.userOrbitImpulse;
    c.userOrbitImpulse *= 0.95;
    c.yaw += orbitDrift * dt;
    c.pitch = clamp(c.pitch + Math.sin(t * 0.17) * 0.0012 * c.autoOrbit, -1.15, 1.1);

    for (let i = 0; i < 3; i++) {
      this.outputs[i] = lerp(this.outputs[i], this.outputsTarget[i], 0.14);
    }
    this.manualPulseBoost = Math.max(0, this.manualPulseBoost - dt * 2.2);
    this.coils.toroidal = this.outputs[0];
    this.coils.poloidal = this.outputs[1];
    this.coils.pulse = this.outputs[2];

    const toroidalSpeed = 0.8 + c.heating * 1.5 + (this.coils.toroidal + 1) * 0.45;
    const poloidalSpeed = 0.2 + c.gain * 0.9 + (this.coils.poloidal + 1) * 0.28;
    const pulseKick = (this.coils.pulse + 1) * 0.5;
    const turbulenceField = c.turbulence * 0.65 + (1 - m.confinement) * 0.45;

    this.sensors.fill(0);
    let radialEnergy = 0;
    let hotCount = 0;

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      const thermalJitter = signedNoise() * (0.06 + c.heating * 0.09 + p.heat * 0.08);
      const pulseWave = Math.sin(t * (2.0 + pulseKick * 3.5) + p.u * 4.0 + p.phase) * 0.22 * pulseKick;
      const confinementDamp = 1.5 + c.gain * 1.7 + (this.coils.toroidal + 1) * 0.45;

      p.du = lerp(
        p.du,
        (toroidalSpeed * (0.75 + p.heat * 0.8) + thermalJitter * 0.6) * (p.charge > 0 ? 1 : -1),
        0.035
      );
      p.dv = lerp(
        p.dv,
        (poloidalSpeed * (0.35 + 0.25 * Math.sin(t + p.phase)) + pulseWave) * (p.charge > 0 ? 1 : -1),
        0.045
      );

      p.u = wrapTau(p.u + p.du * dt);
      p.v = wrapTau(p.v + p.dv * dt);

      const radialNoise = signedNoise() * (0.015 + turbulenceField * 0.04);
      const radialPull = -p.radial * confinementDamp;
      const radialSpiral = Math.sin(p.u * 3.0 + p.v * 2.0 + t * 1.4) * 0.03 * (c.turbulence + 0.15);
      p.radial += (radialPull + radialNoise + radialSpiral + pulseWave * 0.08) * dt;
      p.radial = clamp(p.radial, -this.maxRadial, this.maxRadial);

      radialEnergy += (p.radial / this.maxRadial) ** 2;

      const bin = Math.floor((p.u / TAU) * this.sensorCount) % this.sensorCount;
      const radialBias = 1 - Math.abs(p.radial / this.maxRadial);
      const deposit = (0.3 + p.heat * 0.85) * (0.45 + radialBias * 0.65);
      this.sensors[bin] += deposit;
      if (deposit > 0.9) hotCount += 1;
    }

    const invCount = 1 / this.particles.length;
    radialEnergy *= invCount;
    for (let i = 0; i < this.sensorCount; i++) this.sensors[i] *= 1.75 * invCount;

    let avg = 0;
    for (let i = 0; i < this.sensorCount; i++) {
      const a = this.sensors[(i - 1 + this.sensorCount) % this.sensorCount];
      const b = this.sensors[i];
      const d = this.sensors[(i + 1) % this.sensorCount];
      this.sensorBlur[i] = a * 0.22 + b * 0.56 + d * 0.22;
      this.sensorDelta[i] = d - a;
      avg += this.sensorBlur[i];
    }
    avg /= this.sensorCount;

    let varDiff = 0;
    let deltaEnergy = 0;
    let peak = 0;
    for (let i = 0; i < this.sensorCount; i++) {
      const v = this.sensorBlur[i];
      const prev = this.sensorBlur[(i - 1 + this.sensorCount) % this.sensorCount];
      const diff = v - prev;
      varDiff += diff * diff;
      deltaEnergy += this.sensorDelta[i] * this.sensorDelta[i];
      peak = Math.max(peak, v);
    }
    varDiff /= this.sensorCount;
    deltaEnergy /= this.sensorCount;

    const densityRaw = clamp(avg * 1.65 + peak * 0.25, 0, 1);
    const instabilityRaw = clamp(
      varDiff * 6.0 + radialEnergy * 0.85 + c.turbulence * 0.45 - c.gain * 0.16,
      0,
      1
    );
    const coherenceRaw = clamp(
      1 - varDiff * 4.4 - deltaEnergy * 1.8 - c.turbulence * 0.22 + c.gain * 0.19,
      0,
      1
    );

    const controllerOutputs = this._runController(
      {
        density: densityRaw,
        instability: instabilityRaw,
        coherence: coherenceRaw,
        radialEnergy,
        peak,
        avg,
        hotCount: hotCount * invCount,
      },
      dt
    );

    for (let i = 0; i < 3; i++) this.outputsTarget[i] = clamp(controllerOutputs[i], -1, 1);
    this.outputsTarget[2] = clamp(this.outputsTarget[2] + this.manualPulseBoost * 0.95, -1, 1);

    const pulseBoost = smoothstep(-1, 1, this.outputs[2]);
    const heatingTarget = clamp(
      0.32 + c.heating * 0.95 + pulseBoost * 0.22 + densityRaw * 0.18 - instabilityRaw * 0.12,
      0,
      1
    );
    m.temperature = lerp(m.temperature, heatingTarget, 0.05);
    m.density = lerp(m.density, densityRaw, 0.08);
    m.instability = lerp(m.instability, instabilityRaw, 0.09);
    m.coherence = lerp(m.coherence, coherenceRaw, 0.07);

    const confinementTarget = clamp(
      0.24 +
        c.gain * 0.62 +
        m.coherence * 0.33 -
        m.instability * 0.62 +
        smoothstep(-1, 1, this.outputs[0]) * 0.14,
      0,
      1
    );
    m.confinement = lerp(m.confinement, confinementTarget, 0.06);

    m.fusionPotential = clamp(
      (m.temperature - 0.42) * 1.8 * m.density * (0.3 + m.confinement) * (1.02 - m.instability),
      0,
      1
    );

    this._spawnAndUpdateBursts(dt);
    const burstEnergy = this.bursts.reduce((sum, b) => sum + b.intensity * (1 - b.age / b.ttl), 0);
    const yieldTarget = clamp(
      m.fusionPotential * 0.55 + burstEnergy * 0.12 + (pulseBoost > 0.6 ? 0.06 : 0),
      0,
      1
    );
    m.yield = lerp(m.yield, yieldTarget, 0.055);

    this._updateMission(dt, {
      density: densityRaw,
      instability: instabilityRaw,
      coherence: coherenceRaw,
      radialEnergy,
      peak,
      avg,
    });

    this.audioTelemetry.beat = Math.max(
      this.audioTelemetry.beat * 0.9,
      m.yield * 0.35 + (1 - m.instability) * 0.2 + this.challenge.flash * 0.35
    );
    this.audioTelemetry.spikeImpulse = Math.max(0, this.audioTelemetry.spikeImpulse - dt * 2.8);
    this.audioTelemetry.burstImpulse = Math.max(0, this.audioTelemetry.burstImpulse - dt * 1.6);

    this.sensorTrailAge += dt * (1.7 + c.heating * 1.2);
    if (this.sensorTrailAge >= 1) {
      this.sensorTrailAge = 0;
      this.sensorTrail.pop();
      const snap = new Float32Array(this.sensorCount);
      for (let i = 0; i < this.sensorCount; i++) snap[i] = this.sensorBlur[i];
      this.sensorTrail.unshift(snap);
    }
  }

  _runController(plasma, dt) {
    if (this.config.mode === "cnn") return this._runCNN(plasma, dt);
    if (this.config.mode === "snn") return this._runSNN(plasma, dt);
    return this._runMLP(plasma, dt);
  }

  _runMLP(plasma, dt) {
    const c = this.config;
    const net = this.mlp;
    const inp = net.inputs;
    inp[0] = plasma.density * 2 - 1;
    inp[1] = this.metrics.temperature * 2 - 1;
    inp[2] = plasma.instability * 2 - 1;
    inp[3] = plasma.coherence * 2 - 1;
    inp[4] = c.heating * 2 - 1;

    for (let i = 0; i < net.hiddenCount; i++) {
      let sum = net.b1[i];
      for (let j = 0; j < net.inCount; j++) sum += net.w1[i][j] * inp[j];
      net.hidden[i] = tanh(sum);
    }

    for (let k = 0; k < net.outCount; k++) {
      let sum = net.b2[k];
      for (let i = 0; i < net.hiddenCount; i++) sum += net.w2[k][i] * net.hidden[i];
      net.outputs[k] = tanh(sum);
    }

    net.desired[0] = clamp(
      -0.15 + plasma.instability * 1.15 - plasma.coherence * 0.55 + (0.6 - c.gain) * 0.45,
      -1,
      1
    );
    net.desired[1] = clamp(
      0.08 + (plasma.peak - plasma.avg) * 2.2 + plasma.radialEnergy * 0.8 - plasma.coherence * 0.35,
      -1,
      1
    );
    net.desired[2] = clamp(
      -0.25 + (this.metrics.temperature - 0.65) * 1.5 + plasma.density * 0.75 + plasma.instability * 0.6,
      -1,
      1
    );

    const learn = c.adapt * dt * 0.75;
    if (learn > 0) {
      const hiddenErr = new Float32Array(net.hiddenCount);
      for (let k = 0; k < net.outCount; k++) {
        const err = net.desired[k] - net.outputs[k];
        for (let i = 0; i < net.hiddenCount; i++) {
          hiddenErr[i] += err * net.w2[k][i];
          net.w2[k][i] += learn * 0.45 * err * net.hidden[i];
          net.w2[k][i] = clamp(net.w2[k][i], -2.2, 2.2);
        }
        net.b2[k] = clamp(net.b2[k] + learn * 0.24 * err, -1.2, 1.2);
      }
      for (let i = 0; i < net.hiddenCount; i++) {
        const err = hiddenErr[i] * (1 - net.hidden[i] * net.hidden[i]);
        for (let j = 0; j < net.inCount; j++) {
          net.w1[i][j] += learn * 0.18 * err * inp[j];
          net.w1[i][j] = clamp(net.w1[i][j], -2.2, 2.2);
        }
        net.b1[i] = clamp(net.b1[i] + learn * 0.12 * err, -1.2, 1.2);
      }
    }

    return [net.outputs[0], net.outputs[1], net.outputs[2]];
  }

  _runCNN(plasma, dt) {
    const c = this.config;
    const cnn = this.cnn;
    const n = this.sensorCount;

    let hotIndex = 0;
    let hotVal = -Infinity;
    for (let i = 0; i < n; i++) {
      if (this.sensorBlur[i] > hotVal) {
        hotVal = this.sensorBlur[i];
        hotIndex = i;
      }
    }
    cnn.hotIndex = hotIndex;

    for (let ch = 0; ch < cnn.channels; ch++) {
      let sum = 0;
      let peak = -Infinity;
      const feat = cnn.features[ch];
      for (let i = 0; i < n; i++) {
        let v = cnn.bias[ch];
        for (let k = 0; k < cnn.kernelSize; k++) {
          const idx = (i + k - 2 + n) % n;
          v += cnn.kernels[ch][k] * (this.sensorBlur[idx] * 2 - 1);
        }
        const f = tanh(v);
        feat[i] = f;
        sum += f;
        peak = Math.max(peak, f);
      }
      cnn.pooled[ch] = sum / n;
      cnn.peaks[ch] = peak;
    }

    const localGradient = this.sensorDelta[hotIndex];
    const hotspotContrast = hotVal - plasma.avg;
    const outputs = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      let v = cnn.mixBias[k];
      for (let ch = 0; ch < cnn.channels; ch++) {
        v += cnn.mix[k][ch] * (cnn.pooled[ch] * 0.7 + cnn.peaks[ch] * 0.35);
      }
      if (k === 0) v += plasma.instability * 0.8 - plasma.coherence * 0.35;
      if (k === 1) v += localGradient * 1.65 + (hotIndex / n - 0.5) * 0.3;
      if (k === 2) v += hotspotContrast * 2.2 + (this.metrics.temperature - 0.6) * 0.55;
      outputs[k] = tanh(v);
    }

    const desired = [
      clamp(plasma.instability * 1.2 - plasma.coherence * 0.5 - 0.1, -1, 1),
      clamp(localGradient * 2.5 - hotspotContrast * 0.5, -1, 1),
      clamp(hotspotContrast * 1.8 + (this.metrics.temperature - 0.7) * 1.1 - 0.2, -1, 1),
    ];

    const learn = c.adapt * dt * 0.6;
    if (learn > 0) {
      for (let k = 0; k < 3; k++) {
        const err = desired[k] - outputs[k];
        for (let ch = 0; ch < cnn.channels; ch++) {
          cnn.mix[k][ch] += learn * 0.22 * err * (cnn.pooled[ch] + cnn.peaks[ch] * 0.35);
          cnn.mix[k][ch] = clamp(cnn.mix[k][ch], -2.5, 2.5);
        }
        cnn.mixBias[k] = clamp(cnn.mixBias[k] + learn * 0.1 * err, -1.5, 1.5);
      }

      const reward = (this.metrics.confinement - this.metrics.instability) * 0.5 + hotspotContrast * -0.2;
      for (let ch = 0; ch < cnn.channels; ch++) {
        for (let k = 0; k < cnn.kernelSize; k++) {
          const idx = (hotIndex + k - 2 + n) % n;
          const sample = this.sensorBlur[idx] * 2 - 1;
          const targetSign = -Math.sign(localGradient || 1) * (ch === 1 ? 1 : 0.4);
          cnn.kernels[ch][k] += learn * 0.08 * (reward + targetSign * sample);
          cnn.kernels[ch][k] = clamp(cnn.kernels[ch][k], -2.2, 2.2);
        }
        cnn.bias[ch] = clamp(cnn.bias[ch] + learn * 0.04 * reward, -1.5, 1.5);
      }
    }

    return outputs;
  }

  _runSNN(plasma, dt) {
    const c = this.config;
    const snn = this.snn;
    const count = snn.count;
    const spikes = snn.spikes;
    spikes.fill(0);

    let spikeSum = 0;
    let vecX = 0;
    let vecY = 0;
    const sensorScale = this.sensorCount / count;

    for (let i = 0; i < count; i++) {
      const n = snn.neurons[i];
      const sensorIdx = Math.floor(i * sensorScale) % this.sensorCount;
      const local = this.sensorBlur[sensorIdx];
      const localDelta = this.sensorDelta[sensorIdx];
      const leftSpike = snn.prevSpikes[(i - 1 + count) % count];
      const rightSpike = snn.prevSpikes[(i + 1) % count];

      n.refractory = Math.max(0, n.refractory - dt);
      n.spikeGlow = Math.max(0, n.spikeGlow - dt * 3.2);

      const drive =
        (0.35 + c.heating * 0.65) +
        local * (0.8 + c.gain * 0.9) +
        Math.abs(localDelta) * 0.35 +
        leftSpike * 0.55 +
        rightSpike * 0.22 +
        signedNoise() * (0.015 + c.turbulence * 0.04);

      n.v += ((-n.v * 1.75) + drive * 1.25) * dt * 3.0;

      if (n.refractory <= 0 && n.v > n.threshold) {
        n.v = -0.18;
        n.refractory = randRange(0.055, 0.12);
        n.spikeGlow = 1;
        spikes[i] = 1;
      }

      const targetThreshold = 0.96 + plasma.instability * 0.45 - plasma.coherence * 0.28 + c.turbulence * 0.12;
      n.threshold = lerp(n.threshold, targetThreshold, c.adapt * dt * 0.25);
      n.threshold = clamp(n.threshold, 0.65, 1.55);

      if (spikes[i] > 0) {
        spikeSum += 1;
        const a = (i / count) * TAU;
        vecX += Math.cos(a);
        vecY += Math.sin(a);
      }
    }

    snn.prevSpikes.set(spikes);
    const spikeRate = spikeSum / count;
    snn.spikeRate = lerp(snn.spikeRate, spikeRate, 0.28);
    if (spikeSum > 0) {
      this.audioTelemetry.spikeImpulse = Math.max(
        this.audioTelemetry.spikeImpulse,
        Math.min(1, spikeRate * 3 + spikeSum / count)
      );
    }

    const mag = Math.hypot(vecX, vecY) / Math.max(1, spikeSum);
    let directionality = 0;
    if (spikeSum > 0) {
      const ang = Math.atan2(vecY, vecX);
      directionality = Math.sin(ang + this.time * 0.7) * clamp(mag * 1.7, 0, 1);
    }
    snn.directionality = lerp(snn.directionality, directionality, 0.2);
    snn.pulseMemory = lerp(snn.pulseMemory, spikeRate > 0.14 ? 1 : spikeRate > 0.06 ? 0.45 : 0, 0.14);

    const o0 = tanh(snn.spikeRate * 3.1 + plasma.coherence * 0.8 - plasma.instability * 1.0 - 0.35);
    const o1 = tanh(snn.directionality * 1.7 + this.sensorDelta[Math.floor(this.sensorCount * 0.25)] * 0.5);
    const o2 = tanh((snn.pulseMemory - 0.35) * 2.2 + plasma.instability * 0.7 + (this.metrics.temperature - 0.65) * 0.8);

    return [o0, o1, o2];
  }

  _spawnAndUpdateBursts(dt) {
    const m = this.metrics;
    const pulseBias = smoothstep(-1, 1, this.outputs[2]);
    const chance = m.fusionPotential * (0.22 + pulseBias * 0.8) * dt * 8.5;
    let burstSpawnedEnergy = 0;

    if (Math.random() < chance) {
      const hotspot = this._findHotspotIndex();
      const u = ((hotspot + Math.random() * 1.5 - 0.75) / this.sensorCount) * TAU;
      const v = randRange(-0.5, 0.5) + Math.sin(this.time * 0.7) * 0.2;
      this.bursts.push({
        u,
        v,
        radial: randRange(-0.03, 0.06),
        intensity: clamp(0.45 + Math.random() * 0.8 + m.fusionPotential * 0.5, 0, 1.8),
        ttl: randRange(0.18, 0.48),
        age: 0,
      });
      burstSpawnedEnergy += this.bursts[this.bursts.length - 1].intensity;
    }

    for (let i = this.bursts.length - 1; i >= 0; i--) {
      const b = this.bursts[i];
      b.age += dt;
      b.u = wrapTau(b.u + (0.2 + this.coils.toroidal * 0.35) * dt);
      b.v = wrapTau(b.v + (0.8 + this.coils.poloidal * 0.6) * dt);
      b.radial *= 0.96;
      if (b.age >= b.ttl) this.bursts.splice(i, 1);
    }

    if (this.bursts.length > 80) this.bursts.splice(0, this.bursts.length - 80);
    if (burstSpawnedEnergy > 0) {
      this.audioTelemetry.burstImpulse = Math.max(
        this.audioTelemetry.burstImpulse,
        clamp(burstSpawnedEnergy / 1.6, 0, 1)
      );
      this.audioTelemetry.beat = Math.max(this.audioTelemetry.beat, 0.65);
    }
  }

  _findHotspotIndex() {
    let idx = 0;
    let best = -Infinity;
    for (let i = 0; i < this.sensorCount; i++) {
      if (this.sensorBlur[i] > best) {
        best = this.sensorBlur[i];
        idx = i;
      }
    }
    return idx;
  }

  _torusPoint(u, v, radialOffset = 0) {
    const rr = this.r + radialOffset;
    const c = this.R + rr * Math.cos(v);
    return { x: c * Math.cos(u), y: c * Math.sin(u), z: rr * Math.sin(v) };
  }

  _project(p) {
    let x = p.x;
    let y = p.z;
    let z = p.y;

    const yaw = this.config.yaw;
    const pitch = this.config.pitch;

    const cosy = Math.cos(yaw);
    const siny = Math.sin(yaw);
    const x1 = x * cosy - z * siny;
    const z1 = x * siny + z * cosy;

    const cosp = Math.cos(pitch);
    const sinp = Math.sin(pitch);
    const y2 = y * cosp - z1 * sinp;
    let z2 = y * sinp + z1 * cosp;

    z2 += 4.1;
    const focal = (this.viewport.height * 0.88) * this.config.zoom;
    const s = focal / z2;

    return {
      x: this.viewport.cx + x1 * s,
      y: this.viewport.cy + y2 * s,
      z: z2,
      s,
      visible: z2 > 0.1,
    };
  }

  _drawCurve(points, stroke, lineWidth, alpha = 1) {
    const ctx = this.ctx;
    ctx.beginPath();
    let first = true;
    for (let i = 0; i < points.length; i++) {
      const q = points[i];
      if (!q.visible) continue;
      if (first) {
        ctx.moveTo(q.x, q.y);
        first = false;
      } else {
        ctx.lineTo(q.x, q.y);
      }
    }
    if (first) return;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }

  _render() {
    const ctx = this.ctx;
    const { width, height } = this.viewport;
    ctx.clearRect(0, 0, width, height);

    this._renderBackdrop(ctx, width, height);
    this._renderTorusWire(ctx);
    this._renderFieldLines(ctx);
    this._renderSensorRibbon(ctx);
    this._renderParticles(ctx);
    this._renderBursts(ctx);
    this._renderNeuralLayer(ctx);
    this._renderOnCanvasHUD(ctx);
  }

  _renderBackdrop(ctx, width, height) {
    const t = this.time;
    const vignette = ctx.createRadialGradient(
      width * 0.52, height * 0.5, 30,
      width * 0.52, height * 0.57, Math.max(width, height) * 0.58
    );
    vignette.addColorStop(0, "rgba(116, 166, 255, 0.06)");
    vignette.addColorStop(0.45, "rgba(20, 28, 48, 0.08)");
    vignette.addColorStop(1, "rgba(0, 0, 0, 0.42)");
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    for (let i = 0; i < this.stars.length; i++) {
      const s = this.stars[i];
      const x = s.x * width + Math.sin(t * 0.06 + s.twinkle) * 8;
      const y = s.y * height + Math.cos(t * 0.05 + s.twinkle) * 6;
      const a = (0.12 + 0.22 * (0.5 + 0.5 * Math.sin(t * 0.8 + s.twinkle))) * s.amp;
      const r = 0.6 + s.z * 1.4;
      ctx.globalAlpha = a;
      ctx.fillStyle = "rgba(220, 236, 255, 0.95)";
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TAU);
      ctx.fill();
    }
    ctx.restore();
  }

  _renderTorusWire(ctx) {
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const majorLoops = 10;
    for (let i = 0; i < majorLoops; i++) {
      const v = (i / majorLoops) * TAU + this.time * 0.05;
      const points = [];
      for (let s = 0; s <= 120; s++) {
        const u = (s / 120) * TAU;
        points.push(this._project(this._torusPoint(u, v)));
      }
      const pulseShade = smoothstep(-1, 1, this.coils.pulse);
      const alpha = 0.07 + (i % 2 === 0 ? 0.03 : 0) + pulseShade * 0.03;
      this._drawCurve(points, `rgba(125, 136, 255, ${alpha})`, 1);
    }

    const minorLoops = 22;
    for (let i = 0; i < minorLoops; i++) {
      const u = (i / minorLoops) * TAU + this.time * (0.04 + this.coils.toroidal * 0.01);
      const points = [];
      for (let s = 0; s <= 48; s++) {
        const v = (s / 48) * TAU;
        points.push(this._project(this._torusPoint(u, v)));
      }
      this._drawCurve(points, "rgba(105, 243, 255, 0.08)", 1);
    }

    const core = [];
    for (let s = 0; s <= 180; s++) {
      const u = (s / 180) * TAU;
      core.push(this._project({ x: this.R * Math.cos(u), y: this.R * Math.sin(u), z: 0 }));
    }
    this._drawCurve(core, "rgba(255, 202, 102, 0.13)", 1.3);
    ctx.restore();
  }

  _renderFieldLines(ctx) {
    ctx.save();
    ctx.lineCap = "round";
    const lines = 7;
    const turns = lerp(1.4, 3.6, smoothstep(-1, 1, this.coils.toroidal));
    const twist = lerp(1.0, 4.2, smoothstep(-1, 1, this.coils.poloidal));
    const pulse = smoothstep(-1, 1, this.coils.pulse);
    const t = this.time;

    for (let l = 0; l < lines; l++) {
      const phase = (l / lines) * TAU;
      const points = [];
      for (let s = 0; s <= 180; s++) {
        const q = s / 180;
        const u = q * TAU * turns + phase + t * 0.35;
        const v = q * TAU * twist + phase * 0.65 + Math.sin(q * 9 - t * 1.8) * 0.12 * pulse;
        const radial = Math.sin(q * 13 + l + t) * 0.01 * (0.2 + pulse * 1.8);
        points.push(this._project(this._torusPoint(u, v, radial)));
      }
      const alpha = 0.12 + pulse * 0.12 + (l % 2 ? 0.04 : 0);
      const color = l < lines / 2
        ? `rgba(105, 243, 255, ${alpha})`
        : `rgba(21, 208, 163, ${alpha})`;
      this._drawCurve(points, color, 1.1 + pulse * 0.9, alpha);
    }
    ctx.restore();
  }

  _renderSensorRibbon(ctx) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const t = this.time;

    for (let depth = this.trailDepth - 1; depth >= 0; depth--) {
      const trail = this.sensorTrail[depth];
      const ageT = 1 - depth / this.trailDepth;
      const alphaBase = 0.014 + ageT * 0.035;
      for (let i = 0; i < this.sensorCount; i++) {
        const sensor = trail[i];
        if (sensor < 0.04) continue;
        const a0 = (i / this.sensorCount) * TAU + t * (0.03 + ageT * 0.03);
        const a1 = ((i + 0.9) / this.sensorCount) * TAU + t * (0.03 + ageT * 0.03);
        const radial = (sensor - 0.18) * 0.12 + Math.sin(t * 2 + i * 0.6 + depth) * 0.008;
        const v = Math.sin(i * 0.45 + depth * 0.25 + t * 0.4) * (0.14 + ageT * 0.08);
        const p0 = this._project(this._torusPoint(a0, v, radial));
        const p1 = this._project(this._torusPoint(a1, v, radial));
        if (!p0.visible || !p1.visible) continue;
        ctx.strokeStyle =
          sensor > 0.36
            ? `rgba(255, 202, 102, ${alphaBase + sensor * 0.12})`
            : `rgba(105, 243, 255, ${alphaBase + sensor * 0.1})`;
        ctx.lineWidth = (1.0 + sensor * 2.8) * p0.s * 0.014;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
    }

    if (this.config.mode === "cnn") {
      const hot = this.cnn.hotIndex;
      for (let band = 0; band < 3; band++) {
        for (let o = -3; o <= 3; o++) {
          const i = (hot + o + this.sensorCount) % this.sensorCount;
          const intensity = Math.max(0, 1 - Math.abs(o) / 3.6);
          const u = (i / this.sensorCount) * TAU;
          const p = this._project(this._torusPoint(u, band * 0.22 + t * 0.9, 0.04 * band));
          if (!p.visible) continue;
          ctx.globalAlpha = 0.1 + intensity * 0.18;
          ctx.fillStyle = band === 2 ? "rgba(255, 202, 102, 0.95)" : "rgba(125, 136, 255, 0.95)";
          ctx.beginPath();
          ctx.arc(p.x, p.y, (5 + band * 2 + intensity * 4) * p.s * 0.05, 0, TAU);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    }

    ctx.restore();
  }

  _renderParticles(ctx) {
    const projected = [];
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      const q = this._project(this._torusPoint(p.u, p.v, p.radial));
      if (!q.visible) continue;
      projected.push({ q, p });
    }
    projected.sort((a, b) => b.q.z - a.q.z);

    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < projected.length; i++) {
      const { q, p } = projected[i];
      const radialBias = 1 - Math.abs(p.radial / this.maxRadial);
      const alpha = 0.14 + radialBias * 0.18;
      const perspectiveScale = Math.min(1.45, Math.max(0.35, q.s * 0.012));
      const r = (1.2 + p.heat * 1.3 + (1 - q.z / 8) * 0.75) * perspectiveScale;
      const isHot = p.heat > 0.76;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = isHot ? "rgba(255, 164, 100, 0.92)" : "rgba(100, 240, 255, 0.9)";
      ctx.beginPath();
      ctx.arc(q.x, q.y, r, 0, TAU);
      ctx.fill();
    }
    ctx.restore();
  }

  _renderBursts(ctx) {
    if (!this.bursts.length) return;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < this.bursts.length; i++) {
      const b = this.bursts[i];
      const life = 1 - b.age / b.ttl;
      const p = this._project(this._torusPoint(b.u, b.v, b.radial));
      if (!p.visible) continue;
      const perspectiveScale = Math.min(2.4, Math.max(0.45, p.s * 0.01));
      const radius = (8 + b.intensity * 10) * perspectiveScale * (0.35 + life * 0.7);
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius);
      g.addColorStop(0, `rgba(255, 244, 195, ${0.32 * life})`);
      g.addColorStop(0.35, `rgba(255, 202, 102, ${0.24 * life})`);
      g.addColorStop(1, "rgba(255, 202, 102, 0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, TAU);
      ctx.fill();
    }
    ctx.restore();
  }

  _renderNeuralLayer(ctx) {
    if (this.config.mode === "cnn") return this._renderCNNOverlay(ctx);
    if (this.config.mode === "snn") return this._renderSNNOverlay(ctx);
    return this._renderMLPOverlay(ctx);
  }

  _renderMLPOverlay(ctx) {
    const net = this.mlp;
    const t = this.time;
    const inputNodes = [];
    const hiddenNodes = [];
    const outputNodes = [];

    for (let i = 0; i < net.inCount; i++) {
      const u = 0.15 * TAU + (i / net.inCount) * 0.24 * TAU + t * 0.09;
      const v = -0.6 + (i - (net.inCount - 1) / 2) * 0.22;
      inputNodes.push({ pos: this._project(this._torusPoint(u, v, 0.18)), act: net.inputs[i] });
    }
    for (let i = 0; i < net.hiddenCount; i++) {
      const u = 0.38 * TAU + (i / net.hiddenCount) * 0.3 * TAU - t * 0.05;
      const v = -0.9 + ((i % 4) - 1.5) * 0.5 + Math.floor(i / 4) * 0.22;
      hiddenNodes.push({
        pos: this._project(this._torusPoint(u, v, 0.22 + Math.sin(t + i) * 0.02)),
        act: net.hidden[i],
      });
    }
    for (let i = 0; i < net.outCount; i++) {
      const u = 0.72 * TAU + (i / net.outCount) * 0.1 * TAU + t * 0.12;
      const v = -0.45 + i * 0.45;
      outputNodes.push({ pos: this._project(this._torusPoint(u, v, 0.24)), act: net.outputs[i] });
    }

    ctx.save();
    ctx.globalCompositeOperation = "lighter";

    const connect = (aNodes, bNodes, weightGetter) => {
      for (let a = 0; a < aNodes.length; a++) {
        for (let b = 0; b < bNodes.length; b++) {
          const p0 = aNodes[a].pos;
          const p1 = bNodes[b].pos;
          if (!p0.visible || !p1.visible) continue;
          const w = weightGetter(a, b);
          const mag = Math.min(1, Math.abs(w) / 1.6);
          const alpha = 0.03 + mag * 0.12;
          ctx.strokeStyle = w >= 0 ? `rgba(105, 243, 255, ${alpha})` : `rgba(255, 95, 122, ${alpha})`;
          ctx.lineWidth = 0.7 + mag * 1.2;
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();
        }
      }
    };

    connect(inputNodes, hiddenNodes, (a, b) => net.w1[b][a]);
    connect(hiddenNodes, outputNodes, (a, b) => net.w2[b][a]);

    const drawNodes = (nodes, kind) => {
      for (let i = 0; i < nodes.length; i++) {
        const { pos, act } = nodes[i];
        if (!pos.visible) continue;
        const glow = 0.18 + Math.abs(act) * 0.28;
        const base = kind === "output" ? 10 : kind === "hidden" ? 7 : 6;
        const perspectiveScale = Math.min(1.45, Math.max(0.35, pos.s * 0.012));
        const r = base * 0.28 * perspectiveScale;
        ctx.globalAlpha = glow;
        ctx.fillStyle = act >= 0 ? "rgba(105, 243, 255, 0.95)" : "rgba(255, 95, 122, 0.95)";
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, TAU);
        ctx.fill();
      }
    };

    drawNodes(inputNodes, "input");
    drawNodes(hiddenNodes, "hidden");
    drawNodes(outputNodes, "output");
    ctx.restore();
  }

  _renderCNNOverlay(ctx) {
    const cnn = this.cnn;
    const t = this.time;
    const n = this.sensorCount;

    ctx.save();
    ctx.globalCompositeOperation = "lighter";

    for (let ch = 0; ch < cnn.channels; ch++) {
      const feat = cnn.features[ch];
      for (let i = 0; i < n; i++) {
        const f = feat[i];
        const mag = Math.abs(f);
        if (mag < 0.08) continue;
        const u = (i / n) * TAU + t * 0.04;
        const v = -0.55 + ch * 0.55;
        const radial = 0.16 + ch * 0.05 + f * 0.05;
        const p = this._project(this._torusPoint(u, v, radial));
        if (!p.visible) continue;
        const perspectiveScale = Math.min(1.35, Math.max(0.35, p.s * 0.011));
        const r = (2.2 + mag * 5) * perspectiveScale;
        ctx.globalAlpha = 0.06 + mag * 0.16;
        ctx.fillStyle =
          ch === 0 ? "rgba(105, 243, 255, 0.95)" :
          ch === 1 ? "rgba(125, 136, 255, 0.95)" :
          "rgba(255, 202, 102, 0.95)";
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, TAU);
        ctx.fill();
      }
    }

    const hot = cnn.hotIndex;
    for (let i = -2; i <= 2; i++) {
      const idx = (hot + i + n) % n;
      const u = (idx / n) * TAU;
      const p0 = this._project(this._torusPoint(u, -0.95, 0.22));
      const p1 = this._project(this._torusPoint(u, 0.95, 0.22));
      if (!p0.visible || !p1.visible) continue;
      const strength = 1 - Math.abs(i) / 3;
      ctx.strokeStyle = `rgba(255, 202, 102, ${0.08 + strength * 0.18})`;
      ctx.lineWidth = 1 + strength * 2;
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
    }

    ctx.restore();
  }

  _renderSNNOverlay(ctx) {
    const snn = this.snn;
    const t = this.time;
    const count = snn.count;
    const nodes = [];
    for (let i = 0; i < count; i++) {
      const a = (i / count) * TAU;
      const ripple = Math.sin(a * 3 + t * 1.2) * 0.03;
      const p = this._project(this._torusPoint(a + t * 0.07, ripple, 0.24 + ripple * 0.3));
      nodes.push({ p, n: snn.neurons[i], spike: snn.prevSpikes[i] });
    }

    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < count; i++) {
      const a = nodes[i];
      const b = nodes[(i + 1) % count];
      if (!a.p.visible || !b.p.visible) continue;
      const pulse = Math.max(a.n.spikeGlow, b.n.spikeGlow);
      ctx.strokeStyle = `rgba(105, 243, 255, ${0.035 + pulse * 0.15})`;
      ctx.lineWidth = 0.8 + pulse * 1.8;
      ctx.beginPath();
      ctx.moveTo(a.p.x, a.p.y);
      ctx.lineTo(b.p.x, b.p.y);
      ctx.stroke();
    }
    for (let i = 0; i < count; i++) {
      const item = nodes[i];
      if (!item.p.visible) continue;
      const activation = clamp((item.n.v + 0.2) / 1.4, 0, 1);
      const spikeGlow = item.n.spikeGlow;
      const perspectiveScale = Math.min(1.35, Math.max(0.35, item.p.s * 0.011));
      const r = (2.4 + activation * 3.6 + spikeGlow * 5.5) * perspectiveScale;
      ctx.globalAlpha = 0.08 + activation * 0.15 + spikeGlow * 0.28;
      ctx.fillStyle = spikeGlow > 0.01 ? "rgba(255, 202, 102, 0.96)" : "rgba(125, 136, 255, 0.95)";
      ctx.beginPath();
      ctx.arc(item.p.x, item.p.y, r, 0, TAU);
      ctx.fill();
    }

    const mag = Math.abs(snn.directionality);
    if (mag > 0.01) {
      const angle = (snn.directionality > 0 ? 1 : -1) * 0.6 + t * 0.25;
      const p0 = this._project({ x: 0, y: 0, z: 0 });
      const p1 = this._project({
        x: Math.cos(angle) * this.R * 1.35,
        y: Math.sin(angle) * this.R * 1.35,
        z: 0.25 * Math.sin(t * 0.9),
      });
      if (p0.visible && p1.visible) {
        ctx.strokeStyle = `rgba(255, 202, 102, ${0.1 + mag * 0.25})`;
        ctx.lineWidth = 1.2 + mag * 2.5;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  _renderOnCanvasHUD(ctx) {
    const { width, height } = this.viewport;
    const m = this.metrics;
    ctx.save();
    const boxW = Math.min(280 * this.viewport.dpr, width * 0.32);
    const boxH = 94 * this.viewport.dpr;
    const x = width - boxW - 16 * this.viewport.dpr;
    const y = height - boxH - 14 * this.viewport.dpr;
    const r = 12 * this.viewport.dpr;

    ctx.fillStyle = "rgba(4, 8, 14, 0.42)";
    ctx.strokeStyle = "rgba(173, 210, 255, 0.12)";
    ctx.lineWidth = 1;
    this._roundRect(ctx, x, y, boxW, boxH, r);
    ctx.fill();
    ctx.stroke();

    ctx.font = `${12 * this.viewport.dpr}px "IBM Plex Mono", monospace`;
    ctx.fillStyle = "rgba(240, 248, 255, 0.85)";
    ctx.fillText(`mode=${this.config.mode.toUpperCase()}`, x + 12 * this.viewport.dpr, y + 18 * this.viewport.dpr);

    const rows = [
      ["Toroidal", this.coils.toroidal, "rgba(105, 243, 255, 0.95)"],
      ["Poloidal", this.coils.poloidal, "rgba(125, 136, 255, 0.95)"],
      ["Pulse", this.coils.pulse, "rgba(255, 202, 102, 0.95)"],
    ];
    rows.forEach((row, idx) => {
      const ry = y + (30 + idx * 20) * this.viewport.dpr;
      const barX = x + 80 * this.viewport.dpr;
      const barW = boxW - 92 * this.viewport.dpr;
      const pct = (row[1] + 1) / 2;
      ctx.fillStyle = "rgba(154, 167, 189, 0.8)";
      ctx.fillText(row[0], x + 12 * this.viewport.dpr, ry);
      ctx.fillStyle = "rgba(255, 255, 255, 0.06)";
      this._roundRect(ctx, barX, ry - 9 * this.viewport.dpr, barW, 8 * this.viewport.dpr, 999);
      ctx.fill();
      ctx.fillStyle = row[2];
      this._roundRect(ctx, barX, ry - 9 * this.viewport.dpr, barW * pct, 8 * this.viewport.dpr, 999);
      ctx.fill();
    });

    ctx.font = `${11 * this.viewport.dpr}px "IBM Plex Mono", monospace`;
    ctx.fillStyle = "rgba(154, 167, 189, 0.9)";
    const regime =
      m.confinement > 0.7 && m.instability < 0.35 ? "stable burn" :
      m.yield > 0.5 ? "burst train" :
      m.instability > 0.62 ? "runaway edge" :
      "adaptive confinement";
    const text = `regime=${regime}`;
    ctx.fillText(text, width - 16 * this.viewport.dpr - ctx.measureText(text).width, 20 * this.viewport.dpr);
    ctx.restore();
  }

  _roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  _updateUI() {
    const m = this.metrics;
    const fmtPct = (v) => `${Math.round(v * 100)}%`;
    const modeInfo = MODE_INFO[this.config.mode];
    if (modeInfo) {
      if (this.ui.modeTitle) this.ui.modeTitle.textContent = modeInfo.title;
      if (this.ui.modeDescription) this.ui.modeDescription.textContent = modeInfo.description;
      if (Array.isArray(this.ui.modeTabs)) {
        this.ui.modeTabs.forEach((btn) => {
          const active = btn.dataset.mode === this.config.mode;
          btn.classList.toggle("is-active", active);
          btn.setAttribute("aria-selected", String(active));
        });
      }
    }
    const tempMK = Math.round(65 + m.temperature * 190);
    const densityK = (m.density * 10.5).toFixed(2);
    const yieldMW = Math.round(m.yield * 1650);

    this.ui.metricTemp.textContent = `${tempMK} MK`;
    this.ui.metricDensity.textContent = `${densityK} a.u.`;
    this.ui.metricConfinement.textContent = fmtPct(m.confinement);
    this.ui.metricInstability.textContent = fmtPct(m.instability);
    this.ui.metricYield.textContent = `${yieldMW} MW`;
    this.ui.metricCoherence.textContent = fmtPct(m.coherence);

    this._setBar(this.ui.barTemp, m.temperature, "temp");
    this._setBar(this.ui.barDensity, m.density, "density");
    this._setBar(this.ui.barConfinement, m.confinement, "confinement");
    this._setBar(this.ui.barInstability, m.instability, "instability");
    this._setBar(this.ui.barYield, m.yield, "yield");
    this._setBar(this.ui.barCoherence, m.coherence, "coherence");

    const mission = this.challenge;
    const phase = this._getCurrentPhase();
    if (this.ui.missionState) this.ui.missionState.textContent = mission.state.toUpperCase();
    if (this.ui.missionPhase) {
      const current = Math.min(mission.phases.length, mission.phaseIndex + (mission.state === "won" ? 0 : 1));
      this.ui.missionPhase.textContent = `${current} / ${mission.phases.length}`;
    }
    if (this.ui.missionTimer) this.ui.missionTimer.textContent = fmtSeconds(mission.timeLeft);
    if (this.ui.missionScore) this.ui.missionScore.textContent = Math.round(mission.score).toLocaleString();
    if (this.ui.missionIntegrity) this.ui.missionIntegrity.textContent = fmtPct(mission.integrity);
    if (this.ui.missionProgress) {
      const phasePct = phase ? clamp(mission.phaseProgress / phase.progressNeeded, 0, 1) : 0;
      this.ui.missionProgress.textContent = fmtPct(phasePct);
    }
    if (this.ui.missionObjective) this.ui.missionObjective.textContent = mission.objectiveText;
    if (this.ui.missionHint) this.ui.missionHint.textContent = mission.hintText;
    this._setBar(this.ui.barIntegrity, mission.integrity, mission.integrity < 0.3 ? "instability" : "confinement");
    if (phase) this._setBar(this.ui.barMissionProgress, mission.phaseProgress / phase.progressNeeded, "yield");
    else this._setBar(this.ui.barMissionProgress, 0, "yield");

    if (this.ui.challengeStateBadge) {
      this.ui.challengeStateBadge.textContent =
        mission.state === "running" ? "Live Mission" :
        mission.state === "won" ? "Mission Won" :
        mission.state === "lost" ? "Containment Lost" :
        "Sandbox";
      this.ui.challengeStateBadge.className = `badge ${
        mission.statusTone === "danger" ? "badge-danger" :
        mission.statusTone === "warn" ? "badge-warn" :
        "badge-live"
      }`;
    }
    if (this.ui.challengeStatusInline) this.ui.challengeStatusInline.textContent = mission.statusText;

    if (this.ui.challengeBtn) {
      this.ui.challengeBtn.textContent =
        mission.state === "running" ? "Restart Challenge" : "Start Challenge";
    }

    if (this.ui.missionReportOverlay) {
      this.ui.missionReportOverlay.hidden = !mission.reportVisible;
    }
    if (mission.summary) {
      if (this.ui.reportTitle) {
        this.ui.reportTitle.textContent =
          mission.summary.outcome === "won" ? "Controlled Burn Achieved" : "Containment Failure";
      }
      if (this.ui.reportSubtitle) {
        this.ui.reportSubtitle.textContent =
          mission.summary.outcome === "won"
            ? "Mission success summary and scoring breakdown"
            : "Failure diagnostics and performance breakdown";
      }
      if (this.ui.reportScore) this.ui.reportScore.textContent = mission.summary.totalScore.toLocaleString();
      if (this.ui.reportTimeLeft) this.ui.reportTimeLeft.textContent = fmtSeconds(mission.summary.timeLeft);
      if (this.ui.reportIntegrity) this.ui.reportIntegrity.textContent = `${mission.summary.integrityPct}%`;
      if (this.ui.reportSwitches) this.ui.reportSwitches.textContent = String(mission.summary.modeSwitches);
      if (this.ui.reportModeUsage) {
        const u = mission.summary.modeUsage;
        this.ui.reportModeUsage.textContent = `MLP ${u.mlp}s | CNN ${u.cnn}s | SNN ${u.snn}s`;
      }
      if (this.ui.reportBreakdown) {
        const b = mission.summary.scoreBreakdown;
        this.ui.reportBreakdown.textContent =
          `Base ${b.base} | Bonus ${b.modeBonus + b.pulseBonus + b.phaseClear} | Penalty ${Math.abs(b.modePenalty)}`;
      }
      if (this.ui.reportReason) {
        this.ui.reportReason.textContent =
          `${mission.summary.reason} Pulses: ${mission.summary.manualPulseCount}. ` +
          `Phase clear times: ${mission.summary.phaseClearTimes.length ? mission.summary.phaseClearTimes.join("s, ") + "s" : "none"}.`;
      }
    }

    if (this.ui.audioStatus) {
      const engine = this.audioEngine;
      this.ui.audioStatus.textContent = engine
        ? engine.getStatusText()
        : "Audio engine unavailable";
    }
    if (this.ui.audioBtn && this.audioEngine) {
      this.ui.audioBtn.textContent = this.audioEngine.enabled ? "Audio: On" : "Audio: Off";
      this.ui.audioBtn.setAttribute("aria-pressed", String(this.audioEngine.enabled));
    }
    if (this.ui.audioToggleBtn && this.audioEngine) {
      this.ui.audioToggleBtn.textContent = this.audioEngine.enabled ? "Disable Audio" : "Enable Audio";
      this.ui.audioToggleBtn.setAttribute("aria-pressed", String(this.audioEngine.enabled));
    }
    if (this.ui.audioBeat) {
      this.ui.audioBeat.classList.toggle("is-hot", this.audioTelemetry.beat > 0.45 || this.audioTelemetry.burstImpulse > 0.2);
    }
  }

  _setBar(el, value, kind) {
    if (!el) return;
    el.style.width = `${Math.round(clamp(value, 0, 1) * 100)}%`;
    switch (kind) {
      case "instability":
        el.style.background = "linear-gradient(90deg, rgba(255,95,122,0.9), rgba(255,202,102,0.95))";
        break;
      case "yield":
        el.style.background = "linear-gradient(90deg, rgba(255,202,102,0.9), rgba(255,244,195,0.95))";
        break;
      case "confinement":
        el.style.background = "linear-gradient(90deg, rgba(21,208,163,0.9), rgba(105,243,255,0.95))";
        break;
      case "coherence":
        el.style.background = "linear-gradient(90deg, rgba(125,136,255,0.9), rgba(105,243,255,0.95))";
        break;
      default:
        el.style.background = "linear-gradient(90deg, rgba(21,208,163,0.9), rgba(105,243,255,0.95))";
        break;
    }
  }
}

class ReactorAudioEngine {
  constructor(sim) {
    this.sim = sim;
    this.ctx = null;
    this.enabled = false;
    this.supported = typeof window !== "undefined" && ("AudioContext" in window || "webkitAudioContext" in window);
    this.nodes = null;
    this.burstEnv = 0;
    this.spikeEnv = 0;
    this.beatEnv = 0;
  }

  _createNoiseBuffer(ctx) {
    const buffer = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * 0.35;
    return buffer;
  }

  _ensure() {
    if (!this.supported || this.ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC();

    const master = this.ctx.createGain();
    master.gain.value = 0;
    master.connect(this.ctx.destination);

    const bedOsc = this.ctx.createOscillator();
    bedOsc.type = "sawtooth";
    const bedFilter = this.ctx.createBiquadFilter();
    bedFilter.type = "lowpass";
    bedFilter.frequency.value = 220;
    const bedGain = this.ctx.createGain();
    bedGain.gain.value = 0;
    bedOsc.connect(bedFilter);
    bedFilter.connect(bedGain);
    bedGain.connect(master);

    const pulseOsc = this.ctx.createOscillator();
    pulseOsc.type = "triangle";
    const pulseGain = this.ctx.createGain();
    pulseGain.gain.value = 0;
    pulseOsc.connect(pulseGain);
    pulseGain.connect(master);

    const spikeOsc = this.ctx.createOscillator();
    spikeOsc.type = "square";
    const spikeFilter = this.ctx.createBiquadFilter();
    spikeFilter.type = "bandpass";
    spikeFilter.frequency.value = 1200;
    const spikeGain = this.ctx.createGain();
    spikeGain.gain.value = 0;
    spikeOsc.connect(spikeFilter);
    spikeFilter.connect(spikeGain);
    spikeGain.connect(master);

    const noiseSrc = this.ctx.createBufferSource();
    noiseSrc.buffer = this._createNoiseBuffer(this.ctx);
    noiseSrc.loop = true;
    const noiseFilter = this.ctx.createBiquadFilter();
    noiseFilter.type = "bandpass";
    noiseFilter.frequency.value = 620;
    const noiseGain = this.ctx.createGain();
    noiseGain.gain.value = 0;
    noiseSrc.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(master);

    const lfo = this.ctx.createOscillator();
    lfo.type = "sine";
    lfo.frequency.value = 0.7;
    const lfoGain = this.ctx.createGain();
    lfoGain.gain.value = 18;
    lfo.connect(lfoGain);
    lfoGain.connect(bedFilter.frequency);

    const now = this.ctx.currentTime;
    bedOsc.start(now);
    pulseOsc.start(now);
    spikeOsc.start(now);
    noiseSrc.start(now);
    lfo.start(now);

    this.nodes = {
      master,
      bedOsc,
      bedFilter,
      bedGain,
      pulseOsc,
      pulseGain,
      spikeOsc,
      spikeFilter,
      spikeGain,
      noiseSrc,
      noiseFilter,
      noiseGain,
      lfo,
      lfoGain,
    };
  }

  async enable() {
    if (!this.supported) return false;
    this._ensure();
    if (!this.ctx) return false;
    await this.ctx.resume();
    this.enabled = true;
    return true;
  }

  async disable() {
    if (!this.ctx) {
      this.enabled = false;
      return false;
    }
    this.enabled = false;
    try {
      await this.ctx.suspend();
    } catch {
      // ignore
    }
    return false;
  }

  async toggle() {
    return this.enabled ? this.disable() : this.enable();
  }

  setVolume(v) {
    this.sim.config.masterVolume = clamp(v, 0, 1);
  }

  getStatusText() {
    if (!this.supported) return "Web Audio unsupported";
    if (!this.ctx) return "Audio engine idle";
    if (!this.enabled) return "Audio suspended";
    return `Running (${this.ctx.state})`;
  }

  update(dt) {
    if (!this.nodes || !this.ctx) return;
    const { master, bedOsc, bedFilter, bedGain, pulseOsc, pulseGain, spikeOsc, spikeFilter, spikeGain, noiseFilter, noiseGain } = this.nodes;
    const s = this.sim;
    const m = s.metrics;
    const mode = s.config.mode;
    const now = this.ctx.currentTime;
    const ramp = 0.04;

    this.burstEnv = Math.max(this.burstEnv * (1 - dt * 2.4), s.audioTelemetry.burstImpulse);
    this.spikeEnv = Math.max(this.spikeEnv * (1 - dt * 4.6), s.audioTelemetry.spikeImpulse);
    this.beatEnv = Math.max(this.beatEnv * (1 - dt * 2.2), s.audioTelemetry.beat);

    const modeOffset = mode === "cnn" ? 22 : mode === "snn" ? 38 : 0;
    const bedFreq = 52 + s.config.heating * 110 + (s.coils.toroidal + 1) * 18 + modeOffset;
    const pulseFreq = 96 + (s.coils.pulse + 1) * 120 + m.yield * 180;
    const spikeFreq = 420 + this.spikeEnv * 1200 + (mode === "snn" ? 220 : 0);

    const masterTarget = this.enabled && this.ctx.state === "running" ? s.config.masterVolume * 0.25 : 0;
    master.gain.setTargetAtTime(masterTarget, now, ramp);
    bedOsc.frequency.setTargetAtTime(bedFreq, now, ramp);
    bedFilter.frequency.setTargetAtTime(160 + m.coherence * 500 + (1 - m.instability) * 180, now, ramp);
    bedGain.gain.setTargetAtTime(0.02 + m.confinement * 0.045 + this.beatEnv * 0.03, now, ramp);

    pulseOsc.frequency.setTargetAtTime(pulseFreq, now, 0.03);
    pulseGain.gain.setTargetAtTime(0.004 + this.beatEnv * 0.06 + this.burstEnv * 0.03, now, 0.03);

    spikeOsc.frequency.setTargetAtTime(spikeFreq, now, 0.02);
    spikeFilter.frequency.setTargetAtTime(800 + this.spikeEnv * 1800, now, 0.03);
    spikeGain.gain.setTargetAtTime(0.0 + this.spikeEnv * 0.055, now, 0.02);

    noiseFilter.frequency.setTargetAtTime(300 + m.instability * 1900 + this.burstEnv * 1200, now, 0.03);
    noiseGain.gain.setTargetAtTime(0.002 + this.burstEnv * 0.07 + m.yield * 0.012, now, 0.03);
  }
}

function toggleFullscreen(canvas) {
  if (!document.fullscreenElement) {
    canvas.requestFullscreen?.().catch(() => {});
  } else {
    document.exitFullscreen?.().catch(() => {});
  }
}

function attachKeyboardControls(sim) {
  const stepControl = (name, delta) => {
    sim.setControl(name, clamp(sim.config[name] + delta, 0, 1));
    const input = sim.ui.controls?.[name];
    if (input) input.value = String(Math.round(sim.config[name] * 100));
    const output = sim.ui.controlOutputs?.[name];
    if (output) output.textContent = `${Math.round(sim.config[name] * 100)}%`;
  };

  window.addEventListener("keydown", (e) => {
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;

    if (e.key === "Enter") {
      e.preventDefault();
      sim.restartChallenge();
      return;
    }
    if (e.key === "Escape" && sim.challenge.reportVisible) {
      e.preventDefault();
      sim.setReportVisible(false);
      return;
    }
    if (e.code === "Space") {
      e.preventDefault();
      sim.triggerManualPulse(1);
      return;
    }
    if (e.key === "1") return sim.setMode("mlp");
    if (e.key === "2") return sim.setMode("cnn");
    if (e.key === "3") return sim.setMode("snn");
    if (e.key === "ArrowLeft") { e.preventDefault(); stepControl("heating", -0.03); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); stepControl("heating", 0.03); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); stepControl("gain", -0.03); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); stepControl("gain", 0.03); return; }
    if (e.key.toLowerCase() === "a") { e.preventDefault(); stepControl("adapt", 0.03); return; }
    if (e.key.toLowerCase() === "b") { e.preventDefault(); stepControl("turbulence", 0.03); return; }
    if (e.key.toLowerCase() === "m") {
      e.preventDefault();
      sim.audioEngine?.toggle().then(() => sim.renderFrame());
      return;
    }
    if (e.key.toLowerCase() === "f") {
      e.preventDefault();
      toggleFullscreen(sim.canvas);
      return;
    }
    if (e.key.toLowerCase() === "r") {
      e.preventDefault();
      sim.reseed();
    }
    if (e.key.toLowerCase() === "o") {
      e.preventDefault();
      sim.setReportVisible(!sim.challenge.reportVisible);
      return;
    }
    if (e.key.toLowerCase() === "p") {
      e.preventDefault();
      sim.setPaused(!sim.config.paused);
    }
  });
}

function installAutomationHooks(sim) {
  window.render_game_to_text = () => sim.buildTextState();
  window.advanceTime = async (ms) => {
    const frames = Math.max(1, Math.round(ms / (1000 / 60)));
    for (let i = 0; i < frames; i++) sim.step(1 / 60);
    sim.renderFrame();
  };
  window.get_bridge_profile = () => sim.getBridgeProfile();
  window.apply_bridge_profile = (profileOrText) => {
    if (typeof profileOrText === "string") return sim.importBridgeProfileText(profileOrText);
    return sim.applyBridgeProfile(profileOrText);
  };
}

function setupUI(sim) {
  const qs = (sel) => document.querySelector(sel);
  const qsa = (sel) => Array.from(document.querySelectorAll(sel));

  const ui = {
    canvas: qs("#reactorCanvas"),
    challengeBtn: qs("#challengeBtn"),
    audioBtn: qs("#audioBtn"),
    pauseBtn: qs("#pauseBtn"),
    resetBtn: qs("#resetBtn"),
    startMissionBtn: qs("#startMissionBtn"),
    pulseBtn: qs("#pulseBtn"),
    challengeStateBadge: qs("#challengeStateBadge"),
    challengeStatusInline: qs("#challengeStatusInline"),
    missionState: qs("#missionState"),
    missionPhase: qs("#missionPhase"),
    missionTimer: qs("#missionTimer"),
    missionScore: qs("#missionScore"),
    missionIntegrity: qs("#missionIntegrity"),
    missionProgress: qs("#missionProgress"),
    missionObjective: qs("#missionObjective"),
    missionHint: qs("#missionHint"),
    barIntegrity: qs("#barIntegrity"),
    barMissionProgress: qs("#barMissionProgress"),
    modeTitle: qs("#modeTitle"),
    modeDescription: qs("#modeDescription"),
    modeTabs: qsa(".mode-tab"),
    controls: {
      heating: qs("#heating"),
      gain: qs("#gain"),
      adapt: qs("#adapt"),
      turbulence: qs("#turbulence"),
    },
    volume: qs("#volume"),
    volumeOut: qs("#volumeOut"),
    audioToggleBtn: qs("#audioToggleBtn"),
    audioStatus: qs("#audioStatus"),
    audioBeat: qs("#audioBeat"),
    missionReportOverlay: qs("#missionReportOverlay"),
    reportTitle: qs("#missionReportTitle"),
    reportSubtitle: qs("#missionReportSubtitle"),
    reportScore: qs("#reportScore"),
    reportTimeLeft: qs("#reportTimeLeft"),
    reportIntegrity: qs("#reportIntegrity"),
    reportSwitches: qs("#reportSwitches"),
    reportModeUsage: qs("#reportModeUsage"),
    reportBreakdown: qs("#reportBreakdown"),
    reportReason: qs("#reportReason"),
    reportRestartBtn: qs("#reportRestartBtn"),
    reportCloseBtn: qs("#reportCloseBtn"),
    controlOutputs: {
      heating: qs("#heatingOut"),
      gain: qs("#gainOut"),
      adapt: qs("#adaptOut"),
      turbulence: qs("#turbulenceOut"),
    },
    metricTemp: qs("#metricTemp"),
    metricDensity: qs("#metricDensity"),
    metricConfinement: qs("#metricConfinement"),
    metricInstability: qs("#metricInstability"),
    metricYield: qs("#metricYield"),
    metricCoherence: qs("#metricCoherence"),
    barTemp: qs("#barTemp"),
    barDensity: qs("#barDensity"),
    barConfinement: qs("#barConfinement"),
    barInstability: qs("#barInstability"),
    barYield: qs("#barYield"),
    barCoherence: qs("#barCoherence"),
    exportJsonBtn: qs("#exportJsonBtn"),
    copyCppBtn: qs("#copyCppBtn"),
    loadBridgeFileBtn: qs("#loadBridgeFileBtn"),
    bridgeFileInput: qs("#bridgeFileInput"),
    bridgeImportInput: qs("#bridgeImportInput"),
    applyBridgeBtn: qs("#applyBridgeBtn"),
    saveLocalBridgeBtn: qs("#saveLocalBridgeBtn"),
    loadLocalBridgeBtn: qs("#loadLocalBridgeBtn"),
    bridgeStatus: qs("#bridgeStatus"),
    bridgePreview: qs("#bridgePreview"),
  };

  sim.ui = ui;

  const syncControlText = (name, value) => {
    ui.controlOutputs[name].textContent = `${Math.round(value * 100)}%`;
  };

  const syncFormFromSim = () => {
    for (const [name, input] of Object.entries(ui.controls)) {
      if (!input) continue;
      input.value = String(Math.round(sim.config[name] * 100));
      syncControlText(name, sim.config[name]);
    }
    if (ui.volume) ui.volume.value = String(Math.round(sim.config.masterVolume * 100));
    if (ui.volumeOut) ui.volumeOut.textContent = `${Math.round(sim.config.masterVolume * 100)}%`;
  };

  for (const [name, input] of Object.entries(ui.controls)) {
    const v = Number(input.value) / 100;
    sim.setControl(name, v);
    syncControlText(name, v);
    input.addEventListener("input", () => {
      const raw = Number(input.value) / 100;
      sim.setControl(name, raw);
      syncControlText(name, raw);
    });
  }

  if (ui.volume && ui.volumeOut) {
    const syncVolume = (v) => {
      sim.config.masterVolume = clamp(v, 0, 1);
      ui.volumeOut.textContent = `${Math.round(sim.config.masterVolume * 100)}%`;
      if (sim.audioEngine) sim.audioEngine.setVolume(sim.config.masterVolume);
    };
    syncVolume(Number(ui.volume.value) / 100);
    ui.volume.addEventListener("input", () => syncVolume(Number(ui.volume.value) / 100));
  }

  const setModeUI = (mode) => {
    sim.setMode(mode);
    const info = MODE_INFO[mode];
    ui.modeTitle.textContent = info.title;
    ui.modeDescription.textContent = info.description;
    ui.modeTabs.forEach((btn) => {
      const active = btn.dataset.mode === mode;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", String(active));
    });
  };
  ui.modeTabs.forEach((btn) => btn.addEventListener("click", () => setModeUI(btn.dataset.mode)));
  setModeUI(sim.config.mode);

  ui.pauseBtn.addEventListener("click", () => {
    sim.setPaused(!sim.config.paused);
    ui.pauseBtn.setAttribute("aria-pressed", String(sim.config.paused));
    ui.pauseBtn.textContent = sim.config.paused ? "Resume" : "Pause";
  });
  ui.resetBtn.addEventListener("click", () => sim.reseed());
  ui.challengeBtn?.addEventListener("click", () => sim.restartChallenge());
  ui.startMissionBtn?.addEventListener("click", () => sim.restartChallenge());
  ui.pulseBtn?.addEventListener("click", () => sim.triggerManualPulse(1));

  const toggleAudio = async () => {
    if (!sim.audioEngine) return;
    const enabled = await sim.audioEngine.toggle();
    const label = enabled ? "Audio: On" : "Audio: Off";
    if (ui.audioBtn) {
      ui.audioBtn.textContent = label;
      ui.audioBtn.setAttribute("aria-pressed", String(enabled));
    }
    if (ui.audioToggleBtn) {
      ui.audioToggleBtn.textContent = enabled ? "Disable Audio" : "Enable Audio";
      ui.audioToggleBtn.setAttribute("aria-pressed", String(enabled));
    }
  };
  ui.audioBtn?.addEventListener("click", toggleAudio);
  ui.audioToggleBtn?.addEventListener("click", toggleAudio);

  ui.exportJsonBtn?.addEventListener("click", () => {
    const text = sim.exportBridgeProfile();
    if (ui.bridgePreview) ui.bridgePreview.textContent = text.slice(0, 1600);
    if (ui.bridgeImportInput) ui.bridgeImportInput.value = text;
    if (ui.bridgeStatus) ui.bridgeStatus.textContent = "Exported JSON bridge profile from current live state.";
  });
  ui.copyCppBtn?.addEventListener("click", async () => {
    const snippet = sim.getBridgeCppSnippet();
    let copied = false;
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(snippet);
        copied = true;
      } catch {
        copied = false;
      }
    }
    if (ui.bridgePreview) ui.bridgePreview.textContent = snippet;
    if (ui.bridgeImportInput) ui.bridgeImportInput.value = JSON.stringify(sim.getBridgeProfile(), null, 2);
    if (ui.bridgeStatus) {
      ui.bridgeStatus.textContent = copied
        ? "Copied C++ preset snippet to clipboard."
        : "C++ preset snippet generated (clipboard unavailable; preview shown below).";
    }
  });

  if (ui.bridgePreview) {
    ui.bridgePreview.textContent = sim.getBridgeCppSnippet();
  }
  if (ui.bridgeImportInput) {
    ui.bridgeImportInput.value = JSON.stringify(sim.getBridgeProfile(), null, 2);
  }

  ui.applyBridgeBtn?.addEventListener("click", () => {
    if (!ui.bridgeImportInput) return;
    try {
      sim.importBridgeProfileText(ui.bridgeImportInput.value);
      syncFormFromSim();
      if (ui.bridgeStatus) ui.bridgeStatus.textContent = "Imported bridge profile applied.";
    } catch (err) {
      if (ui.bridgeStatus) ui.bridgeStatus.textContent = `Import failed: ${err.message}`;
    }
  });

  ui.loadBridgeFileBtn?.addEventListener("click", () => ui.bridgeFileInput?.click());
  ui.bridgeFileInput?.addEventListener("change", async () => {
    const file = ui.bridgeFileInput.files?.[0];
    if (!file) return;
    const text = await file.text();
    if (ui.bridgeImportInput) ui.bridgeImportInput.value = text;
    if (ui.bridgeStatus) ui.bridgeStatus.textContent = `Loaded file: ${file.name}`;
  });

  ui.saveLocalBridgeBtn?.addEventListener("click", () => {
    try {
      const text = sim.saveBridgePresetLocal();
      if (ui.bridgeImportInput) ui.bridgeImportInput.value = JSON.stringify(JSON.parse(text), null, 2);
      if (ui.bridgeStatus) ui.bridgeStatus.textContent = "Saved current bridge profile to localStorage.";
    } catch (err) {
      if (ui.bridgeStatus) ui.bridgeStatus.textContent = `Save failed: ${err.message}`;
    }
  });

  ui.loadLocalBridgeBtn?.addEventListener("click", () => {
    try {
      const profile = sim.loadBridgePresetLocal();
      syncFormFromSim();
      if (ui.bridgeImportInput) ui.bridgeImportInput.value = JSON.stringify(profile, null, 2);
      if (ui.bridgeStatus) ui.bridgeStatus.textContent = "Loaded and applied local bridge preset.";
    } catch (err) {
      if (ui.bridgeStatus) ui.bridgeStatus.textContent = `Load failed: ${err.message}`;
    }
  });

  ui.reportRestartBtn?.addEventListener("click", () => {
    sim.setReportVisible(false);
    sim.restartChallenge();
  });
  ui.reportCloseBtn?.addEventListener("click", () => sim.setReportVisible(false));

  attachPointerControls(ui.canvas, sim);
  attachKeyboardControls(sim);
  return ui;
}

function attachPointerControls(canvas, sim) {
  canvas.style.touchAction = "none";

  canvas.addEventListener("pointerdown", (e) => {
    sim.pointer.down = true;
    sim.pointer.id = e.pointerId;
    sim.pointer.lastX = e.clientX;
    sim.pointer.lastY = e.clientY;
    canvas.setPointerCapture?.(e.pointerId);
    sim.config.autoOrbit = 0;
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!sim.pointer.down || sim.pointer.id !== e.pointerId) return;
    const dx = e.clientX - sim.pointer.lastX;
    const dy = e.clientY - sim.pointer.lastY;
    sim.pointer.lastX = e.clientX;
    sim.pointer.lastY = e.clientY;
    sim.config.yaw += dx * 0.006;
    sim.config.pitch = clamp(sim.config.pitch + dy * 0.004, -1.15, 1.1);
    sim.config.userOrbitImpulse = clamp(dx * 0.0015, -0.12, 0.12);
  });

  const endPointer = (e) => {
    if (e?.pointerId !== undefined && sim.pointer.id !== e.pointerId) return;
    sim.pointer.down = false;
    sim.pointer.id = null;
    sim.config.autoOrbit = 1;
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = Math.sign(e.deltaY);
    sim.config.zoom = clamp(sim.config.zoom * (delta > 0 ? 0.94 : 1.06), 0.72, 1.55);
    sim.config.autoOrbit = 0;
    clearTimeout(canvas._orbitTimeout);
    canvas._orbitTimeout = setTimeout(() => { sim.config.autoOrbit = 1; }, 900);
  }, { passive: false });
}

function start() {
  const canvas = document.querySelector("#reactorCanvas");
  const sim = new NeuroFusionReactor(canvas, {});
  const audioEngine = new ReactorAudioEngine(sim);
  sim.setAudioEngine(audioEngine);
  setupUI(sim);
  installAutomationHooks(sim);
  window.__neurofusion = { sim, audioEngine };

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    sim.resize(rect.width, rect.height, window.devicePixelRatio || 1);
  };
  resize();
  window.addEventListener("resize", resize);
  window.addEventListener("fullscreenchange", resize);

  const loop = (time) => {
    sim.tick(time);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

start();
