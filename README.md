# NeuroFusion Reactor

`NeuroFusion Reactor` is a standalone interactive prototype that hybridizes:

- `FusionCpp` (tokamak fusion / plasma / confinement simulation ideas)
- Denis Dmitriev's "Neural Network 3D Simulation" (3D MLP/CNN/SNN visual language)

It turns the tokamak torus into a neural control substrate and layers a playable challenge mode on top.

## Concept

The core idea is a "neuralized tokamak" where plasma telemetry is fed into stylized neural controllers that directly modulate magnetic behavior:

- `MLP Coil Brain`
  - global feed-forward control for broad stabilization
- `CNN Flux Vision`
  - circular convolution over torus sensor bins to find/quench hotspots
- `SNN Spike Grid`
  - event-driven spike pulses for burst timing and rhythmic control

Controller outputs drive:

- toroidal response
- poloidal response
- pulse injection / burst behavior

## Features

### Simulation + Visuals

- Real-time torus plasma visualization (canvas-based)
- Neural overlays for `MLP`, `CNN`, and `SNN` modes
- Fusion burst glows and field-line rendering
- Orbit camera with drag + scroll zoom

### Mission / Game Layer

- 3-phase challenge mode
  - Phase 1: Edge stabilization (`MLP`)
  - Phase 2: Hotspot quench (`CNN`)
  - Phase 3: Spike burn window (`SNN`)
- Score system with breakdown
  - base scoring
  - mode bonuses
  - mode penalties
  - phase clear bonuses
  - pulse bonuses
- Integrity + timer failure conditions
- Mission report overlay (win/lose summary)
- Mode usage analytics and mode switch tracking

### Audio (Web Audio)

- Live sonification of:
  - spikes
  - bursts
  - yield
  - instability
- Master volume slider
- Audio enable/disable controls

### FusionCpp Bridge (v2)

- Export current live state as JSON profile (`neurofusion-bridge-v2`)
- Copy a generated C++ preset snippet
- Paste/import bridge JSON and apply controls/mode/camera
- Load bridge JSON from file
- Save/load local bridge preset (`localStorage`)

## Run Locally

From `neurofusion-reactor/`:

```bash
python -m http.server 4173
```

Open:

```text
http://127.0.0.1:4173
```

## Controls

### Mouse

- Drag: orbit camera
- Mouse wheel: zoom

### Keyboard

- `Enter`: start/restart mission
- `Space`: manual pulse
- `1` / `2` / `3`: switch `MLP / CNN / SNN`
- `ArrowLeft` / `ArrowRight`: heating down/up
- `ArrowDown` / `ArrowUp`: magnetic gain down/up
- `A`: adaptive learning up
- `B`: turbulence bias up
- `P`: pause/resume
- `R`: re-seed simulation
- `F`: toggle fullscreen
- `O`: toggle mission report overlay
- `Esc`: close mission report overlay (if open)

### UI

- Mode tabs: switch controller mode
- Sliders: tune control loop parameters
- Mission panel: start/restart mission, manual pulse
- Audio panel: enable audio + adjust volume
- Bridge panel: export/import/apply presets

## Automation Hooks (Testing / Bots)

The app exposes helpers for automated browser control:

- `window.render_game_to_text()`
  - returns concise JSON state for the current simulation/game state
- `window.advanceTime(ms)`
  - deterministic stepping for automated test loops
- `window.get_bridge_profile()`
  - returns current bridge JSON object
- `window.apply_bridge_profile(profileOrText)`
  - applies imported bridge profile (object or JSON string)

These hooks are useful for Playwright-based testing and scripted scenarios.

## Project Files

- `index.html` - UI layout and panels
- `styles.css` - visual styling and responsive layout
- `app.js` - simulation, rendering, mission systems, audio engine, bridge logic
- `favicon.svg` - app icon
- `progress.md` - build log / handoff notes

## Notes / Limitations

- This is a creative prototype, not a physically accurate reactor controller.
- The bridge export is a practical integration layer for FusionCpp, but FusionCpp-side preset loading is not implemented here.
- Audio starts only after user interaction due to browser autoplay restrictions.

## Credits / Inspiration

- `FusionCpp` by Amine Harrabi (tokamak fusion simulation inspiration)
- Denis Dmitriev's "Neural Network 3D Simulation" (visual + conceptual inspiration for MLP/CNN/SNN modes)
