# Cymatics Wave Simulation - Design Document

## What We're Building

A real-time GPU-accelerated simulation of **water surface cymatics**—the visual patterns created when audio vibrates a water surface. This replicates a physical rig: speaker facing up through water, multi-color ring light reflecting off the surface, camera capturing the result.

## The Physical Phenomenon

**Real Setup:**
- Waterproof speaker vibrates upward through water
- Creates traveling and standing waves on the surface
- Multi-color ring light positioned above reflects off the wave-distorted surface
- Camera in center of ring light captures the reflections
- Complex audio (music, data sonification) creates rich interference patterns

**What We See:**
- Not Chladni patterns (rigid plate modes)
- Continuous, fluid wave superposition
- Light caustics and reflections reveal the wave structure
- Multiple frequencies interfere to create emergent complexity

## Core Physics: The Wave Equation

Surface displacement follows: **∂²u/∂t² = c²∇²u**

Where:
- `u(x,y,t)` = height at position (x,y) at time t
- `c` = wave propagation speed
- `∇²u` = Laplacian (curvature) = how "bent" the surface is

**Physical Intuition:**
- High curvature → strong restoring force → rapid acceleration
- Flat areas → no force → no acceleration
- This creates wave propagation

**Why This Works:**
- Water surface wants to be flat (gravity + surface tension)
- Disturbances create curvature
- Curvature creates forces that propagate the disturbance
- The equation captures this naturally

## Architecture: GPU Simulation via Ping-Pong Rendering

### Why GPU?

CPU simulation at 1024×1024 = 1M pixels = too slow for real-time
GPU shader runs the same calculation on ALL pixels in parallel = fast

### Ping-Pong Pattern

We need **two time steps** to compute the next state (Verlet integration):
- `next = current + (current - previous) + acceleration * dt²`

**Implementation:**
- Two render targets: `readTarget` and `writeTarget`
- Each frame:
  1. Read from `readTarget` (has current and previous states)
  2. Compute next state in shader
  3. Write to `writeTarget`
  4. Swap targets
- This gives us the temporal evolution

### Data Storage

Each pixel stores height in the R channel:
- `texture.r` = displacement at that point
- Values typically range from -0.1 to +0.1 (adjustable)

### Shader Logic (simulation.frag)

```glsl
// Sample current and previous frames
float current = texture2D(currentState, uv).r;
float previous = texture2D(previousState, uv).r;

// Compute Laplacian (sample 4 neighbors)
float laplacian = (right + left + up + down - 4.0 * current);

// Wave equation: next = current + velocity + acceleration
float velocity = current - previous;
float acceleration = waveSpeed * laplacian;
float next = current + velocity + acceleration * dt * dt;

// Apply damping (energy loss)
next *= damping;  // 0.998 typical

// Apply forcing (audio input)
next += audioForce * spatialPattern;

// Boundary conditions (container edges)
if (outsideContainer) {
    next *= edgeDamping;
}
```

## Audio → Forcing Pipeline

### Single Frequency (Sine Wave Test)

```
Oscillator (Web Audio API)
    ↓
Generate sine at frequency f
    ↓
Current amplitude value
    ↓
Apply to center of simulation with spatial falloff
```

**Speaker Geometry:**
- Maximum displacement at center (dust cap)
- Smooth falloff to edges (speaker cone)
- Zero at surround (fixed boundary)

```glsl
float distFromCenter = length(uv - speakerPos);
float pattern = smoothstep(speakerRadius, speakerRadius * 0.7, distFromCenter);
float force = audioAmplitude * pattern;
```

### Complex Audio (Music/Data)

```
Audio File/Stream
    ↓
Web Audio API AnalyserNode
    ↓
FFT → Frequency Bins
    ↓
Map bins to spatial frequencies
    ↓
Forcing pattern that matches audio spectrum
```

**Why FFT?**
- Different frequencies create different wavelengths
- Low frequency (100 Hz) → long waves (slow oscillation)
- High frequency (2000 Hz) → short ripples (fast oscillation)
- FFT tells us which frequencies are present
- We apply forcing that matches those spatial frequencies

### Multiple Sources (Stereo/Multi-Channel)

```
Left Channel → Source Position 1
Right Channel → Source Position 2

Each source drives its location independently
Wave interference happens naturally in the simulation
```

**Why This Creates Complexity:**
- In-phase sources: constructive interference (bright patterns)
- Out-of-phase sources: destructive interference (dark nodes)
- Different audio content: asymmetric, evolving patterns
- Like magnetic field lines from multiple poles

## Visualization Pipeline

We have the height field. Now we render it.

### Option 1: Height Coloring (Debug/Simple)

```glsl
float height = texture(heightField, uv).r;
vec3 color = mix(blue, yellow, normalize(height));
```

Simple, direct, good for verifying physics works.

### Option 2: Normal/Gradient Coloring

```glsl
vec3 normal = computeNormal(heightField, uv);
float steepness = length(normal.xy);
vec3 color = mix(darkBlue, brightCyan, steepness);
```

Shows wave activity—steep slopes are bright, flat areas dark.

### Option 3: Reflection Rendering (Physical Match)

```glsl
vec3 normal = computeNormal(heightField, uv);
vec3 viewDir = normalize(cameraPos - surfacePos);
vec3 reflected = reflect(-viewDir, normal);

// Sample ring light environment
vec3 color = sampleEnvironment(reflected);
```

**This is what the camera sees:**
- Surface normals bend light
- Ring light reflects off distorted surface
- Camera captures the reflected light
- Caustics emerge naturally

**Ring Light Implementation:**
- Environment map (texture or procedural)
- RGB positioned at different angles
- Creates chromatic effects when surface is steep

## Parameters & Tuning

### Physics Parameters

- **waveSpeed** (0.1 - 2.0): How fast waves propagate
  - Lower = slow, graceful
  - Higher = fast, energetic

- **damping** (0.95 - 0.9999): Energy loss per frame
  - 0.95 = waves die quickly
  - 0.999 = waves persist, build up resonance

- **containerRadius** (0.3 - 1.0): Size of circular boundary
  - Affects available wavelengths
  - Determines reflection patterns

### Audio Parameters

- **forceStrength**: Amplitude of forcing
  - Too low: barely visible
  - Too high: simulation goes unstable
  - Typical: 0.01 - 0.05

- **speakerRadius**: Size of forcing region
  - Matches physical speaker size
  - Affects how localized the forcing is

### Boundary Conditions

- **Hard walls**: Perfect reflection, standing waves
- **Soft walls**: Energy absorption, flowing patterns
- **Mixed**: Can vary by angle or frequency

## Why This Approach?

**Surface-Only (2D) vs Volumetric (3D):**
- Camera only sees surface reflections
- Surface geometry is all that matters for visuals
- 3D fluid sim would be 100x more expensive
- Doesn't change what you see
- 2D wave equation captures the essential physics

**GPU Shaders vs CPU:**
- 1024×1024 = 1M calculations per frame
- At 60fps = 60M calculations per second
- GPU does this in parallel effortlessly
- CPU would struggle at 256×256

**Verlet Integration:**
- Stable for wave equations
- Conserves energy (with damping parameter)
- Simple to implement
- Two-frame history is sufficient

**Ping-Pong Rendering:**
- Standard technique for iterative simulations
- Avoids read/write conflicts
- Allows temporal evolution
- GPU-friendly pattern

## Extension Points

### Multi-Source Capability

```javascript
sources = [
    { pos: [0.3, 0.5], channel: 'left' },
    { pos: [0.7, 0.5], channel: 'right' },
    // ... more sources
];

// In shader: sum all forcing contributions
for (each source) {
    totalForce += source.pattern * source.audio;
}
```

### Different Visualization Modes

Toggle between height/normal/reflection rendering
Blend modes for hybrid visuals
User-controllable parameters

### Resolution Scaling

Detect device capability
Scale simulation resolution (512/1024/2048)
Keep visual resolution high (can be independent)

### Audio Source Variety

- Microphone input (live)
- File upload
- GitHub/external URLs
- Generated tones/noise
- Multi-channel routing

## Development Philosophy

**Start Simple, Add Complexity:**
1. Does wave equation work? (tap to create waves)
2. Does sine wave forcing work? (single frequency)
3. Does audio forcing work? (real audio)
4. Does visualization look good? (height → normals → reflections)
5. Multi-source? (stereo)
6. Performance? (1024×1024 at 60fps)

**Each step validates the previous:**
If step N doesn't work, step N+1 won't either.
Build the foundation solid, then add layers.

## Success Criteria

**Technical:**
- 60fps at target resolution on target device
- Audio latency < 50ms (forcing responds to sound)
- Stable simulation (no numerical explosion)
- Accurate wave propagation (verified against known patterns)

**Aesthetic:**
- Visually compelling (people want to watch it)
- Reflects complexity of input (rich audio → rich patterns)
- Matches or exceeds physical rig beauty

---

## TL;DR for Another AI

We're simulating water surface waves driven by audio. The wave equation runs on GPU via shaders using ping-pong rendering. Audio (via FFT or direct amplitude) drives forcing at speaker positions. The height field is rendered using surface normals to show reflections of a ring light, matching the physical rig setup. Multiple sources create interference patterns. The whole thing runs real-time in browser at 1024×1024 resolution.
