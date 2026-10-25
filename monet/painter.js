// Monet for the web: drop in a picture and he paints it in oil, stroke by stroke.
//
// A port of the Automonet desktop app. Paint is simulated per wavelength with Kubelka-Munk: every pixel of the canvas
// keeps its reflectance spectrum and how much of every paint is on it, in float textures on the graphics card
// (WebGL 2), and every stroke is laid over it as its own layer of paint by a shader (PaintCanvas).
// Deciding where strokes go (passes from color masses down to highlights, local Shannon entropy, the impressionist
// color shift) runs in JavaScript (AutoPainter), the same way as in autopainter.py.

"use strict";

const Monet = (() => {
  // ---------------- Color ----------------

  // Spectra are sampled every 20 nm from 400 to 700 nm: 16 samples, kept in four RGBA textures
  const SAMPLES = 16;
  const WAVELENGTHS = Array.from({ length: SAMPLES }, (_, i) => 400 + 20 * i);

  // CIE 1931 standard observer under D65 daylight at those wavelengths, scaled so a perfect white has Y = 1
  const XYZ_WEIGHTS = [
    [0.0022363, 0.0000619, 0.0106035], [0.0237102, 0.0007058, 0.1139104], [0.0689707, 0.0045547, 0.3459742],
    [0.0646978, 0.0133489, 0.3713669], [0.0209370, 0.0304335, 0.1779667], [0.0010119, 0.0667026, 0.0561706],
    [0.0125205, 0.1405023, 0.0154849], [0.0572563, 0.1880940, 0.0040024], [0.1122683, 0.1879007, 0.0007365],
    [0.1657502, 0.1573750, 0.0002985], [0.1805444, 0.1072524, 0.0001360], [0.1415093, 0.0630991, 0.0000315],
    [0.0707958, 0.0276608, 0.0000032], [0.0249792, 0.0092404, 0.0000000], [0.0069143, 0.0025132, 0.0000000],
    [0.0015361, 0.0005547, 0.0000000],
  ];
  const RGB_TO_XYZ = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]];
  const XYZ_TO_RGB = invert(RGB_TO_XYZ);
  // For comparing paint with pictures in CIELAB, white is a perfectly reflecting surface
  const WHITE_XYZ = [0, 1, 2].map((axis) => XYZ_WEIGHTS.reduce((sum, weights) => sum + weights[axis], 0));
  // Reflectance spectrum -> linear sRGB in one matrix, a row per sample
  const SPECTRUM_TO_RGB = XYZ_WEIGHTS.map((weights) => XYZ_TO_RGB.map((row) => row[0] * weights[0] + row[1] * weights[1] + row[2] * weights[2]));
  // 8 bit sRGB -> linear
  const SRGB_TO_LINEAR = Float32Array.from({ length: 256 }, (_, value) => {
    const c = value / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });

  function invert(m) {
    const [[a, b, c], [d, e, f], [g, h, i]] = m;
    const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    return [
      [(e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant],
      [(f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant],
      [(d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant],
    ];
  }

  function labCurve(t) {
    return t > (6 / 29) ** 3 ? Math.cbrt(t) : t / (3 * (6 / 29) ** 2) + 4 / 29;
  }

  function xyzToLab(X, Y, Z) {
    const fx = labCurve(X / WHITE_XYZ[0]), fy = labCurve(Y / WHITE_XYZ[1]), fz = labCurve(Z / WHITE_XYZ[2]);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  }

  function spectrumToLab(R) {
    let X = 0, Y = 0, Z = 0;
    for (let i = 0; i < SAMPLES; i++) {
      X += R[i] * XYZ_WEIGHTS[i][0];
      Y += R[i] * XYZ_WEIGHTS[i][1];
      Z += R[i] * XYZ_WEIGHTS[i][2];
    }
    return xyzToLab(X, Y, Z);
  }

  function spectrumToCss(R) {
    // Display color of a reflectance spectrum, as a CSS color
    const channels = [0, 1, 2].map((channel) => {
      let linear = 0;
      for (let i = 0; i < SAMPLES; i++) linear += R[i] * SPECTRUM_TO_RGB[i][channel];
      linear = Math.min(Math.max(linear, 0), 1);
      return Math.round(255 * (linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055));
    });
    return `rgb(${channels.join(", ")})`;
  }

  function rgbaToLab(pixels, count) {
    // 8 bit RGBA pixels -> CIELAB, as three maps
    const L = new Float32Array(count), A = new Float32Array(count), B = new Float32Array(count);
    const [m0, m1, m2] = RGB_TO_XYZ;
    for (let i = 0; i < count; i++) {
      const r = SRGB_TO_LINEAR[pixels[4 * i]], g = SRGB_TO_LINEAR[pixels[4 * i + 1]], b = SRGB_TO_LINEAR[pixels[4 * i + 2]];
      const fx = labCurve((m0[0] * r + m0[1] * g + m0[2] * b) / WHITE_XYZ[0]);
      const fy = labCurve((m1[0] * r + m1[1] * g + m1[2] * b) / WHITE_XYZ[1]);
      const fz = labCurve((m2[0] * r + m2[1] * g + m2[2] * b) / WHITE_XYZ[2]);
      L[i] = 116 * fy - 16;
      A[i] = 500 * (fx - fy);
      B[i] = 200 * (fy - fz);
    }
    return [L, A, B];
  }

  // ---------------- Paints ----------------

  // Monet's own list from 1905 ("flake white, cadmium yellow, vermilion, deep madder, cobalt blue, emerald green")
  // plus the ultramarine and viridian found in his paintings. hue/saturation/brightness give the color straight
  // from the tube, softness how sharp the pigment's absorption edge is, strength how well it covers
  const PALETTE = [
    { name: "lead white", hue: 40, saturation: 0.04, brightness: 0.97, softness: 20, strength: 6.0 },
    { name: "cadmium yellow", hue: 50, saturation: 1.0, brightness: 0.95, softness: 4, strength: 3.0 },
    { name: "vermilion", hue: 6, saturation: 0.95, brightness: 0.88, softness: 4, strength: 3.0 },
    { name: "madder lake", hue: 344, saturation: 0.9, brightness: 0.28, softness: 12, strength: 0.4 },
    { name: "cobalt blue", hue: 216, saturation: 0.85, brightness: 0.5, softness: 14, strength: 1.0 },
    { name: "ultramarine", hue: 236, saturation: 0.92, brightness: 0.22, softness: 10, strength: 0.4 },
    { name: "viridian", hue: 168, saturation: 0.85, brightness: 0.12, softness: 16, strength: 0.4 },
    { name: "emerald green", hue: 148, saturation: 0.8, brightness: 0.8, softness: 12, strength: 1.5 },
  ];

  // Paper reflectance at 400 nm and 700 nm
  const PAPERS = { white: [0.95, 0.95], cream: [0.7, 0.92], grey: [0.35, 0.35], black: [0.005, 0.005] };

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.min(Math.max(x, -60), 60)));
  }

  function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  }

  function makePaint(paint) {
    // Kubelka-Munk absorption (K, per sample) and scattering (S, the same at every wavelength) of a paint
    const [r, g, b] = hsvToRgb(paint.hue / 360, paint.saturation, paint.brightness);
    const K = WAVELENGTHS.map((wavelength) => {
      const blue = sigmoid((500 - wavelength) / paint.softness);
      const red = sigmoid((wavelength - 600) / paint.softness);
      const green = (1 - blue) * (1 - red);
      const R = 0.02 + 0.98 * (r * red + g * green + b * blue) / (red + green + blue);
      return paint.strength * (1 - R) ** 2 / (2 * R);
    });
    return { name: paint.name, K, S: paint.strength };
  }

  const PAINTS = PALETTE.map(makePaint);

  function getOpaqueReflectance(K, S) {
    // Reflectance of a layer so thick that nothing underneath shows through
    return K.map((k) => {
      const ratio = k / S;
      return 1 + ratio - Math.sqrt(ratio * ratio + 2 * ratio);
    });
  }

  function getPaperReflectance(paper) {
    const [low, high] = PAPERS[paper] || PAPERS.white;
    return Float32Array.from(WAVELENGTHS, (wavelength) => low + (high - low) * (wavelength - 400) / 300);
  }

  // ---------------- Mixes ----------------

  class KDTree {
    // Static tree over 3d points (interleaved x, y, z) for finding the nearest one quickly. The tree is implicit:
    // every range of the sorted index is split at its middle point along the axis stored for it
    constructor(points, count) {
      this.points = points;
      this.index = Int32Array.from({ length: count }, (_, i) => i);
      this.axes = new Uint8Array(count);
      this.build(0, count);
    }

    build(low, high) {
      if (high - low < 2) return;
      const p = this.points, index = this.index;
      // Split along the axis the points spread most
      let axis = 0, spread = -1;
      for (let a = 0; a < 3; a++) {
        let min = Infinity, max = -Infinity;
        for (let i = low; i < high; i++) {
          const value = p[3 * index[i] + a];
          if (value < min) min = value;
          if (value > max) max = value;
        }
        if (max - min > spread) [axis, spread] = [a, max - min];
      }
      index.subarray(low, high).sort((i, j) => p[3 * i + axis] - p[3 * j + axis]);
      const middle = (low + high) >> 1;
      this.axes[middle] = axis;
      this.build(low, middle);
      this.build(middle + 1, high);
    }

    nearest(x, y, z) {
      this.query = [x, y, z];
      this.best = -1;
      this.bestDistance = Infinity;
      this.search(0, this.index.length);
      return this.best;
    }

    search(low, high) {
      if (low >= high) return;
      const middle = (low + high) >> 1, point = this.index[middle], p = this.points, q = this.query;
      const dx = q[0] - p[3 * point], dy = q[1] - p[3 * point + 1], dz = q[2] - p[3 * point + 2];
      const distance = dx * dx + dy * dy + dz * dz;
      if (distance < this.bestDistance) [this.best, this.bestDistance] = [point, distance];
      const offset = q[this.axes[middle]] - p[3 * point + this.axes[middle]];
      if (offset < 0) {
        this.search(low, middle);
        if (offset * offset < this.bestDistance) this.search(middle + 1, high);
      } else {
        this.search(middle + 1, high);
        if (offset * offset < this.bestDistance) this.search(low, middle);
      }
    }
  }

  function* getCompositions(total, count) {
    // All ways to split total into count whole, non-negative parts
    if (count === 1) {
      yield [total];
      return;
    }
    for (let first = 0; first <= total; first++) {
      for (const rest of getCompositions(total - first, count - 1)) yield [first, ...rest];
    }
  }

  let mixes = null;

  function getMixes(parts = 12) {
    // Every mix of up to three palette paints in steps of 1/parts, and how each one looks as a thick layer.
    // Built once, the palette never changes here
    if (mixes) return mixes;
    const count = PAINTS.length;
    const rows = new Map();
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        for (let k = j + 1; k < count; k++) {
          for (const split of getCompositions(parts, 3)) {
            const weights = new Array(count).fill(0);
            [i, j, k].forEach((paint, n) => (weights[paint] = split[n] / parts));
            rows.set(weights.join(), weights);
          }
        }
      }
    }

    // Pale tints the way painters make them, by adding a lot of the lightest paint, which 1/parts steps can't reach
    const lightness = PAINTS.map((paint) => spectrumToLab(getOpaqueReflectance(paint.K, paint.S))[0]);
    const white = lightness.indexOf(Math.max(...lightness));
    const all = [];
    for (const amount of [0, 0.5, 0.7, 0.85, 0.93, 0.97]) {
      for (const weights of rows.values()) {
        const tint = weights.map((w) => w * (1 - amount));
        tint[white] += amount;
        all.push(tint);
      }
    }

    const total = all.length;
    const weights = new Float32Array(total * count), S = new Float32Array(total), lab = new Float32Array(total * 3);
    all.forEach((w, mix) => {
      weights.set(w, mix * count);
      const K = new Array(SAMPLES).fill(0);
      let s = 0;
      w.forEach((share, paint) => {
        if (share === 0) return;
        PAINTS[paint].K.forEach((k, sample) => (K[sample] += share * k));
        s += share * PAINTS[paint].S;
      });
      S[mix] = s;
      lab.set(spectrumToLab(getOpaqueReflectance(K, s)), mix * 3);
    });
    const tree = new KDTree(lab, total);
    // Nearest mixes of colors already looked up, by color rounded to half a CIELAB unit
    const cache = new Map();

    mixes = {
      weights, S, lab, tree, count: total,
      nearest: (L, a, b) => tree.nearest(L, a, b),
      nearestRounded(L, a, b) {
        const key = (Math.round(L * 2) + 512) * 1048576 + (Math.round(a * 2) + 512) * 1024 + (Math.round(b * 2) + 512);
        let mix = cache.get(key);
        if (mix === undefined) {
          if (cache.size > 400000) cache.clear();
          mix = tree.nearest(L, a, b);
          cache.set(key, mix);
        }
        return mix;
      },
    };
    return mixes;
  }

  // ---------------- Maps ----------------

  // Maps are Float32Arrays of width * height values, row by row

  function boxFilter(map, width, height, left, right, out = new Float32Array(map.length)) {
    // Mean over a window from left before to right after every value, along rows and then columns. Values beyond
    // the edge repeat the nearest edge value
    const scale = 1 / (left + right + 1);
    const temp = new Float32Array(map.length);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      let sum = 0;
      for (let k = -left; k <= right; k++) sum += map[row + Math.min(Math.max(k, 0), width - 1)];
      for (let x = 0; x < width; x++) {
        temp[row + x] = sum * scale;
        sum += map[row + Math.min(x + right + 1, width - 1)] - map[row + Math.max(x - left, 0)];
      }
    }
    // All columns at once, so memory is read row by row
    const sums = new Float64Array(width);
    for (let k = -left; k <= right; k++) {
      const row = Math.min(Math.max(k, 0), height - 1) * width;
      for (let x = 0; x < width; x++) sums[x] += temp[row + x];
    }
    for (let y = 0; y < height; y++) {
      const row = y * width, add = Math.min(y + right + 1, height - 1) * width, remove = Math.max(y - left, 0) * width;
      for (let x = 0; x < width; x++) {
        out[row + x] = sums[x] * scale;
        sums[x] += temp[add + x] - temp[remove + x];
      }
    }
    return out;
  }

  function gaussianBlur(map, width, height, sigma) {
    // Three box filters in a row are very close to a Gaussian blur, and cost the same for any sigma
    const ideal = Math.sqrt(12 * sigma * sigma / 3 + 1);
    let lower = Math.floor(ideal);
    if (lower % 2 === 0) lower--;
    const smaller = Math.round((12 * sigma * sigma - 3 * lower * lower - 12 * lower - 9) / (-4 * lower - 4));
    let blurred = map;
    for (let i = 0; i < 3; i++) {
      const radius = ((i < smaller ? lower : lower + 2) - 1) / 2;
      if (radius > 0) blurred = boxFilter(blurred, width, height, radius, radius);
    }
    return blurred === map ? map.slice() : blurred;
  }

  function shrinkMap(map, width, height, factor) {
    // Smaller version of a map, averaged over blocks of factor x factor values. A remainder at the right and bottom
    // that doesn't fill a whole block is left out
    if (factor === 1) return map;
    const w = Math.floor(width / factor), h = Math.floor(height / factor);
    const out = new Float32Array(w * h);
    for (let y = 0; y < h * factor; y++) {
      const row = y * width, outRow = Math.floor(y / factor) * w;
      for (let x = 0; x < w * factor; x++) out[outRow + Math.floor(x / factor)] += map[row + x];
    }
    const scale = 1 / (factor * factor);
    for (let i = 0; i < out.length; i++) out[i] *= scale;
    return out;
  }

  function percentile(map, p) {
    // Percentile of the values, from at most about 50000 of them
    const step = Math.max(1, Math.floor(map.length / 50000));
    const sample = new Float32Array(Math.ceil(map.length / step));
    for (let i = 0, j = 0; i < map.length; i += step) sample[j++] = map[i];
    sample.sort();
    const position = p / 100 * (sample.length - 1), low = Math.floor(position);
    const high = Math.min(low + 1, sample.length - 1);
    return sample[low] + (sample[high] - sample[low]) * (position - low);
  }

  function sampleMap(map, width, height, x, y) {
    // Value at a position between map values (their centers are at whole numbers), interpolated
    x = Math.min(Math.max(x, 0), width - 1);
    y = Math.min(Math.max(y, 0), height - 1);
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = Math.min(x0 + 1, width - 1), y1 = Math.min(y0 + 1, height - 1);
    const fx = x - x0, fy = y - y0;
    const top = map[y0 * width + x0] * (1 - fx) + map[y0 * width + x1] * fx;
    const bottom = map[y1 * width + x0] * (1 - fx) + map[y1 * width + x1] * fx;
    return top * (1 - fy) + bottom * fy;
  }

  // What the painting generator yields: some analysis done, a stroke painted, or waiting for the draw speed
  const WORK = 0, STROKE = 1, WAIT = 2;

  function* getLocalEntropy(lab, width, height, window) {
    // Shannon entropy (bits) of the colors in a window around every value. Colors are sorted into 8 x 4 x 4 bins by
    // lightness and the two color axes. High entropy means many different colors close together: detail.
    // Every color is shared between the two nearest bins of each axis, so a smooth gradient doesn't jump from bin
    // to bin. That sharing adds some entropy even to a single flat color, so it is taken off again: what is left
    // is how much the colors in the window differ (their Jensen-Shannon divergence), zero for a flat area
    const n = width * height;
    const own = new Float32Array(n);
    // How much every value belongs to every bin of each axis, and which bins anything belongs to
    const [lightness, redGreen, yellowBlue] = [[0, 100, 8], [-50, 50, 4], [-50, 50, 4]].map(([low, high, levels], axis) => {
      const bins = Array.from({ length: levels }, () => new Float32Array(n));
      const used = new Uint8Array(levels), values = lab[axis];
      for (let i = 0; i < n; i++) {
        const position = Math.min(Math.max((values[i] - low) / (high - low) * levels - 0.5, 0), levels - 1);
        const bin = Math.min(Math.floor(position), levels - 2), upper = position - bin;
        bins[bin][i] = 1 - upper;
        bins[bin + 1][i] = upper;
        used[bin] = used[bin + 1] = 1;
        if (upper > 0 && upper < 1) own[i] -= upper * Math.log2(upper) + (1 - upper) * Math.log2(1 - upper);
      }
      return bins.filter((_, bin) => used[bin]);
    });
    const left = Math.floor(window / 2), right = window - 1 - left;
    const entropy = boxFilter(own, width, height, left, right);
    for (let i = 0; i < n; i++) entropy[i] = -entropy[i];
    yield WORK;

    const weightLA = new Float32Array(n), weight = new Float32Array(n), share = new Float32Array(n);
    for (const weightL of lightness) {
      for (const weightA of redGreen) {
        let any = false;
        for (let i = 0; i < n; i++) {
          weightLA[i] = weightL[i] * weightA[i];
          if (weightLA[i] > 0) any = true;
        }
        if (!any) continue;
        for (const weightB of yellowBlue) {
          any = false;
          for (let i = 0; i < n; i++) {
            weight[i] = weightLA[i] * weightB[i];
            if (weight[i] > 0) any = true;
          }
          if (!any) continue;
          // Share of the window's colors in this bin
          boxFilter(weight, width, height, left, right, share);
          for (let i = 0; i < n; i++) {
            const p = share[i];
            if (p > 1e-12) entropy[i] -= p * Math.log2(p);
          }
          yield WORK;
        }
      }
    }
    return entropy;
  }

  // ---------------- Random numbers ----------------

  function makeRandom(seed) {
    // Small seeded generator (mulberry32) with the distributions the painter needs
    let state = seed >>> 0;
    const random = () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return {
      random,
      uniform: (low, high) => low + (high - low) * random(),
      normal: (mean, deviation) => mean + deviation * Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random()),
    };
  }

  // ---------------- Shaders ----------------

  // Every stroke is drawn as a quad around it, all strokes of a batch in one instanced draw
  const STROKE_VERTEX = `#version 300 es
    precision highp float;
    layout(location = 0) in vec2 corner;
    layout(location = 1) in vec4 shape;      // center x, center y, angle, length
    layout(location = 2) in vec4 edge;       // width, load, raggedness, ragged phase
    layout(location = 3) in vec4 frequency;  // frequencies of three bristle streaks, wetness
    layout(location = 4) in vec4 phase;      // phases of the streaks, thickest the stroke gets
    layout(location = 5) in vec4 mixA;       // share of paints 1-4 in the stroke's mix
    layout(location = 6) in vec4 mixB;       // share of paints 5-8
    uniform vec2 canvasSize;
    flat out vec4 vShape;
    flat out vec4 vEdge;
    flat out vec4 vFrequency;
    flat out vec4 vPhase;
    flat out vec4 vMixA;
    flat out vec4 vMixB;
    void main() {
      // Far enough out for the ragged, soft edge
      float across = 0.66 * edge.x + 1.5;
      float along = shape.w * 0.5 + across;
      vec2 direction = vec2(cos(shape.z), sin(shape.z));
      vec2 position = shape.xy + direction * corner.x * along + vec2(-direction.y, direction.x) * corner.y * across;
      gl_Position = vec4(position / canvasSize * 2.0 - 1.0, 0.0, 1.0);
      vShape = shape;
      vEdge = edge;
      vFrequency = frequency;
      vPhase = phase;
      vMixA = mixA;
      vMixB = mixB;
    }`;

  const STROKE_FRAGMENT = `#version 300 es
    precision highp float;
    precision highp sampler2D;
    uniform sampler2D below0;    // reflectance spectrum of the canvas, samples 1-4
    uniform sampler2D below1;    // samples 5-8
    uniform sampler2D below2;
    uniform sampler2D below3;
    uniform sampler2D amounts0;  // how much of paints 1-4 is on the canvas
    uniform sampler2D amounts1;  // paints 5-8
    uniform vec4 paintK[32];     // K of paint i at samples 4j to 4j+3 is paintK[4i + j]
    uniform vec4 strengthA;      // S of paints 1-4
    uniform vec4 strengthB;
    uniform vec4 paper[4];
    uniform float amountScale;
    uniform int part;            // 0 lays the stroke, 1 adds its paint to the amounts
    flat in vec4 vShape;
    flat in vec4 vEdge;
    flat in vec4 vFrequency;
    flat in vec4 vPhase;
    flat in vec4 vMixA;
    flat in vec4 vMixB;
    layout(location = 0) out vec4 out0;
    layout(location = 1) out vec4 out1;
    layout(location = 2) out vec4 out2;
    layout(location = 3) out vec4 out3;
    const float PI = 3.14159265;

    vec4 getLayerReflectance(vec4 K, float S, vec4 below) {
      // Kubelka-Munk for a layer of paint of finite thickness (K and S already multiplied by it) on top of below
      if (S < 1e-8) return below;
      vec4 a = 1.0 + K / S;
      vec4 b = sqrt(max(a * a - 1.0, 0.0));
      vec4 z = b * S;
      vec4 e = exp(-2.0 * max(z, 1e-4));
      // b * coth(b * S), with its series expansion where b * S is tiny
      vec4 c = mix(1.0 / S + b * b * S / 3.0, b * (1.0 + e) / (1.0 - e), step(1e-4, z));
      return clamp((1.0 - below * (a - c)) / (a + c - below), 0.0, 1.0);
    }

    vec4 getMixK(vec4 shareA, vec4 shareB, int j) {
      return shareA.x * paintK[j] + shareA.y * paintK[4 + j] + shareA.z * paintK[8 + j] + shareA.w * paintK[12 + j]
           + shareB.x * paintK[16 + j] + shareB.y * paintK[20 + j] + shareB.z * paintK[24 + j] + shareB.w * paintK[28 + j];
    }

    float getThickness() {
      // Position along (u) and across (v) the stroke
      vec2 d = gl_FragCoord.xy - vShape.xy;
      float c = cos(vShape.z), s = sin(vShape.z), len = vShape.w, width = vEdge.x;
      float u = d.x * c + d.y * s;
      float v = -d.x * s + d.y * c;

      // A stroke that tapers towards both ends, with ragged soft edges
      float along = clamp(u, -len * 0.5, len * 0.5);
      float off = length(vec2(u - along, v));
      float taper = 0.7 + 0.3 * sin(PI * (along + len * 0.5) / len);
      float ragged = 1.0 + 0.12 * sin(u * vEdge.z * 2.0 * PI / max(width, 1.0) + vEdge.w);
      float body = clamp((width * 0.5 * taper * ragged - off) / max(width * 0.2, 0.7) + 0.5, 0.0, 1.0);

      // The brush runs out of paint along the stroke and the bristles leave streaks
      float unloading = clamp(1.0 - 0.45 * (u + len * 0.5) / len, 0.55, 1.0);
      vec3 streak = sin(v * vFrequency.xyz + vPhase.xyz);
      float streaks = 0.7 + 0.3 * (streak.x + streak.y + streak.z) / 3.0;
      return body * unloading * streaks * vEdge.y;
    }

    vec4 layStroke(int j, vec4 below, float t, float S, vec4 nowA, vec4 nowB, float S0) {
      // The part of the stroke that doesn't blend is laid on top as its own layer. The blended part shows the paint
      // mixed with all the paint already there, strongest where the stroke is thickest
      float wetness = vFrequency.w;
      vec4 layered = getLayerReflectance(getMixK(vMixA, vMixB, j) * ((1.0 - wetness) * t), S * (1.0 - wetness) * t, below);
      if (wetness <= 0.0) return layered;
      vec4 mixed = getLayerReflectance(getMixK(nowA, nowB, j), S0, paper[j]);
      return mix(layered, mixed, wetness * t / vPhase.w);
    }

    void main() {
      ivec2 pixel = ivec2(gl_FragCoord.xy);
      vec4 R0 = texelFetch(below0, pixel, 0);
      vec4 R1 = texelFetch(below1, pixel, 0);
      vec4 R2 = texelFetch(below2, pixel, 0);
      vec4 R3 = texelFetch(below3, pixel, 0);
      vec4 storedA = texelFetch(amounts0, pixel, 0);
      vec4 storedB = texelFetch(amounts1, pixel, 0);
      float t = getThickness();
      if (t > 1e-3) {
        vec4 nowA = storedA * amountScale + t * vMixA;
        vec4 nowB = storedB * amountScale + t * vMixB;
        if (part == 1) {
          storedA = nowA / amountScale;
          storedB = nowB / amountScale;
        } else {
          float S = dot(vMixA, strengthA) + dot(vMixB, strengthB);
          float S0 = dot(nowA, strengthA) + dot(nowB, strengthB);
          R0 = layStroke(0, R0, t, S, nowA, nowB, S0);
          R1 = layStroke(1, R1, t, S, nowA, nowB, S0);
          R2 = layStroke(2, R2, t, S, nowA, nowB, S0);
          R3 = layStroke(3, R3, t, S, nowA, nowB, S0);
        }
      }
      // Pixels of the quad the stroke doesn't reach are copied unchanged
      if (part == 1) {
        out0 = storedA;
        out1 = storedB;
      } else {
        out0 = R0;
        out1 = R1;
        out2 = R2;
        out3 = R3;
      }
    }`;

  const COPY_FRAGMENT = `#version 300 es
    precision highp float;
    precision highp sampler2D;
    uniform sampler2D source0;
    uniform sampler2D source1;
    uniform sampler2D source2;
    uniform sampler2D source3;
    layout(location = 0) out vec4 out0;
    layout(location = 1) out vec4 out1;
    layout(location = 2) out vec4 out2;
    layout(location = 3) out vec4 out3;
    void main() {
      ivec2 pixel = ivec2(gl_FragCoord.xy);
      out0 = texelFetch(source0, pixel, 0);
      out1 = texelFetch(source1, pixel, 0);
      out2 = texelFetch(source2, pixel, 0);
      out3 = texelFetch(source3, pixel, 0);
    }`;

  // One triangle covering the whole target
  const FULL_VERTEX = `#version 300 es
    out vec2 uv;
    void main() {
      uv = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0;
      gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    }`;

  const TO_SRGB = `
    uniform sampler2D below0;
    uniform sampler2D below1;
    uniform sampler2D below2;
    uniform sampler2D below3;
    uniform vec4 toR[4];  // spectrum -> linear sRGB
    uniform vec4 toG[4];
    uniform vec4 toB[4];
    vec3 toSRGB(vec4 R0, vec4 R1, vec4 R2, vec4 R3) {
      vec3 linear = vec3(
        dot(R0, toR[0]) + dot(R1, toR[1]) + dot(R2, toR[2]) + dot(R3, toR[3]),
        dot(R0, toG[0]) + dot(R1, toG[1]) + dot(R2, toG[2]) + dot(R3, toG[3]),
        dot(R0, toB[0]) + dot(R1, toB[1]) + dot(R2, toB[2]) + dot(R3, toB[3]));
      linear = clamp(linear, 0.0, 1.0);
      return mix(12.92 * linear, 1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, linear));
    }`;

  // The canvas on screen. Its first row is the top of the picture
  const DISPLAY_FRAGMENT = `#version 300 es
    precision highp float;
    precision highp sampler2D;
    ${TO_SRGB}
    in vec2 uv;
    out vec4 color;
    void main() {
      vec2 p = vec2(uv.x, 1.0 - uv.y);
      color = vec4(toSRGB(texture(below0, p), texture(below1, p), texture(below2, p), texture(below3, p)), 1.0);
    }`;

  // The canvas for reading back: averaged over blocks of shrink x shrink pixels, first row first
  const READ_FRAGMENT = `#version 300 es
    precision highp float;
    precision highp sampler2D;
    ${TO_SRGB}
    uniform int shrink;
    out vec4 color;
    void main() {
      ivec2 start = ivec2(gl_FragCoord.xy) * shrink;
      vec4 R0 = vec4(0.0), R1 = vec4(0.0), R2 = vec4(0.0), R3 = vec4(0.0);
      for (int y = 0; y < shrink; y++) {
        for (int x = 0; x < shrink; x++) {
          ivec2 pixel = start + ivec2(x, y);
          R0 += texelFetch(below0, pixel, 0);
          R1 += texelFetch(below1, pixel, 0);
          R2 += texelFetch(below2, pixel, 0);
          R3 += texelFetch(below3, pixel, 0);
        }
      }
      float n = float(shrink * shrink);
      color = vec4(toSRGB(R0 / n, R1 / n, R2 / n, R3 / n), 1.0);
    }`;

  // ---------------- Canvas ----------------

  // Floats per stroke in the instance buffer, and the most strokes drawn at once
  const STROKE_FLOATS = 24;
  const BATCH_SIZE = 1024;
  // Strokes in one batch must not overlap. They are checked on a grid of cells this many pixels wide
  const CELL = 16;

  class PaintCanvas {
    // The canvas on the graphics card: for every pixel its reflectance spectrum (below, what is shown) and how much
    // of every paint is on it (amounts, for wet blending). Both are kept twice: strokes are drawn from the canvas
    // into the second copy and the changed pixels copied back, since a shader can't read what it writes.
    // Strokes are collected into batches of strokes that don't overlap, so a batch is the same as painting its
    // strokes one after another

    constructor(element) {
      const gl = element.getContext("webgl2", { alpha: false, antialias: false, depth: false, stencil: false, powerPreference: "high-performance" });
      if (!gl) throw new Error("WebGL 2 is not available");
      this.element = element;
      this.gl = gl;
      this.floats = gl.getExtension("EXT_color_buffer_float") !== null;
      this.lost = false;
      element.addEventListener("webglcontextlost", (event) => {
        event.preventDefault();
        this.lost = true;
      });

      this.strokeProgram = this.makeProgram(STROKE_VERTEX, STROKE_FRAGMENT);
      this.copyProgram = this.makeProgram(STROKE_VERTEX, COPY_FRAGMENT);
      this.displayProgram = this.makeProgram(FULL_VERTEX, DISPLAY_FRAGMENT);
      this.readProgram = this.makeProgram(FULL_VERTEX, READ_FRAGMENT);

      // Paints, and the spectrum -> sRGB matrix
      const paintK = new Float32Array(32 * 4);
      PAINTS.forEach((paint, i) => paintK.set(paint.K, i * SAMPLES));
      const strengths = PAINTS.map((paint) => paint.S);
      this.use(this.strokeProgram, { paintK, strengthA: strengths.slice(0, 4), strengthB: strengths.slice(4, 8) });
      this.bindSamplers(this.strokeProgram, ["below0", "below1", "below2", "below3", "amounts0", "amounts1"]);
      this.bindSamplers(this.copyProgram, ["source0", "source1", "source2", "source3"]);
      const toRGB = (channel) => Float32Array.from(SPECTRUM_TO_RGB, (row) => row[channel]);
      for (const program of [this.displayProgram, this.readProgram]) {
        this.use(program, { toR: toRGB(0), toG: toRGB(1), toB: toRGB(2) });
        this.bindSamplers(program, ["below0", "below1", "below2", "below3"]);
      }

      // A quad's corners, and the strokes of a batch
      this.vertexArray = gl.createVertexArray();
      gl.bindVertexArray(this.vertexArray);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      this.instances = new Float32Array(BATCH_SIZE * STROKE_FLOATS);
      this.instanceBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.instances.byteLength, gl.DYNAMIC_DRAW);
      for (let i = 0; i < 6; i++) {
        gl.enableVertexAttribArray(1 + i);
        gl.vertexAttribPointer(1 + i, 4, gl.FLOAT, false, STROKE_FLOATS * 4, i * 16);
        gl.vertexAttribDivisor(1 + i, 1);
      }
      gl.bindVertexArray(null);

      this.width = 0;
      this.height = 0;
      this.textures = [];
      this.framebuffers = [];
      this.count = 0;
      // Counts the changes, so whoever shows the canvas can tell it changed
      this.version = 0;
    }

    makeProgram(vertexSource, fragmentSource) {
      const gl = this.gl;
      const program = gl.createProgram();
      for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
        gl.attachShader(program, shader);
      }
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
      return { program, locations: {} };
    }

    use(program, uniforms = {}) {
      // Switch to a program and set uniforms: numbers are floats, arrays of 4n are vec4 arrays
      const gl = this.gl;
      gl.useProgram(program.program);
      for (const [name, value] of Object.entries(uniforms)) {
        if (!(name in program.locations)) {
          program.locations[name] = gl.getUniformLocation(program.program, name) ?? gl.getUniformLocation(program.program, name + "[0]");
        }
        const location = program.locations[name];
        if (typeof value === "number") gl.uniform1f(location, value);
        else if (value.length === 2) gl.uniform2fv(location, value);
        else gl.uniform4fv(location, value);
      }
    }

    bindSamplers(program, names) {
      const gl = this.gl;
      gl.useProgram(program.program);
      names.forEach((name, unit) => gl.uniform1i(gl.getUniformLocation(program.program, name), unit));
    }

    makeTexture(width, height, float) {
      const gl = this.gl;
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, float ? gl.RGBA16F : gl.RGBA8, width, height);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return texture;
    }

    makeFramebuffer(textures) {
      const gl = this.gl;
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      textures.forEach((texture, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texture, 0));
      gl.drawBuffers(textures.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
      const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      return complete ? framebuffer : null;
    }

    start(width, height, paper) {
      // A fresh canvas of bare paper. Without float textures the canvas is kept in 8 bits, with paint amounts
      // stored as shares of amountScale
      const gl = this.gl;
      this.release();
      this.width = width;
      this.height = height;
      for (const float of this.floats ? [true, false] : [false]) {
        // below[copy] and amounts[copy] for both copies
        this.below = [0, 1].map(() => [0, 1, 2, 3].map(() => this.makeTexture(width, height, float)));
        this.amounts = [0, 1].map(() => [0, 1].map(() => this.makeTexture(width, height, float)));
        this.belowFramebuffers = this.below.map((textures) => this.makeFramebuffer(textures));
        this.amountFramebuffers = this.amounts.map((textures) => this.makeFramebuffer(textures));
        this.amountScale = float ? 1 : 64;
        if (![...this.belowFramebuffers, ...this.amountFramebuffers].includes(null)) break;
        this.release();
      }
      if (!this.below) throw new Error("the graphics card can't hold a canvas this big");
      this.paper = paper;
      for (const copy of [0, 1]) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.belowFramebuffers[copy]);
        for (let j = 0; j < 4; j++) gl.clearBufferfv(gl.COLOR, j, paper.subarray(4 * j, 4 * j + 4));
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.amountFramebuffers[copy]);
        for (let j = 0; j < 2; j++) gl.clearBufferfv(gl.COLOR, j, [0, 0, 0, 0]);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.use(this.strokeProgram, { canvasSize: [width, height], paper, amountScale: this.amountScale });
      this.use(this.copyProgram, { canvasSize: [width, height] });

      this.gridWidth = Math.ceil(width / CELL);
      this.grid = new Uint32Array(this.gridWidth * Math.ceil(height / CELL));
      this.batch = 1;
      this.count = 0;
      this.version++;
    }

    release() {
      const gl = this.gl;
      for (const texture of [...(this.below || []).flat(), ...(this.amounts || []).flat()]) gl.deleteTexture(texture);
      for (const framebuffer of [...(this.belowFramebuffers || []), ...(this.amountFramebuffers || [])]) gl.deleteFramebuffer(framebuffer);
      if (this.readTarget) {
        gl.deleteTexture(this.readTarget.texture);
        gl.deleteFramebuffer(this.readTarget.framebuffer);
        this.readTarget = null;
      }
      this.below = this.amounts = this.belowFramebuffers = this.amountFramebuffers = null;
    }

    addStroke(x, y, angle, length, width, load, raggedness, raggedPhase, frequencies, phases, wetness, mix, weights) {
      // Queues a stroke, after drawing the batch so far if it overlaps a stroke in it
      const across = 0.66 * width + 1.5, along = length / 2 + across;
      const c = Math.abs(Math.cos(angle)), s = Math.abs(Math.sin(angle));
      const reachX = c * along + s * across, reachY = s * along + c * across;
      const rows = this.grid.length / this.gridWidth;
      const x0 = Math.max(Math.floor((x - reachX) / CELL), 0), x1 = Math.min(Math.floor((x + reachX) / CELL), this.gridWidth - 1);
      const y0 = Math.max(Math.floor((y - reachY) / CELL), 0), y1 = Math.min(Math.floor((y + reachY) / CELL), rows - 1);
      let overlaps = this.count >= BATCH_SIZE;
      for (let row = y0; row <= y1 && !overlaps; row++) {
        for (let column = x0; column <= x1; column++) {
          if (this.grid[row * this.gridWidth + column] === this.batch) {
            overlaps = true;
            break;
          }
        }
      }
      if (overlaps) this.flush();
      for (let row = y0; row <= y1; row++) this.grid.fill(this.batch, row * this.gridWidth + x0, row * this.gridWidth + x1 + 1);

      const data = this.instances, offset = this.count * STROKE_FLOATS;
      data[offset] = x;
      data[offset + 1] = y;
      data[offset + 2] = angle;
      data[offset + 3] = length;
      data[offset + 4] = width;
      data[offset + 5] = load;
      data[offset + 6] = raggedness;
      data[offset + 7] = raggedPhase;
      data.set(frequencies, offset + 8);
      data[offset + 11] = wetness;
      data.set(phases, offset + 12);
      data[offset + 15] = load;
      data.set(weights.subarray(mix * 8, mix * 8 + 8), offset + 16);
      this.count++;
    }

    bindTextures(textures) {
      const gl = this.gl;
      textures.forEach((texture, unit) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, texture);
      });
    }

    flush() {
      // Draws the queued strokes
      if (this.count === 0 || this.lost) return;
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.instances, 0, this.count * STROKE_FLOATS);
      gl.bindVertexArray(this.vertexArray);
      gl.viewport(0, 0, this.width, this.height);

      // From the canvas into the second copy: the new reflectance, then the new paint amounts
      this.use(this.strokeProgram);
      this.bindTextures([...this.below[0], ...this.amounts[0]]);
      for (const [part, framebuffer] of [[0, this.belowFramebuffers[1]], [1, this.amountFramebuffers[1]]]) {
        gl.uniform1i(gl.getUniformLocation(this.strokeProgram.program, "part"), part);
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
      }

      // And the same quads back
      this.use(this.copyProgram);
      this.bindTextures(this.below[1]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.belowFramebuffers[0]);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
      this.bindTextures([...this.amounts[1], ...this.amounts[1]]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.amountFramebuffers[0]);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);

      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.count = 0;
      this.batch++;
      this.version++;
    }

    draw() {
      // Shows the canvas, stretched over the whole element
      if (this.lost || !this.below) return;
      const gl = this.gl;
      this.flush();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.element.width, this.element.height);
      this.use(this.displayProgram);
      this.bindTextures(this.below[0]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    readPixels(shrink = 1) {
      // The canvas as 8 bit RGBA pixels, averaged over blocks of shrink x shrink pixels, top row first
      const gl = this.gl;
      this.flush();
      const width = Math.floor(this.width / shrink), height = Math.floor(this.height / shrink);
      if (!this.readTarget || this.readTarget.width !== width || this.readTarget.height !== height) {
        if (this.readTarget) {
          gl.deleteTexture(this.readTarget.texture);
          gl.deleteFramebuffer(this.readTarget.framebuffer);
        }
        const texture = this.makeTexture(width, height, false);
        this.readTarget = { width, height, texture, framebuffer: this.makeFramebuffer([texture]) };
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.readTarget.framebuffer);
      gl.viewport(0, 0, width, height);
      this.use(this.readProgram);
      gl.uniform1i(gl.getUniformLocation(this.readProgram.program, "shrink"), shrink);
      this.bindTextures(this.below[0]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const pixels = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { width, height, pixels };
    }

    toBlob() {
      // The painting as a PNG file
      const { width, height, pixels } = this.readPixels(1);
      const scratch = document.createElement("canvas");
      scratch.width = width;
      scratch.height = height;
      scratch.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(pixels.buffer), width, height), 0, 0);
      return new Promise((resolve) => scratch.toBlob(resolve, "image/png"));
    }
  }

  // ---------------- Style ----------------

  // Monet's passes in painting order, as tuned in the desktop app: where strokes go, their width (share of the
  // canvas width) and length (in widths), the color difference (threshold, delta E) and missing detail
  // (entropyGap, bits) that call for another stroke, how often a pass looks again (rounds), how much neighbouring
  // strokes vary in color (variation), the impressionist color shift, how well strokes cover (hiding) and how much
  // they blend into the wet paint underneath (wetness)
  const PASSES = [
    { name: "base color masses", placement: "cover", width: 0.02735, length: 1.016, threshold: 0, entropyGap: 0, rounds: 1, variation: 10.06, shift: 0.4, hiding: 1.007, wetness: 0.815 },
    { name: "light and shadow", placement: "needed", width: 0.01666, length: 1.974, threshold: 8, entropyGap: 1.0, rounds: 2, variation: 4, shift: 1.0, hiding: 2.0, wetness: 0.795 },
    { name: "medium forms", placement: "needed", width: 0.01003, length: 3.0, threshold: 5.45, entropyGap: 0.75, rounds: 2, variation: 5, shift: 0.8, hiding: 2.5, wetness: 0.558 },
    { name: "broken color", placement: "broken", width: 0.00676, length: 2.721, threshold: 3, entropyGap: 0.5, rounds: 2, variation: 9, shift: 0.8, hiding: 2.5, wetness: 0.393 },
    { name: "fine structure", placement: "detail", width: 0.00349, length: 2.169, threshold: 2.92, entropyGap: 0.175, rounds: 2, variation: 16.3, shift: 0.8, hiding: 0.712, wetness: 0.146 },
    { name: "highlights and accents", placement: "highlight", width: 0.00341, length: 1.925, threshold: 0, entropyGap: 0, rounds: 1, variation: 4, shift: 1.0, hiding: 6.605, wetness: 0.094 },
  ];
  // Passes that only paint where something is still missing
  const NEEDS = ["needed", "broken", "detail"];

  // Everything else about how he paints: when a pass is done, how much finer strokes get every round, stroke
  // spacing in widths, how clearly the picture must run in one direction before strokes follow it, the window
  // detail is measured in (stroke widths), how shadows and light shift color, and the broken color pass's density
  const GENERAL = {
    doneShare: 0.005, minImprovement: 0.128, roundShrink: 0.854, spacing: 0.623, followStructure: 0.172,
    detailWindow: 2.58, blueShadows: 3.41, violetShadows: 1.66, warmLight: 6.82, saturation: 0.224,
    highlightLift: 7.95, brokenDensity: 0.886,
  };

  // The settings visitors can change, from 0 to 1. 0.5 is the way Monet paints by default
  const STYLE_DEFAULTS = { speed: 0.8, brush: 0.5, detail: 0.5, color: 0.5, wetness: 0.5 };

  class Style {
    // Turns the few settings into the parameters of every pass. Read at the start of every pass and round,
    // so changes apply while Monet is painting

    constructor(values) {
      this.values = values;
      this.general = GENERAL;
    }

    getPassCount() {
      return PASSES.length;
    }

    getDrawSpeed() {
      // Strokes per second, from 15 to 6000, null at the very top: as fast as the computer can
      const speed = this.values.speed;
      return speed >= 0.999 ? null : 15 * 400 ** speed;
    }

    getPass(index) {
      const pass = PASSES[index], v = this.values;
      // From 1/range to range, 1 in the middle
      const scale = (value, range) => range ** ((value - 0.5) * 2);
      const looseness = scale(1 - v.detail, 2.5);
      return {
        ...pass,
        width: pass.width * scale(v.brush, 2),
        threshold: pass.threshold * looseness,
        entropyGap: pass.entropyGap * looseness,
        rounds: Math.max(1, pass.rounds + (v.detail > 0.8 ? 1 : 0) - (v.detail < 0.2 ? 1 : 0)),
        variation: pass.variation * (0.25 + 1.5 * v.color),
        shift: pass.shift * 2 * v.color,
        wetness: v.wetness < 0.5 ? pass.wetness * v.wetness * 2 : pass.wetness + (1 - pass.wetness) * (v.wetness * 2 - 1),
      };
    }
  }

  // ---------------- Auto-painter ----------------

  function loadReference(image, width, height) {
    // The picture scaled to cover the canvas with its middle cropped, in CIELAB. Transparency becomes white
    const scratch = document.createElement("canvas");
    scratch.width = width;
    scratch.height = height;
    const context = scratch.getContext("2d", { willReadFrequently: true });
    context.fillStyle = "#fff";
    context.fillRect(0, 0, width, height);
    context.imageSmoothingQuality = "high";
    const w = image.naturalWidth || image.width, h = image.naturalHeight || image.height;
    const s = Math.max(width / w, height / h);
    context.drawImage(image, (width - w * s) / 2, (height - h * s) / 2, w * s, h * s);
    return rgbaToLab(context.getImageData(0, 0, width, height).data, width * height);
  }

  class AutoPainter {
    // Paints a picture the way Monet might have: from large color masses down to highlights, a few strokes per
    // frame. Every stroke is a mix of the palette's paints laid on the canvas as its own layer.
    //
    // Deciding where another stroke is needed: every pass works at its own scale, blurring picture and canvas to
    // the detail its strokes can paint and measuring local Shannon entropy in windows the size of its strokes.
    // An area needs another stroke where the canvas lacks entropy the picture has there (detail at this scale is
    // missing), or where its color still differs from what Monet aims for. Each pass looks again after every round
    // of strokes, until hardly anything needs paint any more

    constructor(canvas, reference, style, onStroke = null, seed = 1) {
      this.canvas = canvas;
      this.reference = reference;
      this.style = style;
      // Called after every stroke with the share of the canvas it covers and where it is (0 left, 1 right)
      this.onStroke = onStroke;
      this.random = makeRandom(seed);
      this.mixes = getMixes();
      this.passCount = style.getPassCount();
      // Maps the analysis works on have at most about this many values, finer passes compare at lower resolution
      this.smallestShrink = Math.max(1, Math.round(Math.sqrt(canvas.width * canvas.height / 200000)));

      // Shown while painting
      this.stage = "looking at the picture";
      this.passIndex = -1;
      this.progress = 0;
      this.strokes = 0;

      this.paused = false;
      this.finished = false;
      // Strokes the draw speed allows to paint right now
      this.strokeAllowance = 0;
      this.steps = this.run();
    }

    // ---------------- Control ----------------

    step(seconds, frameTime) {
      // Works until the time for this frame is used up, or the draw speed allows no more strokes
      const speed = this.style.getDrawSpeed();
      if (speed !== null) {
        // Strokes are earned as time passes, at most a tenth of a second's worth so a slow frame causes no burst
        this.strokeAllowance = Math.min(this.strokeAllowance + speed * frameTime, Math.max(speed * 0.1, 1));
      }
      const end = performance.now() + seconds * 1000;
      while (!this.paused && !this.finished && performance.now() < end) {
        const { value, done } = this.steps.next();
        if (done) this.finished = true;
        else if (value === WAIT) break;
      }
    }

    *waitForStroke() {
      for (;;) {
        if (this.style.getDrawSpeed() === null) return;
        if (this.strokeAllowance >= 1) {
          this.strokeAllowance -= 1;
          return;
        }
        yield WAIT;
      }
    }

    getOverallProgress() {
      return this.finished ? 1 : (Math.max(this.passIndex, 0) + this.progress) / this.passCount;
    }

    // ---------------- Painting ----------------

    *run() {
      // Generator that paints one stroke per step, and yields in between while looking at picture and canvas
      const W = this.canvas.width, H = this.canvas.height;
      yield WORK;

      for (let index = 0; index < this.passCount; index++) {
        const step = this.style.getPass(index);
        let general = this.style.general;
        this.passIndex = index;
        this.progress = 0;
        this.stage = step.name;

        // The picture at the detail this pass works at, the colors the painter aims for there, and how much detail
        // those have at this scale. Measured on the intended colors, so detail the palette can't mix (like darks
        // darker than any mix) doesn't count as missing. Coarse passes compare canvas and picture at lower
        // resolution: blurred to the detail they paint, the maps look the same there, and it is much faster
        const width = Math.max(step.width * W, 1.5);
        const shrink = Math.max(1, Math.floor(width / 4), this.smallestShrink);
        const blur = width / 2 / shrink;
        const window = Math.max(3, Math.round(width * general.detailWindow / shrink));
        const mw = Math.floor(W / shrink), mh = Math.floor(H / shrink);
        const reference = this.reference.map((channel) => gaussianBlur(shrinkMap(channel, W, H, shrink), mw, mh, blur));
        yield WORK;
        const intended = yield* this.getIntendedColors(reference, step.shift);
        const usesEntropy = NEEDS.includes(step.placement);
        const imageEntropy = usesEntropy ? yield* getLocalEntropy(intended, mw, mh, window) : null;
        const directions = this.getStrokeDirections(reference[0], mw, mh, width / shrink);
        yield WORK;

        let previousShare = 1;
        for (let attempt = 0; attempt < step.rounds; attempt++) {
          general = this.style.general;
          this.stage = step.rounds === 1 ? step.name : `${step.name}, round ${attempt + 1}`;

          // Compare what is painted so far with the picture at the same scale: the colors the painter aims for,
          // and the detail (local entropy) of the picture
          const read = this.canvas.readPixels(shrink);
          const canvas = rgbaToLab(read.pixels, mw * mh);
          const difference = new Float32Array(mw * mh);
          for (let i = 0; i < difference.length; i++) {
            difference[i] = Math.hypot(canvas[0][i] - intended[0][i], canvas[1][i] - intended[1][i], canvas[2][i] - intended[2][i]);
          }
          const error = gaussianBlur(difference, mw, mh, blur);
          let entropyGap = null;
          if (usesEntropy) {
            const blurred = canvas.map((channel) => gaussianBlur(channel, mw, mh, blur));
            const canvasEntropy = yield* getLocalEntropy(blurred, mw, mh, window);
            entropyGap = imageEntropy.map((value, i) => value - canvasEntropy[i]);
          }

          // Stop looking again once hardly anything needs paint, or when even finer strokes didn't help much
          const keep = this.getStrokePlaces(step, width, shrink, mw, mh, error, entropyGap, imageEntropy);
          const share = keep.reduce((sum, value) => sum + value, 0) / keep.length;
          if ((attempt > 0 && share < general.doneShare) || (attempt > 1 && share > previousShare * (1 - general.minImprovement))) break;
          previousShare = share;
          yield WORK;

          // Every further round uses finer strokes for the detail the last one couldn't paint
          const size = width * general.roundShrink ** attempt;
          const positions = this.getPositions(size * general.spacing, keep, shrink, mw, mh);
          const count = positions.length / 2;
          for (let i = 0; i < count; i++) {
            yield* this.waitForStroke();
            const step = this.style.getPass(index);
            this.progress = (attempt + i / count) / step.rounds;
            const x = positions[2 * i], y = positions[2 * i + 1];
            const cell = Math.min(Math.floor(y / shrink), mh - 1) * mw + Math.min(Math.floor(x / shrink), mw - 1);
            const angle = this.getStrokeAngle(directions.angles[cell], directions.coherence[cell], directions.flat[cell]);

            let target = reference.map((channel) => sampleMap(channel, mw, mh, x / shrink - 0.5, y / shrink - 0.5));
            if (step.placement === "highlight") target[0] = Math.min(target[0] + general.highlightLift, 100);
            target = this.shiftColor(target, step.variation, step.shift);

            const strokeWidth = size * this.random.uniform(0.8, 1.2);
            const strokeLength = strokeWidth * step.length * this.random.uniform(0.7, 1.3);
            this.paintStroke(x, y, strokeLength, strokeWidth, angle, target, step.hiding, step.wetness);
            yield STROKE;
          }
        }
      }
      this.progress = 1;
      this.canvas.flush();
    }

    getStrokePlaces(step, width, shrink, mw, mh, error, entropyGap, imageEntropy) {
      // Map of where this pass puts strokes, shrunk like the analysis
      const n = mw * mh, keep = new Uint8Array(n);
      if (step.placement === "cover") return keep.fill(1);
      if (step.placement === "highlight") {
        // Highlights on the brightest spots that stand out from their surroundings
        const lightness = shrinkMap(this.reference[0], this.canvas.width, this.canvas.height, shrink);
        const surroundings = gaussianBlur(lightness, mw, mh, width * 4 / shrink);
        const bright = percentile(lightness, 90);
        for (let i = 0; i < n; i++) {
          keep[i] = lightness[i] > bright && lightness[i] - surroundings[i] > 3 && this.random.random() < 0.5 ? 1 : 0;
        }
        return keep;
      }
      // Detail at this scale is missing, or the color is still off
      for (let i = 0; i < n; i++) {
        let needed = entropyGap[i] > step.entropyGap || error[i] > step.threshold;
        if (step.placement === "broken") needed = needed && this.random.random() < this.style.general.brokenDensity;
        // Small strokes only where the picture has detail at this scale, everything else stays simplified
        if (step.placement === "detail") needed = needed && imageEntropy[i] > step.entropyGap;
        keep[i] = needed ? 1 : 0;
      }
      return keep;
    }

    getStrokeAngle(structureAngle, coherence, flat) {
      // Follow the structure where there is one, sky and water get horizontal strokes, busy areas any direction
      const random = this.random;
      let angle;
      if (coherence > this.style.general.followStructure) angle = structureAngle;
      else if (flat) angle = random.normal(0, 0.25);
      else angle = random.uniform(0, Math.PI);
      angle += random.normal(0, 0.12);
      // Strokes are painted in either direction
      if (random.random() < 0.5) angle += Math.PI;
      return angle;
    }

    paintStroke(x, y, length, width, angle, target, hiding, wetness) {
      // One stroke in the mix of palette paints closest to the target color (CIELAB), laid on thick enough to
      // hide what is underneath to the given degree
      const random = this.random, mixes = this.mixes;
      const mix = mixes.nearest(target[0], target[1], target[2]);
      const load = hiding / Math.max(mixes.S[mix], 1e-3);
      const raggedness = random.uniform(0.5, 1.5), raggedPhase = random.uniform(0, 2 * Math.PI);
      const frequencies = [0, 1, 2].map(() => random.uniform(3, 9) * 2 * Math.PI / Math.max(width, 1));
      const phases = [0, 1, 2].map(() => random.uniform(0, 2 * Math.PI));
      this.canvas.addStroke(x, y, angle, length, width, load, raggedness, raggedPhase, frequencies, phases, wetness, mix, mixes.weights);
      this.strokes++;
      if (this.onStroke) {
        const W = this.canvas.width, H = this.canvas.height;
        this.onStroke(width * (length + width * 0.785) / (W * H), x / W);
      }
    }

    // ---------------- Analysis ----------------

    *getIntendedColors(reference, shift) {
      // The colors the painter aims for everywhere: the picture with the impressionist color shift of shiftColor
      // (without its random variation), as close as the palette can mix them
      const general = this.style.general, mixes = this.mixes;
      const [L, A, B] = reference, n = L.length;
      const intended = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
      const boost = 1 + general.saturation * shift;
      for (let i = 0; i < n; i++) {
        const shadow = Math.min(Math.max((45 - L[i]) / 45, 0), 1) * shift;
        const light = Math.min(Math.max((L[i] - 65) / 35, 0), 1) * shift;
        const a = (A[i] + general.violetShadows * shadow) * boost;
        const b = (B[i] - general.blueShadows * shadow + general.warmLight * light) * boost;
        const mix = mixes.nearestRounded(L[i], a, b);
        for (let axis = 0; axis < 3; axis++) intended[axis][i] = mixes.lab[mix * 3 + axis];
        if ((i & 16383) === 16383) yield WORK;
      }
      return intended;
    }

    getStrokeDirections(lightness, width, height, sigma) {
      // Structure tensor: strokes follow edges and forms, at right angles to the brightness gradient
      const n = width * height;
      const gxx = new Float32Array(n), gxy = new Float32Array(n), gyy = new Float32Array(n);
      // Central differences, one-sided at the edges
      const difference = (i, position, size, stride) => {
        if (size < 2) return 0;
        if (position === 0) return lightness[i + stride] - lightness[i];
        if (position === size - 1) return lightness[i] - lightness[i - stride];
        return (lightness[i + stride] - lightness[i - stride]) / 2;
      };
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = y * width + x;
          const gx = difference(i, x, width, 1), gy = difference(i, y, height, width);
          gxx[i] = gx * gx;
          gxy[i] = gx * gy;
          gyy[i] = gy * gy;
        }
      }
      const Jxx = gaussianBlur(gxx, width, height, sigma);
      const Jxy = gaussianBlur(gxy, width, height, sigma);
      const Jyy = gaussianBlur(gyy, width, height, sigma);
      const angles = new Float32Array(n), coherence = new Float32Array(n), energy = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        angles[i] = 0.5 * Math.atan2(2 * Jxy[i], Jxx[i] - Jyy[i]) + Math.PI / 2;
        energy[i] = Jxx[i] + Jyy[i];
        coherence[i] = Math.sqrt((Jxx[i] - Jyy[i]) ** 2 + 4 * Jxy[i] ** 2) / (energy[i] + 1e-6);
      }
      const calm = percentile(energy, 40);
      return { angles, coherence, flat: energy.map((value) => (value < calm ? 1 : 0)) };
    }

    getPositions(spacing, keep, shrink, mw, mh) {
      // Jittered grid over the canvas, only where keep (shrunk by shrink) is set, in random order.
      // Returns x and y one after another
      const W = this.canvas.width, H = this.canvas.height, random = this.random;
      const columns = Math.ceil((W - spacing / 2) / spacing), rows = Math.ceil((H - spacing / 2) / spacing);
      const kept = [];
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const x = Math.min(Math.max(spacing / 2 + column * spacing + random.uniform(-0.5, 0.5) * spacing, 0), W - 1);
          const y = Math.min(Math.max(spacing / 2 + row * spacing + random.uniform(-0.5, 0.5) * spacing, 0), H - 1);
          if (keep[Math.min(Math.floor(y / shrink), mh - 1) * mw + Math.min(Math.floor(x / shrink), mw - 1)]) kept.push(x, y);
        }
      }
      // Shuffle the pairs
      const count = kept.length / 2, positions = new Float32Array(kept);
      for (let i = count - 1; i > 0; i--) {
        const j = Math.floor(random.random() * (i + 1));
        for (const k of [0, 1]) [positions[2 * i + k], positions[2 * j + k]] = [positions[2 * j + k], positions[2 * i + k]];
      }
      return positions;
    }

    shiftColor([L, a, b], variation, shift) {
      const general = this.style.general, random = this.random;
      // Shadows lean towards blue and violet instead of only getting darker, sunlit areas towards warm yellow
      const shadow = Math.min(Math.max((45 - L) / 45, 0), 1) * shift;
      const light = Math.min(Math.max((L - 65) / 35, 0), 1) * shift;
      a += general.violetShadows * shadow;
      b += -general.blueShadows * shadow + general.warmLight * light;

      // Slightly more saturated, then broken color: neighbouring strokes keep close values but differ in hue and chroma
      const chroma = Math.hypot(a, b) * (1 + general.saturation * shift) * Math.max(1 + random.normal(0, variation / 30), 0);
      const hue = Math.atan2(b, a) + random.normal(0, variation / 45);
      L += random.normal(0, variation * 0.25);
      return [L, chroma * Math.cos(hue), chroma * Math.sin(hue)];
    }
  }

  // ---------------- Sound ----------------

  class Splats {
    // A cartoon splat when paint goes on, deeper and louder for big strokes and panned to where the stroke lands.
    // At most maxPerSecond are played, so Monet painting fast becomes a splattering patter instead of a roar

    constructor(url, isOn) {
      this.url = url;
      this.isOn = isOn;
      this.maxPerSecond = 12;
      // Playback speeds, from deep splats for the biggest strokes to little ones for the finest
      this.pitches = [0.75, 0.85, 0.95, 1.05, 1.2, 1.35];
      this.volume = 0.5;
      this.context = null;
      this.buffer = null;
      this.lastPlayed = 0;
      // Its own random numbers, so sounds never change what gets painted
      this.random = makeRandom(7);
    }

    unlock() {
      // Browsers only allow sound once the visitor did something on the page
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      if (!this.context) {
        this.context = new AudioContext();
        this.output = this.context.createGain();
        this.output.gain.value = this.volume;
        this.output.connect(this.context.destination);
        fetch(this.url)
          .then((response) => response.arrayBuffer())
          .then((data) => this.context.decodeAudioData(data))
          .then((buffer) => {
            // Played from just before the splat gets loud
            const samples = buffer.getChannelData(0);
            const loud = samples.findIndex((sample) => Math.abs(sample) > 0.02);
            this.start = Math.max(loud - 40, 0) / buffer.sampleRate;
            this.buffer = buffer;
          })
          .catch(() => {});
      }
      if (this.context.state === "suspended") this.context.resume();
    }

    getStream() {
      // The splats as a stream, for recording them into a video
      if (!this.context) return null;
      if (!this.recording) {
        this.recording = this.context.createMediaStreamDestination();
        this.output.connect(this.recording);
      }
      return this.recording.stream;
    }

    play(share, pan) {
      // A stroke covering this share of the canvas, pan from 0 (left) to 1 (right). Skipped when splats come too fast
      if (!this.buffer || !this.isOn() || this.context.state !== "running") return;
      const now = this.context.currentTime;
      if (now - this.lastPlayed < 1 / this.maxPerSecond) return;
      this.lastPlayed = now;

      const size = Math.min(Math.sqrt(share / 0.01), 1);
      const last = this.pitches.length - 1;
      const index = Math.min(Math.max(Math.round((1 - size) * last + this.random.uniform(-0.6, 0.6)), 0), last);
      const source = this.context.createBufferSource();
      source.buffer = this.buffer;
      source.playbackRate.value = this.pitches[index];
      const gain = this.context.createGain();
      gain.gain.value = 0.35 + 0.65 * size;
      source.connect(gain);
      if (this.context.createStereoPanner) {
        const panner = this.context.createStereoPanner();
        panner.pan.value = Math.min(Math.max(pan * 2 - 1, -1), 1);
        gain.connect(panner);
        panner.connect(this.output);
      } else {
        gain.connect(this.output);
      }
      source.start(now, this.start);
    }
  }

  return {
    PaintCanvas, AutoPainter, Style, Splats, STYLE_DEFAULTS, PAPERS, PALETTE,
    loadReference, getMixes, getPaperReflectance,
    getPaintColor: (index) => spectrumToCss(getOpaqueReflectance(PAINTS[index].K, PAINTS[index].S)),
    getPaperColor: (paper) => spectrumToCss(getPaperReflectance(paper)),
  };
})();
