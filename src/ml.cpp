#include "ml.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

namespace ml {

// 12 roughly log-spaced bands over FFT bins 1..63 (bin 0 = gravity/DC is skipped).
// At 250 Hz each bin is ~1.95 Hz wide, so the bands cover ~2 Hz to ~125 Hz.
const uint8_t BAND_EDGES[N_BANDS + 1] = {1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48, 64};

static constexpr float PI_F = 3.14159265358979f;
static constexpr float STD_FLOOR = 0.05f;   // log10 units (~12%): stops near-constant features exploding

// ------------------------------------------------------------------ FFT
static float hann[WIN];
static bool hannReady = false;

static void fft(float* re, float* im, int n) {
  // bit-reversal permutation
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  // butterflies
  for (int len = 2; len <= n; len <<= 1) {
    const float ang = -2.0f * PI_F / len;
    const float wr = cosf(ang), wi = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cr = 1.0f, ci = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        const int a = i + k, b = i + k + len / 2;
        const float tr = re[b] * cr - im[b] * ci;
        const float ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;        im[a] += ti;
        const float ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

void computeFeatures(const float* x, const float* y, const float* z, float* feat) {
  if (!hannReady) {
    for (int i = 0; i < WIN; i++) hann[i] = 0.5f - 0.5f * cosf(2.0f * PI_F * i / (WIN - 1));
    hannReady = true;
  }
  const float* axes[3] = {x, y, z};
  float mean[3];
  float spec[WIN / 2];
  memset(spec, 0, sizeof(spec));
  float re[WIN], im[WIN];
  float rms2Sum = 0;

  for (int a = 0; a < 3; a++) {
    const float* v = axes[a];
    float m = 0;
    for (int i = 0; i < WIN; i++) m += v[i];
    m /= WIN;
    mean[a] = m;
    float s2 = 0;
    for (int i = 0; i < WIN; i++) {
      const float d = v[i] - m;        // remove gravity / mounting offset
      s2 += d * d;
      re[i] = d * hann[i];
      im[i] = 0;
    }
    const float rms2 = s2 / WIN;
    rms2Sum += rms2;
    feat[a] = log10f(sqrtf(rms2) + 1e-5f);
    fft(re, im, WIN);
    for (int k = 1; k < WIN / 2; k++) spec[k] += re[k] * re[k] + im[k] * im[k];
  }

  // crest factor of the 3-axis vibration magnitude: high = sharp knocks / impacts
  float peak2 = 0;
  for (int i = 0; i < WIN; i++) {
    const float dx = x[i] - mean[0], dy = y[i] - mean[1], dz = z[i] - mean[2];
    const float m2 = dx * dx + dy * dy + dz * dz;
    if (m2 > peak2) peak2 = m2;
  }
  const float crest = sqrtf(peak2) / (sqrtf(rms2Sum) + 1e-6f);
  feat[3] = log10f(crest + 1e-3f);

  const float norm = 1.0f / ((float)WIN * (float)WIN);
  for (int b = 0; b < N_BANDS; b++) {
    float e = 0;
    for (int k = BAND_EDGES[b]; k < BAND_EDGES[b + 1]; k++) e += spec[k];
    feat[4 + b] = log10f(e * norm + 1e-10f);
  }
}

// ------------------------------------------------------------------ autoencoder
struct Acts { float h1[H1], h2[H2], h3[H1], y[N_FEAT]; };

static void forward(const Params& p, const float* x, Acts& a) {
  for (int j = 0; j < H1; j++) {
    float s = p.b1[j];
    for (int i = 0; i < N_FEAT; i++) s += p.W1[j * N_FEAT + i] * x[i];
    a.h1[j] = tanhf(s);
  }
  for (int k = 0; k < H2; k++) {
    float s = p.b2[k];
    for (int j = 0; j < H1; j++) s += p.W2[k * H1 + j] * a.h1[j];
    a.h2[k] = tanhf(s);
  }
  for (int j = 0; j < H1; j++) {
    float s = p.b3[j];
    for (int k = 0; k < H2; k++) s += p.W3[j * H2 + k] * a.h2[k];
    a.h3[j] = tanhf(s);
  }
  for (int i = 0; i < N_FEAT; i++) {
    float s = p.b4[i];
    for (int j = 0; j < H1; j++) s += p.W4[i * H1 + j] * a.h3[j];
    a.y[i] = s;                                   // linear output
  }
}

static float mse(const float* x, const float* y) {
  float s = 0;
  for (int i = 0; i < N_FEAT; i++) { const float d = y[i] - x[i]; s += d * d; }
  return s / N_FEAT;
}

// Adds the gradient of mse(x, y) for one sample into g.
static void backward(const Params& p, const float* x, const Acts& a, Params& g) {
  float dy[N_FEAT], d3[H1], d2[H2], d1[H1];
  for (int i = 0; i < N_FEAT; i++) {
    dy[i] = 2.0f * (a.y[i] - x[i]) / N_FEAT;
    g.b4[i] += dy[i];
    for (int j = 0; j < H1; j++) g.W4[i * H1 + j] += dy[i] * a.h3[j];
  }
  for (int j = 0; j < H1; j++) {
    float s = 0;
    for (int i = 0; i < N_FEAT; i++) s += p.W4[i * H1 + j] * dy[i];
    d3[j] = s * (1.0f - a.h3[j] * a.h3[j]);
    g.b3[j] += d3[j];
    for (int k = 0; k < H2; k++) g.W3[j * H2 + k] += d3[j] * a.h2[k];
  }
  for (int k = 0; k < H2; k++) {
    float s = 0;
    for (int j = 0; j < H1; j++) s += p.W3[j * H2 + k] * d3[j];
    d2[k] = s * (1.0f - a.h2[k] * a.h2[k]);
    g.b2[k] += d2[k];
    for (int j = 0; j < H1; j++) g.W2[k * H1 + j] += d2[k] * a.h1[j];
  }
  for (int j = 0; j < H1; j++) {
    float s = 0;
    for (int k = 0; k < H2; k++) s += p.W2[k * H1 + j] * d2[k];
    d1[j] = s * (1.0f - a.h1[j] * a.h1[j]);
    g.b1[j] += d1[j];
    for (int i = 0; i < N_FEAT; i++) g.W1[j * N_FEAT + i] += d1[j] * x[i];
  }
}

static uint32_t rngState = 0x12345678u;
static uint32_t rnd() {               // xorshift32
  rngState ^= rngState << 13;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5;
  return rngState;
}
static float rndUniform(float lim) {  // uniform in [-lim, lim]
  return ((rnd() >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * lim;
}
static void initLayer(float* w, int fanOut, int fanIn) {
  const float lim = sqrtf(6.0f / (fanIn + fanOut));   // Xavier/Glorot
  for (int i = 0; i < fanOut * fanIn; i++) w[i] = rndUniform(lim);
}

static void normalise(const Model& m, const float* feat, float* xn) {
  for (int i = 0; i < N_FEAT; i++) xn[i] = (feat[i] - m.mean[i]) / m.stdev[i];
}

static Params grad, adamM, adamV;    // static: keeps them off the task stack

bool train(Model& m, const float* feats, int n, int epochs, ProgressFn cb) {
  if (n < 20) return false;

  // 1. normalisation from all normal windows
  for (int i = 0; i < N_FEAT; i++) {
    float s = 0;
    for (int r = 0; r < n; r++) s += feats[r * N_FEAT + i];
    const float mu = s / n;
    float v = 0;
    for (int r = 0; r < n; r++) { const float d = feats[r * N_FEAT + i] - mu; v += d * d; }
    float sd = sqrtf(v / n);
    if (sd < STD_FLOOR) sd = STD_FLOOR;
    m.mean[i] = mu;
    m.stdev[i] = sd;
  }

  float* xs = (float*)malloc(sizeof(float) * n * N_FEAT);
  int* trainIdx = (int*)malloc(sizeof(int) * n);
  int* valIdx = (int*)malloc(sizeof(int) * n);
  if (!xs || !trainIdx || !valIdx) { free(xs); free(trainIdx); free(valIdx); return false; }
  for (int r = 0; r < n; r++) normalise(m, &feats[r * N_FEAT], &xs[r * N_FEAT]);

  // 2. every 4th window is held back to set the alarm limit fairly
  int nT = 0, nV = 0;
  for (int r = 0; r < n; r++) {
    if (r % 4 == 3) valIdx[nV++] = r; else trainIdx[nT++] = r;
  }

  // 3. initialise weights
  rngState = 0x12345678u ^ (uint32_t)n;
  Params& p = m.p;
  memset(&p, 0, sizeof(p));
  initLayer(p.W1, H1, N_FEAT);
  initLayer(p.W2, H2, H1);
  initLayer(p.W3, H1, H2);
  initLayer(p.W4, N_FEAT, H1);
  memset(&adamM, 0, sizeof(adamM));
  memset(&adamV, 0, sizeof(adamV));

  // 4. mini-batch Adam
  const float lr = 0.005f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-7f;
  const int batch = 8;
  float* P = (float*)&p;
  float* G = (float*)&grad;
  float* Mm = (float*)&adamM;
  float* Vv = (float*)&adamV;
  int step = 0;
  float b1t = 1.0f, b2t = 1.0f;
  Acts acts;

  for (int ep = 0; ep < epochs; ep++) {
    for (int i = nT - 1; i > 0; i--) {            // shuffle
      const int j = rnd() % (i + 1);
      const int t = trainIdx[i]; trainIdx[i] = trainIdx[j]; trainIdx[j] = t;
    }
    float lossSum = 0;
    for (int start = 0; start < nT; start += batch) {
      const int end = (start + batch < nT) ? start + batch : nT;
      memset(&grad, 0, sizeof(grad));
      for (int b = start; b < end; b++) {
        const float* x = &xs[trainIdx[b] * N_FEAT];
        forward(p, x, acts);
        lossSum += mse(x, acts.y);
        backward(p, x, acts, grad);
      }
      const float inv = 1.0f / (end - start);
      step++;
      b1t *= beta1;
      b2t *= beta2;
      for (int k = 0; k < N_PARAMS; k++) {
        const float gk = G[k] * inv;
        Mm[k] = beta1 * Mm[k] + (1 - beta1) * gk;
        Vv[k] = beta2 * Vv[k] + (1 - beta2) * gk * gk;
        const float mh = Mm[k] / (1 - b1t);
        const float vh = Vv[k] / (1 - b2t);
        P[k] -= lr * mh / (sqrtf(vh) + eps);
      }
    }
    if (cb) cb(ep + 1, epochs, lossSum / nT);
  }

  // 5. error statistics on the held-back normal windows
  float s = 0, s2 = 0, mx = 0;
  for (int v = 0; v < nV; v++) {
    const float* x = &xs[valIdx[v] * N_FEAT];
    forward(p, x, acts);
    const float e = mse(x, acts.y);
    s += e; s2 += e * e;
    if (e > mx) mx = e;
  }
  const float mu = s / nV;
  const float var = s2 / nV - mu * mu;
  m.errMean = mu;
  m.errStd = var > 0 ? sqrtf(var) : 0;
  m.errMax = mx;
  m.nTrain = (uint16_t)nT;
  m.nVal = (uint16_t)nV;
  m.magic = MODEL_MAGIC;
  m.version = MODEL_VERSION;

  free(xs); free(trainIdx); free(valIdx);
  return true;
}

float score(const Model& m, const float* feat, float* zOut, float* perFeatErr) {
  float xn[N_FEAT];
  normalise(m, feat, xn);
  Acts a;
  forward(m.p, xn, a);
  if (zOut) memcpy(zOut, xn, sizeof(xn));
  if (perFeatErr) {
    for (int i = 0; i < N_FEAT; i++) { const float d = a.y[i] - xn[i]; perFeatErr[i] = d * d; }
  }
  return mse(xn, a.y);
}

float threshold(const Model& m, int sensitivity) {
  static const float K[3] = {6.0f, 4.0f, 2.5f};   // standard deviations above the mean error
  static const float F[3] = {1.5f, 1.2f, 1.0f};   // ...and at least this times the worst normal error
  if (sensitivity < 0) sensitivity = 0;
  if (sensitivity > 2) sensitivity = 2;
  float t1 = m.errMean + K[sensitivity] * m.errStd;
  float t2 = m.errMax * F[sensitivity];
  float t = t1 > t2 ? t1 : t2;
  return t > 1e-4f ? t : 1e-4f;
}

}  // namespace ml
