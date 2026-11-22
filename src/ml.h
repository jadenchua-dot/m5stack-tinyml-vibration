// Tiny on-device ML for vibration anomaly detection.
//
// 1. Features: every 128 accelerometer samples (~0.5 s at 250 Hz) become 16 numbers:
//    vibration level on X/Y/Z, "spikiness" (crest factor), and the energy in
//    12 frequency bands from an FFT.
// 2. Model: a small autoencoder (16 -> 8 -> 4 -> 8 -> 16) is trained on the device
//    using only "normal" vibration. It learns to squeeze normal windows through a
//    4-number bottleneck and rebuild them.
// 3. Anomaly score: how badly a new window is rebuilt (reconstruction error),
//    divided by a limit worked out from held-back normal windows.
#pragma once
#include <stdint.h>

namespace ml {

constexpr int WIN = 128;               // samples per window
constexpr int HOP = 64;                // a new window every 64 samples (50% overlap)
constexpr int N_BANDS = 12;
constexpr int N_FEAT = 4 + N_BANDS;    // 16 features
constexpr int H1 = 8;
constexpr int H2 = 4;

// FFT bin index where each band starts (13 edges -> 12 bands). Bin k = k * fs / WIN Hz.
extern const uint8_t BAND_EDGES[N_BANDS + 1];

// Fills feat[N_FEAT] from one window of accelerometer readings (in g).
void computeFeatures(const float* x, const float* y, const float* z, float* feat);

struct Params {
  float W1[H1 * N_FEAT]; float b1[H1];
  float W2[H2 * H1];     float b2[H2];
  float W3[H1 * H2];     float b3[H1];
  float W4[N_FEAT * H1]; float b4[N_FEAT];
};
constexpr int N_PARAMS = sizeof(Params) / sizeof(float);

constexpr uint32_t MODEL_MAGIC = 0x31424956;  // "VIB1"
constexpr uint32_t MODEL_VERSION = 1;

struct Model {
  uint32_t magic;
  uint32_t version;
  float mean[N_FEAT];      // feature normalisation learned from normal data
  float stdev[N_FEAT];
  Params p;
  float errMean, errStd, errMax;   // reconstruction error on held-back normal windows
  uint16_t nTrain, nVal;
  float sampleHz;
};

typedef void (*ProgressFn)(int epoch, int epochs, float loss);

// Trains on n raw feature vectors (n * N_FEAT floats). Returns false if n is too small.
bool train(Model& m, const float* feats, int n, int epochs, ProgressFn cb);

// Normalises raw features, runs the autoencoder and returns the mean squared
// reconstruction error. perFeatErr (optional) gets each feature's squared error.
float score(const Model& m, const float* feat, float* zOut, float* perFeatErr);

// Alarm limit for a sensitivity level (0 = LOW, 1 = MED, 2 = HIGH).
float threshold(const Model& m, int sensitivity);

}  // namespace ml
