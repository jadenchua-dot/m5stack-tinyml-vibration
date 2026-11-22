// M5Stack Fire - TinyML vibration anomaly detector (fully offline).
//
// The Fire sits on a machine (fan, pump, motor, washing machine...). You press
// LEARN while the machine runs normally; the Fire records 30 s of vibration and
// trains a small autoencoder neural network on the device itself. From then on
// it scores every half-second of vibration: if the pattern no longer looks like
// "normal", the screen and LED bars turn red and it beeps. No Wi-Fi, no cloud.
//
// Buttons:  A = LEARN normal vibration (press again to cancel)
//           B = mute / unmute the beeper
//           C = sensitivity LOW / MED / HIGH
// Serial (115200): l = learn, m = mute, s = sensitivity, i = model info

#include <M5Unified.h>
#include <Preferences.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>
#include "config.h"
#include "ml.h"

// ---------------------------------------------------------------- hardware
static constexpr int LED_PIN = 15;     // M5Stack Fire side LED bars
static constexpr int LED_COUNT = 10;
Adafruit_NeoPixel leds(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
M5Canvas canvas(&M5.Display);
Preferences prefs;

// ---------------------------------------------------------------- sampling task
struct Window {
  float x[ml::WIN], y[ml::WIN], z[ml::WIN];
  uint32_t endMs;
};
static QueueHandle_t winQueue;
static Window txWin;                  // used only by the sampling task
static Window rxWin;                  // used only by loop()
static volatile uint32_t sampleCounter = 0;

// Polls the IMU every 1 ms; the MPU6886 itself produces a new reading at 250 Hz,
// so samples are evenly spaced by the sensor's own clock.
static void samplingTask(void*) {
  static float rx[ml::WIN], ry[ml::WIN], rz[ml::WIN];
  int head = 0, filled = 0, sinceEmit = 0;
  m5::imu_data_t d;
  for (;;) {
    vTaskDelay(1);
    if (!(M5.Imu.update() & m5::IMU_Class::sensor_mask_accel)) continue;
    M5.Imu.getImuData(&d);
    rx[head] = d.accel.x; ry[head] = d.accel.y; rz[head] = d.accel.z;
    head = (head + 1) % ml::WIN;
    sampleCounter++;
    if (filled < ml::WIN) filled++;
    if (++sinceEmit >= ml::HOP && filled == ml::WIN) {
      sinceEmit = 0;
      for (int i = 0; i < ml::WIN; i++) {     // oldest sample first
        const int k = (head + i) % ml::WIN;
        txWin.x[i] = rx[k]; txWin.y[i] = ry[k]; txWin.z[i] = rz[k];
      }
      txWin.endMs = millis();
      xQueueSend(winQueue, &txWin, 0);       // drop if loop() is busy
    }
  }
}

// ---------------------------------------------------------------- state
enum State { ST_NOMODEL, ST_COUNTDOWN, ST_LEARNING, ST_TRAINING, ST_MONITOR };
State state = ST_NOMODEL;

static ml::Model model;
bool hasModel = false;
int sensitivity = DEFAULT_SENSITIVITY;
bool muted = false;
const char* SENS_NAMES[3] = {"LOW", "MED", "HIGH"};

float sampleHz = 0;
uint32_t stateSince = 0;              // millis() when the current state began
uint32_t quietUntil = 0;              // ignore windows touched by our own beeps / button presses

// learning buffer
static constexpr int MAX_LEARN = (LEARN_SECONDS * 260) / ml::HOP + 8;
static float learnFeats[MAX_LEARN * ml::N_FEAT];
int learnCount = 0;
int trainEpoch = 0;
float trainLoss = 0;

// live values
float feat[ml::N_FEAT];
float zLive[ml::N_FEAT];
float featErr[ml::N_FEAT];
bool haveFeat = false;
float rawScore = 0, smoothScore = 0;
int overCount = 0, underCount = 0;
bool alarmOn = false, warnOn = false;
int topFeature = -1;
uint32_t lastBeep = 0;
uint32_t anomalyCount = 0;

static constexpr int HIST = 140;
float hist[HIST];
int histHead = 0, histLen = 0;

// ---------------------------------------------------------------- helpers
static void markQuiet(uint32_t ms) {
  const uint32_t until = millis() + ms;
  if ((int32_t)(until - quietUntil) > 0) quietUntil = until;
}

static void beep(int freq, int ms) {
  if (muted) return;
  M5.Speaker.tone(freq, ms);
  markQuiet(ms + 600);                 // the speaker shakes the case too
}

static float bandHz(int edgeIndex) {
  const float fs = sampleHz > 50 ? sampleHz : 250.0f;
  return ml::BAND_EDGES[edgeIndex] * fs / ml::WIN;
}

static void featureName(int i, char* out, size_t n) {
  if (i == 0) snprintf(out, n, "X-axis vibration");
  else if (i == 1) snprintf(out, n, "Y-axis vibration");
  else if (i == 2) snprintf(out, n, "Z-axis vibration");
  else if (i == 3) snprintf(out, n, "knocks/spikes");
  else {
    const int b = i - 4;
    snprintf(out, n, "%.0f-%.0f Hz band", bandHz(b), bandHz(b + 1));
  }
}

static void saveModel() {
  prefs.putBytes("model", &model, sizeof(model));
}

static void loadSettings() {
  sensitivity = prefs.getInt("sens", DEFAULT_SENSITIVITY);
  if (sensitivity < 0 || sensitivity > 2) sensitivity = DEFAULT_SENSITIVITY;
  muted = prefs.getBool("mute", false);
  hasModel = false;
  if (prefs.isKey("model") && prefs.getBytesLength("model") == sizeof(model)) {
    prefs.getBytes("model", &model, sizeof(model));
    hasModel = (model.magic == ml::MODEL_MAGIC && model.version == ml::MODEL_VERSION);
  }
}

static void resetDetector() {
  rawScore = smoothScore = 0;
  overCount = underCount = 0;
  alarmOn = warnOn = false;
  topFeature = -1;
  histHead = histLen = 0;
}

static void setState(State s) {
  state = s;
  stateSince = millis();
}

static void startLearning() {
  learnCount = 0;
  setState(ST_COUNTDOWN);
  beep(1800, 60);
  Serial.println("# learn: countdown");
}

static void cancelLearning() {
  setState(hasModel ? ST_MONITOR : ST_NOMODEL);
  resetDetector();
  beep(900, 80);
  Serial.println("# learn: cancelled");
}

static void printInfo() {
  Serial.printf("# IMU sample rate: %.1f Hz\n", sampleHz);
  if (!hasModel) { Serial.println("# no model yet - press A (or send 'l') to learn"); return; }
  Serial.printf("# model: %u train + %u held-back windows @ %.0f Hz\n", model.nTrain, model.nVal, model.sampleHz);
  Serial.printf("# held-back error: mean %.4f  std %.4f  max %.4f\n", model.errMean, model.errStd, model.errMax);
  Serial.printf("# limit (%s): %.4f   sensitivity LOW %.4f / MED %.4f / HIGH %.4f\n", SENS_NAMES[sensitivity],
                ml::threshold(model, sensitivity), ml::threshold(model, 0), ml::threshold(model, 1),
                ml::threshold(model, 2));
}

// ---------------------------------------------------------------- drawing
static uint16_t stateColor() {
  switch (state) {
    case ST_NOMODEL:   return canvas.color565(70, 75, 85);
    case ST_COUNTDOWN:
    case ST_LEARNING:  return canvas.color565(30, 100, 200);
    case ST_TRAINING:  return canvas.color565(120, 60, 190);
    case ST_MONITOR:
    default:
      if (alarmOn) return ((millis() / 300) & 1) ? canvas.color565(215, 30, 30) : canvas.color565(140, 0, 0);
      if (warnOn) return canvas.color565(240, 160, 0);
      return canvas.color565(25, 145, 60);
  }
}

static void drawFooter() {
  canvas.drawFastHLine(0, 216, 320, canvas.color565(60, 60, 60));
  canvas.setFont(&fonts::FreeSansBold9pt7b);
  canvas.setTextDatum(bottom_center);
  canvas.setTextColor((state == ST_COUNTDOWN || state == ST_LEARNING) ? TFT_YELLOW : TFT_WHITE);
  canvas.drawString((state == ST_COUNTDOWN || state == ST_LEARNING) ? "CANCEL" : "LEARN", 65, 238);
  canvas.setTextColor(muted ? TFT_YELLOW : TFT_WHITE);
  canvas.drawString(muted ? "UNMUTE" : "MUTE", 160, 238);
  canvas.setTextColor(TFT_WHITE);
  char buf[16];
  snprintf(buf, sizeof(buf), "SENS %s", SENS_NAMES[sensitivity]);
  canvas.drawString(buf, 255, 238);
}

static void drawHeader(const char* title, const char* right) {
  const uint16_t bg = stateColor();
  canvas.fillRect(0, 0, 320, 46, bg);
  const bool dark = (state == ST_MONITOR && warnOn && !alarmOn);
  canvas.setTextColor(dark ? TFT_BLACK : TFT_WHITE);
  canvas.setFont(&fonts::FreeSansBold18pt7b);
  canvas.setTextDatum(middle_left);
  canvas.drawString(title, 10, 24);
  if (right) {
    canvas.setFont(&fonts::FreeSans9pt7b);
    canvas.setTextDatum(middle_right);
    canvas.drawString(right, 312, 24);
  }
}

// 12 band bars. With a model: how far each band is from normal (z-score).
// Without one: the raw band energy.
static void drawBands(int y0, int h) {
  const int x0 = 10, w = 300;
  const int bw = w / ml::N_BANDS;
  canvas.setFont(&fonts::Font0);
  canvas.setTextColor(canvas.color565(160, 160, 160));
  canvas.setTextDatum(top_left);
  if (!haveFeat) return;
  if (hasModel && state == ST_MONITOR) {
    const int mid = y0 + h / 2;
    canvas.drawFastHLine(x0, mid, w, canvas.color565(90, 90, 90));
    for (int b = 0; b < ml::N_BANDS; b++) {
      float z = zLive[4 + b];
      if (z > 6) z = 6;
      if (z < -6) z = -6;
      const int len = (int)(fabsf(z) / 6.0f * (h / 2 - 1));
      const float az = fabsf(z);
      const uint16_t col = az < 3 ? canvas.color565(40, 190, 90) : (az < 5 ? canvas.color565(240, 160, 0) : canvas.color565(230, 40, 40));
      const int bx = x0 + b * bw + 2;
      if (z >= 0) canvas.fillRect(bx, mid - len, bw - 4, len, col);
      else canvas.fillRect(bx, mid + 1, bw - 4, len, col);
    }
    canvas.drawString("frequency bands vs. normal (up = stronger, down = weaker)", x0, y0 + h + 2);
  } else {
    for (int b = 0; b < ml::N_BANDS; b++) {
      float v = (feat[4 + b] + 9.0f) / 8.0f;    // log10 energy -9..-1 -> 0..1
      if (v < 0) v = 0;
      if (v > 1) v = 1;
      const int len = (int)(v * h);
      canvas.fillRect(x0 + b * bw + 2, y0 + h - len, bw - 4, len, canvas.color565(70, 150, 230));
    }
    char buf[48];
    snprintf(buf, sizeof(buf), "live spectrum  %.0f Hz ... %.0f Hz", bandHz(0), bandHz(ml::N_BANDS));
    canvas.drawString(buf, x0, y0 + h + 2);
  }
}

static void drawProgress(int y, float frac, uint16_t col) {
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
  canvas.drawRoundRect(10, y, 300, 18, 6, TFT_WHITE);
  canvas.fillRoundRect(12, y + 2, (int)(296 * frac), 14, 5, col);
}

static void drawHistory(int y0, int h) {
  const int x0 = 10, w = 300;
  const float maxS = 2.0f;
  canvas.drawRect(x0, y0, w, h, canvas.color565(70, 70, 70));
  const int yThr = y0 + h - (int)(1.0f / maxS * h);
  for (int x = x0; x < x0 + w; x += 6) canvas.drawFastHLine(x, yThr, 3, canvas.color565(230, 60, 60));
  const int yWarn = y0 + h - (int)(WARN_SCORE / maxS * h);
  for (int x = x0; x < x0 + w; x += 6) canvas.drawPixel(x, yWarn, canvas.color565(240, 160, 0));
  canvas.setFont(&fonts::Font0);
  canvas.setTextColor(canvas.color565(230, 90, 90));
  canvas.setTextDatum(bottom_right);
  canvas.drawString("alarm limit", x0 + w - 3, yThr - 1);
  if (histLen < 2) return;
  int px = -1, py = -1;
  for (int i = 0; i < histLen; i++) {
    const int idx = (histHead - histLen + i + HIST) % HIST;
    float s = hist[idx];
    if (s > maxS) s = maxS;
    const int x = x0 + w - 1 - (histLen - 1 - i) * (w - 2) / (HIST - 1);
    const int y = y0 + h - 1 - (int)(s / maxS * (h - 2));
    const uint16_t col = hist[idx] >= 1.0f ? canvas.color565(255, 80, 80) : TFT_WHITE;
    if (px >= 0) canvas.drawLine(px, py, x, y, col);
    px = x; py = y;
  }
}

static void drawScreen() {
  canvas.fillScreen(TFT_BLACK);
  char buf[64];
  canvas.setTextColor(TFT_WHITE);

  switch (state) {
    case ST_NOMODEL: {
      drawHeader("NO MODEL", "TinyML vibration");
      canvas.setFont(&fonts::FreeSans9pt7b);
      canvas.setTextDatum(top_left);
      canvas.drawString("1. Put the Fire on the machine", 12, 58);
      canvas.drawString("2. Run the machine normally", 12, 80);
      canvas.drawString("3. Press A (LEARN) - 30 s", 12, 102);
      drawBands(130, 62);
      break;
    }
    case ST_COUNTDOWN: {
      drawHeader("GET READY", "hands off!");
      const int left = COUNTDOWN_SECONDS - (int)((millis() - stateSince) / 1000);
      canvas.setFont(&fonts::Font7);
      canvas.setTextDatum(middle_center);
      snprintf(buf, sizeof(buf), "%d", left > 0 ? left : 0);
      canvas.drawString(buf, 160, 110);
      canvas.setFont(&fonts::FreeSans9pt7b);
      canvas.drawString("Recording starts soon. Don't touch it.", 160, 180);
      break;
    }
    case ST_LEARNING: {
      const float el = (millis() - stateSince) / 1000.0f;
      snprintf(buf, sizeof(buf), "%.0f s left", fmaxf(0.0f, LEARN_SECONDS - el));
      drawHeader("LEARNING", buf);
      canvas.setFont(&fonts::FreeSans9pt7b);
      canvas.setTextDatum(top_left);
      canvas.drawString("Recording normal vibration...", 12, 56);
      drawProgress(80, el / LEARN_SECONDS, canvas.color565(60, 150, 240));
      snprintf(buf, sizeof(buf), "%d windows captured", learnCount);
      canvas.drawString(buf, 12, 104);
      drawBands(130, 62);
      break;
    }
    case ST_TRAINING: {
      snprintf(buf, sizeof(buf), "epoch %d/%d", trainEpoch, TRAIN_EPOCHS);
      drawHeader("TRAINING", buf);
      canvas.setFont(&fonts::FreeSans9pt7b);
      canvas.setTextDatum(top_left);
      canvas.drawString("Training the autoencoder on-device", 12, 58);
      drawProgress(84, (float)trainEpoch / TRAIN_EPOCHS, canvas.color565(160, 90, 230));
      snprintf(buf, sizeof(buf), "loss %.3f   %d windows", trainLoss, learnCount);
      canvas.drawString(buf, 12, 112);
      canvas.drawString("16 > 8 > 4 > 8 > 16 neurons", 12, 136);
      break;
    }
    case ST_MONITOR: {
      const char* title = alarmOn ? "ANOMALY!" : (warnOn ? "CAREFUL" : "NORMAL");
      snprintf(buf, sizeof(buf), "score %.2f", smoothScore);
      drawHeader(title, buf);
      drawHistory(52, 74);
      canvas.setFont(&fonts::FreeSans9pt7b);
      canvas.setTextDatum(top_left);
      if (alarmOn && topFeature >= 0) {
        char name[32];
        featureName(topFeature, name, sizeof(name));
        snprintf(buf, sizeof(buf), "Changed most: %s", name);
      } else {
        snprintf(buf, sizeof(buf), "Anomalies today: %lu", (unsigned long)anomalyCount);
      }
      canvas.drawString(buf, 12, 130);
      drawBands(152, 50);
      break;
    }
  }
  drawFooter();
  canvas.pushSprite(0, 0);
}

static void drawTrainingProgress(int epoch, int epochs, float loss) {
  trainEpoch = epoch;
  trainLoss = loss;
  if (epoch == 1 || epoch % 10 == 0 || epoch == epochs) {
    drawScreen();
    const int lit = (epoch * LED_COUNT) / epochs;
    for (int i = 0; i < LED_COUNT; i++) leds.setPixelColor(i, i < lit ? leds.Color(120, 50, 200) : 0);
    leds.show();
  }
  if (epoch % 20 == 0) Serial.printf("# train: epoch %d/%d  loss %.4f\n", epoch, epochs, loss);
  delay(0);
}

// ---------------------------------------------------------------- LEDs
static void updateLeds() {
  uint32_t c = 0;
  const uint32_t t = millis();
  switch (state) {
    case ST_NOMODEL: {
      const int v = 20 + (int)(20 * sinf(t / 600.0f));
      c = leds.Color(v, v, v);
      break;
    }
    case ST_COUNTDOWN: c = ((t / 250) & 1) ? leds.Color(0, 80, 255) : 0; break;
    case ST_LEARNING: {
      const float frac = (t - stateSince) / (LEARN_SECONDS * 1000.0f);
      const int lit = (int)(frac * 5 + 1);           // fills up on both sides
      for (int i = 0; i < 5; i++) {
        const uint32_t col = i < lit ? leds.Color(0, 80, 255) : 0;
        leds.setPixelColor(i, col);
        leds.setPixelColor(9 - i, col);
      }
      leds.show();
      return;
    }
    case ST_TRAINING: return;                         // drawn by the training callback
    case ST_MONITOR:
      if (alarmOn) c = ((t / 300) & 1) ? leds.Color(255, 0, 0) : 0;
      else if (warnOn) c = leds.Color(255, 140, 0);
      else c = leds.Color(0, 200, 40);
      break;
  }
  for (int i = 0; i < LED_COUNT; i++) leds.setPixelColor(i, c);
  leds.show();
}

// ---------------------------------------------------------------- logic
static void runTraining() {
  setState(ST_TRAINING);
  trainEpoch = 0;
  Serial.printf("# train: %d windows, %d epochs\n", learnCount, TRAIN_EPOCHS);
  const uint32_t t0 = millis();
  ml::Model fresh;
  const bool ok = ml::train(fresh, learnFeats, learnCount, TRAIN_EPOCHS, drawTrainingProgress);
  if (ok) {
    fresh.sampleHz = sampleHz;
    model = fresh;
    hasModel = true;
    saveModel();
    Serial.printf("# train: done in %lu ms\n", (unsigned long)(millis() - t0));
    printInfo();
    beep(2400, 80);
    delay(120);
    beep(3000, 120);
  } else {
    Serial.println("# train: failed (not enough data)");
    beep(600, 300);
  }
  xQueueReset(winQueue);              // windows that piled up during training are stale
  resetDetector();
  anomalyCount = 0;
  setState(hasModel ? ST_MONITOR : ST_NOMODEL);
}

static void processWindow(const Window& w) {
  ml::computeFeatures(w.x, w.y, w.z, feat);
  haveFeat = true;

  // ignore windows that overlap our own beeps or button presses
  const uint32_t winStart = w.endMs - (uint32_t)(ml::WIN * 1000 / (sampleHz > 50 ? sampleHz : 250));
  const bool tainted = (int32_t)(quietUntil - winStart) > 0;

  if (state == ST_LEARNING) {
    if (winStart < stateSince || tainted) return;
    if (learnCount < MAX_LEARN) {
      memcpy(&learnFeats[learnCount * ml::N_FEAT], feat, sizeof(feat));
      learnCount++;
    }
    return;
  }
  if (state != ST_MONITOR || !hasModel) return;

  const float err = ml::score(model, feat, zLive, featErr);
  const float thr = ml::threshold(model, sensitivity);
  rawScore = err / thr;
  smoothScore = 0.6f * smoothScore + 0.4f * rawScore;
  hist[histHead] = rawScore;
  histHead = (histHead + 1) % HIST;
  if (histLen < HIST) histLen++;

  Serial.printf("score:%.3f limit:1.000 err:%.4f%s\n", rawScore, err, tainted ? " (ignored)" : "");
  if (tainted) return;

  if (rawScore >= 1.0f) { overCount++; underCount = 0; }
  else { underCount++; overCount = 0; }

  if (!alarmOn && overCount >= ALARM_WINDOWS) {
    alarmOn = true;
    anomalyCount++;
    int best = 0;
    for (int i = 1; i < ml::N_FEAT; i++) if (featErr[i] > featErr[best]) best = i;
    topFeature = best;
    char name[32];
    featureName(best, name, sizeof(name));
    Serial.printf("# ANOMALY  score %.2f  changed most: %s\n", rawScore, name);
    lastBeep = 0;
  } else if (alarmOn && underCount >= CLEAR_WINDOWS) {
    alarmOn = false;
    Serial.println("# back to normal");
    beep(2200, 200);
  }
  if (alarmOn && rawScore >= 1.0f) {     // keep the explanation up to date while it lasts
    int best = 0;
    for (int i = 1; i < ml::N_FEAT; i++) if (featErr[i] > featErr[best]) best = i;
    topFeature = best;
  }
  warnOn = !alarmOn && smoothScore >= WARN_SCORE;
}

static void handleInput() {
  if (M5.BtnA.isPressed() || M5.BtnB.isPressed() || M5.BtnC.isPressed()) markQuiet(700);

  int cmd = Serial.available() ? Serial.read() : -1;
  const bool a = M5.BtnA.wasClicked() || cmd == 'l';
  const bool b = M5.BtnB.wasClicked() || cmd == 'm';
  const bool c = M5.BtnC.wasClicked() || cmd == 's';

  if (a) {
    if (state == ST_COUNTDOWN || state == ST_LEARNING) cancelLearning();
    else if (state != ST_TRAINING) startLearning();
  }
  if (b) {
    muted = !muted;
    prefs.putBool("mute", muted);
    if (muted) M5.Speaker.stop();
    beep(2000, 60);
    Serial.printf("# muted: %d\n", muted);
  }
  if (c) {
    sensitivity = (sensitivity + 1) % 3;
    prefs.putInt("sens", sensitivity);
    beep(2400, 40);
    Serial.printf("# sensitivity: %s\n", SENS_NAMES[sensitivity]);
  }
  if (cmd == 'i') printInfo();
}

// ---------------------------------------------------------------- main
void setup() {
  auto cfg = M5.config();
  cfg.serial_baudrate = 115200;
  M5.begin(cfg);
  M5.Display.setRotation(1);
  M5.Speaker.setVolume(SPEAKER_VOLUME);

  leds.begin();
  leds.setBrightness(LED_BRIGHTNESS);
  leds.clear();
  leds.show();

  prefs.begin("vibml", false);
  loadSettings();

  canvas.setPsram(true);
  canvas.setColorDepth(16);
  if (!canvas.createSprite(M5.Display.width(), M5.Display.height())) {
    canvas.setColorDepth(8);
    canvas.createSprite(M5.Display.width(), M5.Display.height());
  }

  if (!M5.Imu.isEnabled()) {
    M5.Display.fillScreen(TFT_RED);
    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.drawString("IMU not found!", 20, 100, &fonts::FreeSansBold12pt7b);
    Serial.println("IMU not found");
    while (true) delay(1000);
  }
  if (M5.Imu.getType() == m5::imu_t::imu_mpu6886) {
    // accelerometer low-pass at ~99 Hz so nothing above the 125 Hz Nyquist limit aliases in
    M5.In_I2C.writeRegister8(0x68, 0x1D, 0x02, 400000);
  }

  winQueue = xQueueCreate(2, sizeof(Window));
  xTaskCreatePinnedToCore(samplingTask, "imu", 4096, nullptr, 3, nullptr, 0);

  setState(hasModel ? ST_MONITOR : ST_NOMODEL);
  Serial.println("# TinyML vibration anomaly detector ready");
  printInfo();
}

void loop() {
  M5.update();
  handleInput();

  static uint32_t lastRate = 0, lastCount = 0;
  if (millis() - lastRate >= 1000) {
    const uint32_t now = sampleCounter;
    const float hz = (now - lastCount) * 1000.0f / (millis() - lastRate);
    sampleHz = sampleHz < 1 ? hz : 0.8f * sampleHz + 0.2f * hz;
    lastCount = now;
    lastRate = millis();
  }

  if (state == ST_COUNTDOWN && millis() - stateSince >= COUNTDOWN_SECONDS * 1000UL) {
    setState(ST_LEARNING);
    Serial.println("# learn: recording");
  }

  while (xQueueReceive(winQueue, &rxWin, 0) == pdTRUE) processWindow(rxWin);

  if (state == ST_LEARNING && millis() - stateSince >= LEARN_SECONDS * 1000UL) {
    if (learnCount >= 40) runTraining();
    else {
      Serial.printf("# learn: only %d windows, try again\n", learnCount);
      cancelLearning();
    }
  }

  if (state == ST_MONITOR && alarmOn && millis() - lastBeep >= BEEP_INTERVAL_MS) {
    lastBeep = millis();
    beep(1400, 120);
  }

  static uint32_t lastDraw = 0;
  if (millis() - lastDraw >= 50) {
    lastDraw = millis();
    updateLeds();
    drawScreen();
  }
}
