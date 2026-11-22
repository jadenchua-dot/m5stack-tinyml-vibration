#pragma once

// ======================= LEARNING =========================
#define COUNTDOWN_SECONDS   3      // "hands off" delay after pressing LEARN
#define LEARN_SECONDS       30     // how long normal vibration is recorded
#define TRAIN_EPOCHS        200    // autoencoder training passes over the data

// ======================= DETECTION ========================
#define DEFAULT_SENSITIVITY 1      // 0 = LOW, 1 = MED, 2 = HIGH
#define ALARM_WINDOWS       3      // windows in a row over the limit before alarming (~0.8 s)
#define CLEAR_WINDOWS       6      // windows in a row under the limit before clearing (~1.5 s)
#define WARN_SCORE          0.6f   // amber "CAREFUL" from this smoothed score (1.0 = alarm limit)

// ======================= OUTPUTS ==========================
#define BEEP_INTERVAL_MS    2000   // beep repeat while the anomaly lasts
#define SPEAKER_VOLUME      140    // 0-255
#define LED_BRIGHTNESS      40     // side LED bars, 0-255
