# Changelog

## Unreleased

- The teammates (Byte, Tempo, Alesia, Maks), their song library, `humanoid-perform` and the colour
  looks behind them are no longer part of the desktop companion: the music and edu characters are
  kept separately by the author. The companion is the plain humanoid again, as before the teammates.

## 0.1.0 (unreleased)

First public version, extracted from the author's humanoid experiments.

- Walking: bundled OP3 PPO policy (103 M steps), NumPy export, 50 Hz control loop with safety stops,
  C MuJoCo simulation and macOS viewer.
- Brain and conversation over any OpenAI-compatible chat server, JSON-schema answers, validation,
  clamps and stop words; optional name (`HUMANOID_USER_NAME`), never "owner".
- Face page with lip sync and live body camera; face renderer for videos.
- Gestures (wave, nod, celebrate, look around, dance), stability-checked.
- Voice over the OpenAI audio API; captions-only when no speech server runs; Belarusian goodbye
  through belarusian-tts.
- Training, hardened environment, evaluation, export and speed sweep.
