# Changelog

## Unreleased

- Teammates: Byte explains computer science, Tempo sings. Each has its own persona, voice and look
  (face colours, antenna or headphones): `humanoid-talk --teammate byte|tempo`.
- A song library for the singer (`--songs`): the robot sings a requested song on the face page with
  the sung line as the caption while the body dances standing.
- `humanoid-perform`: speech or a song in, a character video out (transparent `.mov` or `.webm`, or
  `.mp4` on a colour). The mouth follows the voice or the sung words, and the singer dances on the beat.
- The face takes a colour look, and the renderer can draw transparent frames.

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
