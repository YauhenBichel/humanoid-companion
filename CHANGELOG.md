# Changelog

## Unreleased

- Teammates: Byte explains computer science, Tempo sings. Each has its own persona, voice and look
  (face colours, antenna or headphones): `humanoid-talk --teammate byte|tempo`.
- A song library for the singer (`--songs`): the robot sings a requested song on the face page with
  the sung line as the caption while the body dances standing.
- `humanoid-perform`: speech or a song in, a character video out (transparent `.mov` or `.webm`, or
  `.mp4` on a colour). The mouth follows the voice or the sung words, and the singer dances on the beat.
- The face takes a colour look, and the renderer can draw transparent frames.
- Two singer teammates, Alesia (Алеся) and Maks (Максім): original animated characters of young
  Belarusian singers, drawn in code with gradient shading and a rim light, each with over-ear headphones
  and a wireless microphone (`humanoid-talk --teammate alesia|maks`, `humanoid-perform --teammate
  alesia|maks`). Their mouths take the shape of the vowel being sung (from `--words`), the microphone
  follows the mouth while they sing and lowers in the pauses, and the free arm dances through four
  poses on the beat. Chat voices: Kokoro `af_sky` and `am_michael`. Design sheets in `docs/media/`.
- A teammate has a `character` (how clips draw it) and an optional Belarusian `native_name`; a teammate
  file `based_on` a singer is drawn as that singer. Byte's and Tempo's clips are unchanged, checked
  frame by frame against frames recorded before the change.

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
