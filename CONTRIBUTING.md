# Contributing

Thank you for helping a small robot learn to walk and talk.

- **Issues:** a bug (what you ran, what happened, your OS and model server), an idea, or a result
  from your own robot or policy. Results are especially welcome: a speed sweep or gesture check
  from a policy you trained, a model that follows the reply schema well (or badly), a real robot.
- **Pull requests:** one change per pull request, with a test for the behaviour you changed.
  `uv sync --frozen && uv run pytest -q` must pass; the tests run on a CPU and fake every outside
  service (model, speech, transcription), so no server or GPU is needed.
- **Safety rules stay:** the model's answers are validated and clamped before anything moves; stop
  words never reach the model; a failure means standing still. A change that weakens one of these
  needs a very good reason and a test.
- **Gestures:** a new gesture needs a row in `docs/gesture-stability.md` from
  `python -m humanoid_companion.gestures --check`, standing and, if it is allowed while walking,
  walking.

By contributing you agree that your contribution is licensed under Apache-2.0.
