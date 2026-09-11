# Security policy

Please report a vulnerability privately through GitHub's
[private vulnerability reporting](https://github.com/YauhenBichel/humanoid-companion/security/advisories/new),
not in a public issue. You will get an answer within a week.

In scope: the face server (`humanoid_companion.face.server`, an HTTP server bound to 127.0.0.1 by
default), handling of model and speech server responses, and anything that could make the robot
move outside the validated, clamped commands.

Keep API keys in the environment (`HUMANOID_LLM_API_KEY`), never in code or recordings; the
recorded `conversation.json` stores what was said and which model answered, not keys.
