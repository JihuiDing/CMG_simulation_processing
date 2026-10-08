# Agent Instructions

## Before Starting

1. Read `00_project/README.md`.
2. Read `00_project/PROJECT_STATE.md`.
3. Read `00_project/FACTS.md`.
4. Read `00_project/DECISIONS.md`.
5. Read `00_project/TODO.md`.
6. Inspect relevant existing code and results.
7. Inspect relevant previous agent handoffs.

Do not begin by assuming that the task has never been attempted.

---

## General Rules

### Data

- Never modify raw data.
- Never overwrite original files.
- Preserve provenance.
- Document transformations.

### Code

- Prefer modifying/reusing existing code over creating duplicates.
- Keep code reproducible.
- Avoid hard-coded paths when possible.
- Add tests for important functions.
- Do not silently change existing methodology.

### Results

- Save important results to `03_results/`.
- Do not treat temporary console output as a project result.
- Record the code/version used to generate important results.

### Scientific Reasoning

- Clearly distinguish facts, assumptions, interpretations, and decisions.
- Do not assume previous agent conclusions are correct.
- Flag uncertainty.
- Do not hide failed experiments.

### Communication

- Important information must be written to files.
- Do not rely on chat history as project memory.
- Create a handoff note when another agent needs to continue or review the work.

---

## Before Finishing

1. Save all important files.
2. Check that code runs.
3. Record important results.
4. Update `PROJECT_STATE.md`.
5. Update `TODO.md`.
6. Update `DECISIONS.md` if a methodological decision was made.
7. Create a handoff note if another agent should review or continue the work.