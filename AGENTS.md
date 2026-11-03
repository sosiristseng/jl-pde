# AGENTS.md

Tutorial/documentation site for partial differential equations (MethodOfLines.jl) and universal differential equations / PINNs (DiffEqFlux.jl, NeuralPDE.jl) in Julia. Not a Julia package: there is no src/test suite. Verification = the Literate notebooks in `docs/` execute end-to-end. The UDE notebooks were merged from https://github.com/sosiristseng/jl-ude (archived).

## Source of truth

- `docs/pde-*.jl`, `docs/ude-*.jl` and `docs/index.jl` are Literate notebooks written in Julia. They are the canonical sources; generated `.ipynb` files and the `.cache/` dir are gitignored and must not be committed.
- Literate markdown-string format (`mdstrings=true`): `#===` ... `===#` wraps markdown (LaTeX `$...` allowed), `#---` separates code blocks. Follow existing style in `docs/*.jl`.
- `ci.jl` only accepts `.jl` sources; `.ipynb` files in `docs/` are ignored and left untouched (its `to_literate` reverse-converter is kept but unused, for manual conversions).
- All notebook deps live in the root `Project.toml`; notebooks `using` them directly. `Startup/` is an empty placeholder sub-project of the Pkg workspace, not a library to write code in.
- `docs/data/*.txt` holds reference solutions read by `pde-04-cfd.jl`.

## Commands

- Env uses the Pkg workspace at the repo root — always `--project=@.`:
  `julia --project=@. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`
- Rebuild/execute all notebooks like CI (SHA256 cache in `.cache/`, misses rerun automatically):
  `julia --project=@. -p 4 .github/ci.jl` (env vars: `DOCDIR=docs`, `NBCACHE=.cache`; locally `-p 16` is fine)
- Run a single notebook the way CI matrix jobs do (`NB` switches `ci.jl` to single-notebook mode):
  `NB=docs/ude-01-firstexample.jl julia --project=@. --color=yes .github/ci.jl`
- Run a single notebook:
  `julia --project=@. -e 'using Literate; Literate.notebook("docs/ude-01-firstexample.jl", "docs"; mdstrings=true, execute=true)'`
  (the generated `docs/*.ipynb` is harmless: gitignored, and `ci.jl` ignores it)
- No lint/typecheck/test commands exist. There is no local CI shortcut other than running `ci.jl`.

## Notebook execution gotchas

- CI executes one notebook per matrix job (dynamic matrix over `docs/**/*.jl`, `max-parallel: 10`, `fail-fast: false`); a green `CI` conclusion job is the status check for Kodiak.
- Notebooks are compute-heavy (MethodOfLines discretizations, neural ODE training, PINNs, AdvancedHMC sampling). CI matrix jobs run with `JULIA_NUM_THREADS=auto` and `OPENBLAS_NUM_THREADS=1`; pin `JULIA_NUM_THREADS=1` locally to reproduce CI timing/behavior. `ENV["GKSwstype"]="100"` is set by `ci.jl` for headless Plots.jl. CI also sets `JULIA_CONDAPKG_ENV=@mpl`; keep it.
- Plots.jl output is post-processed by `ci.jl` (SVG/HTML stripped when PNG exists); don't hand-edit generated `.ipynb` outputs.
- Notebooks are seeded (`Random.seed!`), but trained-network outputs may still drift slightly across package versions — cosmetic, not a failure.

## Julia version & manifests

- Julia is pinned to 1.13 (`[compat]` in `Project.toml`) and CI reads the exact version from the nonstandard `julia_version = "1.13.x"` line in `Manifest.toml` — change it there, not in the workflow.
- `Manifest.toml` is committed on purpose (static environment). It is updated by weekly "Julia Dependency Update" PRs (`Pkg.update()`); Dependabot covers GitHub Actions only. PRs with the `automerge` label are squash-merged by Kodiak, so manifest churn on `main` is expected — don't hand-edit dependency versions in `Manifest.toml`.
- CI's notebook cache is keyed per notebook (env hash + that notebook file's hash); editing a `docs/*.jl` forces only its re-execution. Built notebooks reach the `render` job as artifacts.

## Site build

- `docs/myst.yml` (MyST) is the site config; its TOC references generated `index.ipynb` / `pde-*.ipynb` / `ude-*.ipynb`, so notebooks must be built first.
- The site uses the **npm** `jupyter-book` package (`npm i -g jupyter-book`), not the Python one. Build in `docs/`: `jupyter-book build --html`. Deploy to GitHub Pages happens on `main` only.
- The site and notebooks use no Python.

## Formatting

Per `.editorconfig`: 4-space indent in `.jl`, 2-space in `.yml`/`.toml`/`.md`/`.json`; trailing whitespace trimmed.
