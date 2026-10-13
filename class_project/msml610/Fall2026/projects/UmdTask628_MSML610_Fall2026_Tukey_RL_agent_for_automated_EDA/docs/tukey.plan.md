# Tukey: plan for review

- MSML610 Fall 2026, group 31: Rayhan Patel, Ronak Vimal, Sai Katupilla
- Issue: gpsaggese/umd_classes#628
- Spec: `research/ideas/in_progress.RL_for_auto_EDA.md`
- Builds on: #506 and the prototype merged in #527
  (`research/agentic_data_science/RL_for_auto_EDA_prototype.py`)
- The spec's next steps ask for "GP to review / approve the plan"; this page is
  that plan, kept in sync with the build; each milestone first delivers the result
  the spec names, and anything more is marked beyond the spec

## Milestones

| # | The spec's result | Beyond the spec |
|---|---|---|
| 1 | Gym-style environment: random graph G*, data split into D_train / D_test, G* held out | Generator families incl. the spec's pgmpy and `make_regression`; standardised data; a per-instance identifiable target (DAG, CPDAG or pattern); leak checks; sortability report |
| 2 | Toolbox, scripted agent emitting a graph, scoring with SHD and out-of-sample prediction | Tools as costed actions; baseline table (scripted, tuned scripted, PC, GES, DirectLiNGAM, sort-and-regress, FCI for hidden confounders, AVICI optional) |
| 3 | Policy trained with REINFORCE / PPO that beats the scripted baseline on held-out graphs | Size-independent policy; decoder that always emits an acyclic graph with an exact probability; baselines without tools and with stronger hand-written rules; matched-budget comparison |
| 4 | Curriculum over size, density, noise; transfer to real data with unknown ground truth; report on the sim-to-real gap and failure modes | Unseen generators and topologies; the variance shortcut; confounding; Sachs and a dataset with no ground truth |

## Design choices to confirm

1. **Score**: the spec's combined form, `-lam * SHD / |E*| - (1 - lam) * (1 -
   pred_frac)` with `lam = 0.5`, where `pred_frac` is the held-out
   log-likelihood gain over the empty graph relative to the true graph's gain; SHD
   is measured against what the data can identify (the CPDAG for standardised
   Gaussian data, the full DAG for non-Gaussian noise)
2. **Key Example 1**: with Gaussian noise, regression alone cannot orient edges
   unless noise variances are equal on the raw scale; we score only identifiable
   structure
3. **Key Example 2**: a CI test removes the spurious edge only if the confounder is
   observed; with a truly hidden confounder the correct answer keeps the adjacency,
   so we test both cases separately
4. **Evaluation**: all tuning on validation graphs; one confirmatory run on sealed
   test graphs after a freeze; the trained policy must beat the scripted agent on
   both the combined score and SHD at the scripted agent's own call budget

## Schedule

- Build order: setup, integration spike with executable gates, then Milestones 1-4,
  docs, clean reproduction and video, as fast as each step is verified
- Outer limits: midterm PR review Wed Nov 4; final submission Mon Dec 7

## Questions for GP

- Do the weighting `lam = 0.5` and the identifiable-target scoring match your intent?
- Is a dataset with no ground truth (scored without labels) plus Sachs (scored
  against its disputed reference) the right reading of the Milestone 4 real-data
  test?
