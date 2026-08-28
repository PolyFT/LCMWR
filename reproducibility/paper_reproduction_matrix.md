# Paper reproduction matrix

This matrix links every paper and Supporting Information item to the repository evidence used to validate it. “Validated” means the tracked artifact and its declared numerical invariants pass `python scripts/validate_paper_alignment.py`; it does not imply that restricted inputs can be redistributed or that the complete workflow has been recreated in a clean environment.

## Main-text figures

| Paper item | Script/command | Configuration | Inputs | Expected outputs | Validation status |
|---|---|---|---|---|---|
| Figure 1 | Manual scientific schematic; no repository renderer | LCMWR workflow stages | Workflow design and methods | Typeset paper panel | Manual assembly documented; no generated asset claimed |
| Figure 2 | Manual scientific schematic; no repository renderer | Hierarchical composition rule | Component motif vectors and composition equations | Typeset paper panel | Manual assembly documented; composition behavior covered by unit tests |
| Figure 3 | `python scripts/feature_selection/run_figure3_analysis_notebook.py tg` for panels b–h; panel a is manual | Tg task, UMAP seed 48, selected screening thresholds | Tg vocabulary, selected features, 750-search and stepwise tables | `results/tg_motif_select/figure3_analysis/Figure3b-h_*.svg` | Panels b–h validated; panel a manual |
| Figure 4 | `python scripts/model_comparison/figure4_actual_best.py` for panels c–f; panels a–b are externally assembled | Seed 48; V-0 positive; task-specific best models | Local OOF predictions and tracked model summaries | `figure4c_tg_oof.svg`, `figure4d_t5_oof.svg`, `figure4e_loi_oof.svg`, `figure4f_ul94_oof_roc.svg` | Panels c–f and aggregate metrics validated; panels a–b manual/external |
| Figure 5 | `python scripts/shap_analysis/run_actual_best_shap.py loi` | ExtraTrees; descriptive full-data refit | Complete selected LOI features and saved model parameters | LOI SHAP summary and four dependence SVGs | Tracked rankings and five SVGs validated; not OOF-SHAP |
| Figure 6 | `python scripts/shap_analysis/run_actual_best_shap.py tg` | LightGBM; descriptive full-data refit | Complete selected Tg features and saved model parameters | Tg SHAP summary and four dependence SVGs | Tracked rankings and five SVGs validated; not OOF-SHAP |

## Supporting Information figures

| Paper item | Script/command | Configuration | Inputs | Expected outputs | Validation status |
|---|---|---|---|---|---|
| Figure S1 | Manual rendering of the IA component structures | IA row in the public LOI table | `dataset/LOI.csv` | Typeset structure panel | Structures and composition validated by the IA check; panel manual |
| Figure S2 | `python scripts/feature_selection/run_figure3_analysis_notebook.py loi` | LOI thresholds; UMAP seed 48 | LOI screening artifacts | `results/loi_motif_select/figure3_analysis/Figure3b-h_*.svg` | Seven SVG panels validated |
| Figure S3 | `python scripts/feature_selection/run_figure3_analysis_notebook.py t5` | T5 thresholds; UMAP seed 48 | T5 screening artifacts | `results/t5_motif_select/figure3_analysis/Figure3b-h_*.svg` | Seven SVG panels validated |
| Figure S4 | `python scripts/feature_selection/run_figure3_analysis_notebook.py ul94` | UL-94 thresholds; ROC-AUC; UMAP seed 48 | UL-94 screening artifacts | `results/ul94_motif_select/figure3_analysis/Figure3b-h_*.svg` | Seven SVG panels validated |
| Figure S5 | `python scripts/shap_analysis/run_actual_best_shap.py t5` | XGBoost; descriptive full-data refit | Complete selected T5 features and saved parameters | T5 SHAP summary and four dependence SVGs | Rankings and SVGs validated; not OOF-SHAP |
| Figure S6 | `python scripts/shap_analysis/run_actual_best_shap.py ul94` | XGBoost; `P(V-0)`; descriptive full-data refit | Complete selected UL-94 features and saved parameters | UL-94 SHAP summary and four dependence SVGs | Positive-class rankings and SVGs validated; not OOF-SHAP |
| Figure S7 | `python experiments/tg_grea/scripts/plot_figures.py` | GREA-only vocabulary; UMAP seed 48 | GREA screening artifacts | `experiments/tg_grea/results/figures/figure3_analysis/Figure3b-h_*.svg` | Seven aggregate SVG panels validated |
| Figure S8 | `python experiments/tg_grea/scripts/plot_figures.py` | LightGBM pooled outer-fold OOF | Local target-bearing GREA OOF table | `experiments/tg_grea/results/figures/figure4_oof/Figure4_tg_grea_oof.svg` | Published SVG and aggregate metrics validated; sample-level labels remain local |

## Supporting Information tables

| Paper item | Script/command | Configuration | Inputs | Expected outputs | Validation status |
|---|---|---|---|---|---|
| Table S1 | `python scripts/validate_paper_alignment.py` | IA identity and 0.5/0.5 molar composition | Public LOI row `PolymerName=IA` | Validated IA composition record | Structure, composition and LOI value validated |
| Table S2 | `python scripts/validate_paper_alignment.py` | Selected LOI vocabulary and unique RDKit substructure matches | IA components and LOI selected-feature list | 21 non-zero selected motif values | Motif counts and weighted values validated exactly |
| Table S3 | `python scripts/validate_paper_alignment.py` | Public and complete task scopes | Public source tables and complete selected matrices | Task sample counts and declared system distributions | Public/complete counts validated; complete system distributions declared because restricted rows are not published |
| Table S4 | `python scripts/validate_paper_alignment.py` | Four task-specific screening configurations | Search tables and Figure 3 summaries | Global, active and selected counts plus optimal thresholds | Counts, 750-row grids and optimal threshold rows validated |
| Table S5 | `results/model_compare/LOI/model_performance_summary.csv` | Nested 5×3 CV, 20 iterations, seed 48 | Complete selected LOI features | Metrics for 11 models | Eleven successful models and headline metrics validated |
| Table S6 | `results/model_compare/T5/model_performance_summary.csv` | Nested 5×3 CV, 20 iterations, seed 48 | Complete selected T5 features | Metrics for 11 models | Eleven successful models and headline metrics validated |
| Table S7 | `results/model_compare/Tg/model_performance_summary.csv` | Nested 5×3 CV, 20 iterations, seed 48 | Complete selected Tg features | Metrics for 11 models | Eleven successful models and headline metrics validated |
| Table S8 | `results/model_compare/UL94/model_performance_summary.csv` | Nested stratified 5×3 CV, V-0 positive, 20 iterations, seed 48 | Complete selected UL-94 features | Metrics for 11 models | Eleven successful models and headline metrics validated |
| Table S9 | `results/model_compare/figure4_oof/figure4_oof_metrics_summary.csv` | Pooled outer-fold held-out predictions/probabilities | Local actual-best OOF tables | Pooled OOF R²/RMSE or ROC-AUC | Four task rows and headline values validated |
| Table S10 | `results/loi_motif_select/best_improved_final_features_info.csv` | LOI optimal screening row | Complete LOI motif matrix | Ordered 149-motif list | Count and downstream ranking inputs validated |
| Table S11 | `results/t5_motif_select/best_improved_final_features_info.csv` | T5 optimal screening row | Complete T5 motif matrix | Ordered 582-motif list | Count and downstream ranking inputs validated |
| Table S12 | `results/tg_motif_select/best_improved_final_features_info.csv` | Tg optimal screening row | Complete Tg motif matrix | Ordered 156-motif list | Count and downstream ranking inputs validated |
| Table S13 | `results/ul94_motif_select/best_improved_final_features_info.csv` | UL-94 optimal screening row | Complete UL-94 motif matrix | Ordered 84-motif list | Count and downstream ranking inputs validated |
| Table S14 | `python experiments/tg_grea/scripts/run_pipeline.py validate` | 7174 structures, 7875 active and 1531 selected features | Structure-only release input plus local targets | GREA nested-CV and pooled OOF summary | Counts, LightGBM metrics and pooled OOF metrics validated |

## Headline numerical checks

The machine-readable expectations live in `reproducibility/paper_claims.json`. They are intentionally limited to publication-facing counts, thresholds, selected models, metrics and feature rankings; scientific runtime settings remain documented separately in `configs/scientific_protocol.json`.
