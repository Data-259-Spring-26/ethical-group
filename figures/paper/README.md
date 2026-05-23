# Paper figures (Quarto)

Link in Quarto with, e.g.:

```{{r fig-cap="BNPL use by financial vulnerability quartile."}}
knitr::include_graphics("../figures/paper/figure-01-bnpl-by-vulnerability-quartile.png")
```

| Figure | File | Research question |
|--------|------|-------------------|
| 1 | `figure-01-vulnerability-density-by-bnpl.png` | Vulnerability distribution by BNPL use (notebook) |
| 2 | `figure-02-predicted-bnpl-wls.png` | Predicted BNPL vs. vulnerability, WLS (notebook) |
| 3 | `figure-03-stratum-gaps-age-income.png` | BNPL gap by age × income stratum (notebook) |
| 4 | `figure-04-marketing-language-by-provider.png` | Marketing tone by provider |
| 5 | `figure-05-complaint-harm-by-provider.png` | CFPB complaint harm categories |
| 6 | `figure-06-marketing-urgency-vs-complaint-harm.png` | Marketing urgency vs. complaints |

Analyses 1–4 (SHED): `python scripts/export_notebook_figures.py`  
Analysis 5 (marketing/CFPB): `python scripts/generate_paper_figures.py`
