## Data cleaning report

| stage | in | out | dropped | drop % |
|---|---|---|---|---|
| input | 8898 | 8898 | 0 | 0.0% |
| exact_dedup | 8898 | 8027 | 871 | 9.8% |
| near_dedup_simhash | 8027 | 1309 | 6718 | 83.7% |
| length_filter | 1309 | 1295 | 14 | 1.1% |
| language_filter | 1295 | 1275 | 20 | 1.5% |
| toxicity_filter | 1275 | 1214 | 61 | 4.8% |

**Overall retention: 1214/8898 = 13.6%**
