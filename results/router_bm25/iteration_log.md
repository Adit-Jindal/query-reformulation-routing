# Router iteration log

- datasets: ['scifact', 'trec-covid', 'nfcorpus', 'fiqa', 'webis-touche2020']  | retriever: bm25 | metric: ndcg10 | queries: 1370 (dropped for missing values: 0)
- seeds used for cross-fit oracle: [0, 1, 2] | oracle: crossfit | features: stage<=1 (34 usable)
- lambda grid (nDCG per second): [0.0, 0.002, 0.005, 0.01, 0.02, 0.05] | folds 5 x repeats 3 | bootstrap B=5000

## Baselines (nDCG@10)

```
         dataset    n  fixed_a0  fixed_prf  fixed_query2doc  fixed_hyde  fixed_decomp  oracle_crossfit  oracle_naive  go_headroom_crossfit  oracle_go_rate
         scifact  300    0.6617     0.5692           0.6982      0.6662        0.6192           0.7618        0.7739                0.1001          0.3367
      trec-covid   50    0.5945     0.5977           0.7071      0.5982        0.6247           0.7482        0.7655                0.1537          0.8200
        nfcorpus  323    0.3064     0.3044           0.3427      0.2973        0.3085           0.3915        0.4012                0.0851          0.6006
            fiqa  648    0.2326     0.1848           0.2735      0.2125        0.2033           0.3409        0.3556                0.1083          0.5077
webis-touche2020   49    0.3579     0.3749           0.4414      0.3006        0.3386           0.4710        0.4926                0.1131          0.7959
             ALL 1370    0.3616     0.3190           0.4047      0.3491        0.3394           0.4645        0.4778                0.1029          0.5139
```

## Protocol `within`, model `const`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
  within const     1    go_nogo           all 1370          0.453         0.547    -0.032 0.494
  within const     1    go_nogo oracle_routed 1370          0.453         0.547    -0.032 0.494
  within const     2 llm_vs_prf           all 1370          0.488         0.512     0.063 0.466
  within const     2 llm_vs_prf oracle_routed  621          0.718         0.718     0.197 0.581
  within const     3 llm_choice           all 1370          0.685         0.685     0.035   NaN
  within const     3 llm_choice oracle_routed  446          0.392         0.392     0.063   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6617      0.6982  0.7536  0.7618     0.9179      1.0 0.0919   0.0690   0.1168     0.0024
      trec-covid      1  0.0   50 0.5945      0.7071  0.7368  0.7482     0.9258      1.0 0.1423   0.0934   0.1927     0.0024
        nfcorpus      1  0.0  323 0.3064      0.3427  0.3891  0.3915     0.9717      1.0 0.0827   0.0683   0.0979     0.0024
            fiqa      1  0.0  648 0.2326      0.2735  0.3298  0.3409     0.8976      1.0 0.0972   0.0820   0.1136     0.0024
webis-touche2020      1  0.0   49 0.3579      0.4414  0.4658  0.4710     0.9541      1.0 0.1079   0.0723   0.1442     0.0024
             ALL      1  0.0 1370 0.3616      0.4047  0.4563  0.4645     0.9202      1.0 0.0947   0.0845   0.1047     0.0024
```

Failure sample L1: harmful_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6617      0.6982  0.7400  0.7618     0.7819      1.0 0.0783   0.0557   0.1026     0.0024
      trec-covid      2  0.0   50 0.5945      0.7071  0.7143  0.7482     0.7791      1.0 0.1197   0.0704   0.1730     0.0024
        nfcorpus      2  0.0  323 0.3064      0.3427  0.3602  0.3915     0.6324      1.0 0.0538   0.0405   0.0687     0.0024
            fiqa      2  0.0  648 0.2326      0.2735  0.3156  0.3409     0.7668      1.0 0.0831   0.0685   0.0993     0.0024
webis-touche2020      2  0.0   49 0.3579      0.4414  0.4348  0.4710     0.6800      1.0 0.0769   0.0405   0.1134     0.0024
             ALL      2  0.0 1370 0.3616      0.4047  0.4379  0.4645     0.7411      1.0 0.0762   0.0662   0.0862     0.0024
```

Failure sample L2: harmful_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6617      0.6982  0.6982  0.7618     0.3644      1.0 0.0365   0.0201   0.0547     0.0024
      trec-covid      3  0.0   50 0.5945      0.7071  0.7071  0.7482     0.7321      1.0 0.1125   0.0722   0.1543     0.0024
        nfcorpus      3  0.0  323 0.3064      0.3427  0.3427  0.3915     0.4267      1.0 0.0363   0.0252   0.0487     0.0024
            fiqa      3  0.0  648 0.2326      0.2735  0.2735  0.3409     0.3781      1.0 0.0410   0.0295   0.0531     0.0024
webis-touche2020      3  0.0   49 0.3579      0.4414  0.4414  0.4710     0.7383      1.0 0.0835   0.0500   0.1178     0.0024
             ALL      3  0.0 1370 0.3616      0.4047  0.4047  0.4645     0.4181      1.0 0.0430   0.0355   0.0506     0.0024
```

Failure sample L3: harmful_go=15, wrong_family=15, success=12

## Protocol `within`, model `stump`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
  within stump     1    go_nogo           all 1370          0.475         0.547     0.042 0.522
  within stump     1    go_nogo oracle_routed 1370          0.475         0.547     0.042 0.522
  within stump     2 llm_vs_prf           all 1370          0.488         0.512     0.063 0.466
  within stump     2 llm_vs_prf oracle_routed  621          0.718         0.718     0.197 0.581
  within stump     3 llm_choice           all 1370          0.685         0.685     0.035   NaN
  within stump     3 llm_choice oracle_routed  446          0.392         0.392     0.063   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6617      0.6982  0.7495  0.7618     0.8770   0.8844 0.0878   0.0653   0.1123     0.0024
      trec-covid      1  0.0   50 0.5945      0.7071  0.7294  0.7482     0.8772   0.9267 0.1348   0.0875   0.1849     0.0024
        nfcorpus      1  0.0  323 0.3064      0.3427  0.3807  0.3915     0.8723   0.8607 0.0743   0.0604   0.0892     0.0024
            fiqa      1  0.0  648 0.2326      0.2735  0.3199  0.3409     0.8063   0.9033 0.0873   0.0736   0.1026     0.0024
webis-touche2020      1  0.0   49 0.3579      0.4414  0.4567  0.4710     0.8731   0.9592 0.0988   0.0647   0.1334     0.0024
             ALL      1  0.0 1370 0.3616      0.4047  0.4481  0.4645     0.8407   0.8920 0.0865   0.0768   0.0960     0.0024
```

Failure sample L1: harmful_go=15, missed_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6617      0.6982  0.7363  0.7618     0.7452   0.8844 0.0746   0.0523   0.0985     0.0024
      trec-covid      2  0.0   50 0.5945      0.7071  0.7111  0.7482     0.7584   0.9267 0.1166   0.0691   0.1693     0.0024
        nfcorpus      2  0.0  323 0.3064      0.3427  0.3542  0.3915     0.5609   0.8607 0.0478   0.0352   0.0615     0.0024
            fiqa      2  0.0  648 0.2326      0.2735  0.3079  0.3409     0.6953   0.9033 0.0753   0.0619   0.0902     0.0024
webis-touche2020      2  0.0   49 0.3579      0.4414  0.4311  0.4710     0.6466   0.9592 0.0731   0.0376   0.1091     0.0024
             ALL      2  0.0 1370 0.3616      0.4047  0.4317  0.4645     0.6812   0.8920 0.0701   0.0605   0.0795     0.0024
```

Failure sample L2: harmful_go=15, missed_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6617      0.6982  0.6962  0.7618     0.3451   0.8844 0.0346   0.0183   0.0525     0.0024
      trec-covid      3  0.0   50 0.5945      0.7071  0.7028  0.7482     0.7047   0.9267 0.1083   0.0693   0.1484     0.0024
        nfcorpus      3  0.0  323 0.3064      0.3427  0.3388  0.3915     0.3805   0.8607 0.0324   0.0221   0.0442     0.0024
            fiqa      3  0.0  648 0.2326      0.2735  0.2716  0.3409     0.3598   0.9033 0.0390   0.0280   0.0506     0.0024
webis-touche2020      3  0.0   49 0.3579      0.4414  0.4376  0.4710     0.7048   0.9592 0.0797   0.0476   0.1132     0.0024
             ALL      3  0.0 1370 0.3616      0.4047  0.4021  0.4645     0.3931   0.8920 0.0404   0.0334   0.0479     0.0024
```

Failure sample L3: harmful_go=15, missed_go=15, wrong_family=15, success=12

## Protocol `within`, model `ridge`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
  within ridge     1    go_nogo           all 1370          0.465         0.547     0.071 0.535
  within ridge     1    go_nogo oracle_routed 1370          0.465         0.547     0.071 0.535
  within ridge     2 llm_vs_prf           all 1370          0.498         0.512     0.200 0.572
  within ridge     2 llm_vs_prf oracle_routed  621          0.710         0.718     0.300 0.632
  within ridge     3 llm_choice           all 1370          0.677         0.685     0.130   NaN
  within ridge     3 llm_choice oracle_routed  446          0.383         0.392     0.133   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6617      0.6982  0.7521  0.7618     0.9026   0.9156 0.0904   0.0676   0.1152     0.0024
      trec-covid      1  0.0   50 0.5945      0.7071  0.7368  0.7482     0.9258   1.0000 0.1423   0.0934   0.1927     0.0024
        nfcorpus      1  0.0  323 0.3064      0.3427  0.3891  0.3915     0.9717   1.0000 0.0827   0.0683   0.0979     0.0024
            fiqa      1  0.0  648 0.2326      0.2735  0.3298  0.3409     0.8976   1.0000 0.0972   0.0820   0.1136     0.0024
webis-touche2020      1  0.0   49 0.3579      0.4414  0.4493  0.4710     0.8077   0.8435 0.0914   0.0582   0.1263     0.0024
             ALL      1  0.0 1370 0.3616      0.4047  0.4554  0.4645     0.9111   0.9759 0.0937   0.0835   0.1038     0.0024
```

Null control (features shuffled), ALL: recovered=0.914

Failure sample L1: harmful_go=15, success=12, missed_go=9

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6617      0.6982  0.7376  0.7618     0.7579   0.9156 0.0759   0.0535   0.1001     0.0024
      trec-covid      2  0.0   50 0.5945      0.7071  0.7142  0.7482     0.7789   1.0000 0.1197   0.0704   0.1729     0.0024
        nfcorpus      2  0.0  323 0.3064      0.3427  0.3602  0.3915     0.6323   1.0000 0.0538   0.0405   0.0684     0.0024
            fiqa      2  0.0  648 0.2326      0.2735  0.3156  0.3409     0.7668   1.0000 0.0831   0.0685   0.0993     0.0024
webis-touche2020      2  0.0   49 0.3579      0.4414  0.4125  0.4710     0.4829   0.8435 0.0546   0.0192   0.0904     0.0072
             ALL      2  0.0 1370 0.3616      0.4047  0.4366  0.4645     0.7282   0.9759 0.0749   0.0649   0.0847     0.0024
```

Null control (features shuffled), ALL: recovered=0.727

Failure sample L2: harmful_go=15, wrong_family=15, success=12, missed_go=9

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6617      0.6982  0.6957  0.7618     0.3395   0.9156 0.0340   0.0173   0.0522     0.0024
      trec-covid      3  0.0   50 0.5945      0.7071  0.7067  0.7482     0.7300   1.0000 0.1122   0.0714   0.1540     0.0024
        nfcorpus      3  0.0  323 0.3064      0.3427  0.3424  0.3915     0.4233   1.0000 0.0360   0.0247   0.0485     0.0024
            fiqa      3  0.0  648 0.2326      0.2735  0.2726  0.3409     0.3695   1.0000 0.0400   0.0283   0.0525     0.0024
webis-touche2020      3  0.0   49 0.3579      0.4414  0.4134  0.4710     0.4901   0.8435 0.0554   0.0208   0.0907     0.0112
             ALL      3  0.0 1370 0.3616      0.4047  0.4026  0.4645     0.3980   0.9759 0.0410   0.0333   0.0485     0.0024
```

Null control (features shuffled), ALL: recovered=0.394

Failure sample L3: harmful_go=15, wrong_family=15, success=12, missed_go=9

## Protocol `within`, model `gbr`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
  within   gbr     1    go_nogo           all 1370          0.453         0.547     0.091 0.569
  within   gbr     1    go_nogo oracle_routed 1370          0.453         0.547     0.091 0.569
  within   gbr     2 llm_vs_prf           all 1370          0.501         0.512     0.165 0.555
  within   gbr     2 llm_vs_prf oracle_routed  621          0.718         0.718     0.256 0.609
  within   gbr     3 llm_choice           all 1370          0.678         0.685     0.109   NaN
  within   gbr     3 llm_choice oracle_routed  446          0.397         0.392     0.109   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6617      0.6982  0.7536  0.7618     0.9179   0.9989 0.0919   0.0690   0.1168     0.0024
      trec-covid      1  0.0   50 0.5945      0.7071  0.7346  0.7482     0.9111   0.9867 0.1400   0.0923   0.1901     0.0024
        nfcorpus      1  0.0  323 0.3064      0.3427  0.3891  0.3915     0.9717   1.0000 0.0827   0.0683   0.0979     0.0024
            fiqa      1  0.0  648 0.2326      0.2735  0.3298  0.3409     0.8976   1.0000 0.0972   0.0820   0.1136     0.0024
webis-touche2020      1  0.0   49 0.3579      0.4414  0.4635  0.4710     0.9336   0.9932 0.1056   0.0709   0.1411     0.0024
             ALL      1  0.0 1370 0.3616      0.4047  0.4562  0.4645     0.9185   0.9990 0.0945   0.0843   0.1045     0.0024
```

Null control (features shuffled), ALL: recovered=0.920

Failure sample L1: harmful_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6617      0.6982  0.7391  0.7618     0.7729   0.9989 0.0774   0.0549   0.1019     0.0024
      trec-covid      2  0.0   50 0.5945      0.7071  0.7011  0.7482     0.6934   0.9867 0.1066   0.0567   0.1605     0.0024
        nfcorpus      2  0.0  323 0.3064      0.3427  0.3610  0.3915     0.6410   1.0000 0.0546   0.0411   0.0693     0.0024
            fiqa      2  0.0  648 0.2326      0.2735  0.3156  0.3409     0.7668   1.0000 0.0831   0.0685   0.0993     0.0024
webis-touche2020      2  0.0   49 0.3579      0.4414  0.4235  0.4710     0.5794   0.9932 0.0655   0.0293   0.1002     0.0024
             ALL      2  0.0 1370 0.3616      0.4047  0.4370  0.4645     0.7322   0.9990 0.0753   0.0651   0.0853     0.0024
```

Null control (features shuffled), ALL: recovered=0.731

Failure sample L2: harmful_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6617      0.6982  0.6946  0.7618     0.3288   0.9989 0.0329   0.0154   0.0524     0.0024
      trec-covid      3  0.0   50 0.5945      0.7071  0.6836  0.7482     0.5794   0.9867 0.0890   0.0464   0.1327     0.0024
        nfcorpus      3  0.0  323 0.3064      0.3427  0.3432  0.3915     0.4325   1.0000 0.0368   0.0255   0.0496     0.0024
            fiqa      3  0.0  648 0.2326      0.2735  0.2727  0.3409     0.3703   1.0000 0.0401   0.0284   0.0522     0.0024
webis-touche2020      3  0.0   49 0.3579      0.4414  0.4279  0.4710     0.6190   0.9932 0.0700   0.0359   0.1032     0.0024
             ALL      3  0.0 1370 0.3616      0.4047  0.4023  0.4645     0.3948   0.9990 0.0406   0.0329   0.0483     0.0024
```

Null control (features shuffled), ALL: recovered=0.392

Failure sample L3: harmful_go=15, wrong_family=15, success=12

## Protocol `lodo`, model `const`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
    lodo const     1    go_nogo           all 1370          0.453         0.547     0.004 0.520
    lodo const     1    go_nogo oracle_routed 1370          0.453         0.547     0.004 0.520
    lodo const     2 llm_vs_prf           all 1370          0.488         0.512    -0.124 0.511
    lodo const     2 llm_vs_prf oracle_routed  621          0.718         0.718    -0.277 0.397
    lodo const     3 llm_choice           all 1370          0.685         0.685    -0.051   NaN
    lodo const     3 llm_choice oracle_routed  446          0.392         0.392    -0.086   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6617      0.6982  0.7536  0.7618     0.9179      1.0 0.0919   0.0690   0.1168     0.0024
      trec-covid      1  0.0   50 0.5945      0.7071  0.7368  0.7482     0.9258      1.0 0.1423   0.0934   0.1927     0.0024
        nfcorpus      1  0.0  323 0.3064      0.3427  0.3891  0.3915     0.9717      1.0 0.0827   0.0683   0.0979     0.0024
            fiqa      1  0.0  648 0.2326      0.2735  0.3298  0.3409     0.8976      1.0 0.0972   0.0820   0.1136     0.0024
webis-touche2020      1  0.0   49 0.3579      0.4414  0.4658  0.4710     0.9541      1.0 0.1079   0.0723   0.1442     0.0024
             ALL      1  0.0 1370 0.3616      0.4047  0.4563  0.4645     0.9202      1.0 0.0947   0.0845   0.1047     0.0024
```

Failure sample L1: harmful_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6617      0.6982  0.7400  0.7618     0.7819      1.0 0.0783   0.0557   0.1026     0.0024
      trec-covid      2  0.0   50 0.5945      0.7071  0.7143  0.7482     0.7791      1.0 0.1197   0.0704   0.1730     0.0024
        nfcorpus      2  0.0  323 0.3064      0.3427  0.3602  0.3915     0.6324      1.0 0.0538   0.0405   0.0687     0.0024
            fiqa      2  0.0  648 0.2326      0.2735  0.3156  0.3409     0.7668      1.0 0.0831   0.0685   0.0993     0.0024
webis-touche2020      2  0.0   49 0.3579      0.4414  0.4348  0.4710     0.6800      1.0 0.0769   0.0405   0.1134     0.0024
             ALL      2  0.0 1370 0.3616      0.4047  0.4379  0.4645     0.7411      1.0 0.0762   0.0662   0.0862     0.0024
```

Failure sample L2: harmful_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6617      0.6982  0.6982  0.7618     0.3644      1.0 0.0365   0.0201   0.0547     0.0024
      trec-covid      3  0.0   50 0.5945      0.7071  0.7071  0.7482     0.7321      1.0 0.1125   0.0722   0.1543     0.0024
        nfcorpus      3  0.0  323 0.3064      0.3427  0.3427  0.3915     0.4267      1.0 0.0363   0.0252   0.0487     0.0024
            fiqa      3  0.0  648 0.2326      0.2735  0.2735  0.3409     0.3781      1.0 0.0410   0.0295   0.0531     0.0024
webis-touche2020      3  0.0   49 0.3579      0.4414  0.4414  0.4710     0.7383      1.0 0.0835   0.0500   0.1178     0.0024
             ALL      3  0.0 1370 0.3616      0.4047  0.4047  0.4645     0.4181      1.0 0.0430   0.0355   0.0506     0.0024
```

Failure sample L3: harmful_go=15, wrong_family=15, success=12

## Protocol `lodo`, model `stump`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
    lodo stump     1    go_nogo           all 1370          0.462         0.547     0.028 0.500
    lodo stump     1    go_nogo oracle_routed 1370          0.462         0.547     0.028 0.500
    lodo stump     2 llm_vs_prf           all 1370          0.488         0.512    -0.124 0.511
    lodo stump     2 llm_vs_prf oracle_routed  621          0.718         0.718    -0.277 0.397
    lodo stump     3 llm_choice           all 1370          0.685         0.685    -0.051   NaN
    lodo stump     3 llm_choice oracle_routed  446          0.392         0.392    -0.086   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6617      0.6982  0.7477  0.7618     0.8590   0.9400 0.0860   0.0634   0.1104     0.0024
      trec-covid      1  0.0   50 0.5945      0.7071  0.7169  0.7482     0.7964   0.8800 0.1224   0.0797   0.1703     0.0024
        nfcorpus      1  0.0  323 0.3064      0.3427  0.3821  0.3915     0.8891   0.8978 0.0757   0.0619   0.0903     0.0024
            fiqa      1  0.0  648 0.2326      0.2735  0.3238  0.3409     0.8425   0.9012 0.0913   0.0764   0.1073     0.0024
webis-touche2020      1  0.0   49 0.3579      0.4414  0.4570  0.4710     0.8759   0.9388 0.0991   0.0630   0.1357     0.0024
             ALL      1  0.0 1370 0.3616      0.4047  0.4495  0.4645     0.8539   0.9095 0.0879   0.0779   0.0976     0.0024
```

Failure sample L1: harmful_go=15, missed_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6617      0.6982  0.7341  0.7618     0.7231   0.9400 0.0724   0.0507   0.0959     0.0024
      trec-covid      2  0.0   50 0.5945      0.7071  0.6938  0.7482     0.6458   0.8800 0.0992   0.0545   0.1504     0.0024
        nfcorpus      2  0.0  323 0.3064      0.3427  0.3552  0.3915     0.5728   0.8978 0.0488   0.0363   0.0623     0.0024
            fiqa      2  0.0  648 0.2326      0.2735  0.3121  0.3409     0.7340   0.9012 0.0795   0.0652   0.0951     0.0024
webis-touche2020      2  0.0   49 0.3579      0.4414  0.4292  0.4710     0.6302   0.9388 0.0713   0.0352   0.1078     0.0024
             ALL      2  0.0 1370 0.3616      0.4047  0.4328  0.4645     0.6913   0.9095 0.0711   0.0614   0.0807     0.0024
```

Failure sample L2: harmful_go=15, missed_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6617      0.6982  0.6944  0.7618     0.3268   0.9400 0.0327   0.0166   0.0503     0.0024
      trec-covid      3  0.0   50 0.5945      0.7071  0.6843  0.7482     0.5838   0.8800 0.0897   0.0547   0.1291     0.0024
        nfcorpus      3  0.0  323 0.3064      0.3427  0.3396  0.3915     0.3904   0.8978 0.0332   0.0223   0.0456     0.0024
            fiqa      3  0.0  648 0.2326      0.2735  0.2736  0.3409     0.3786   0.9012 0.0410   0.0298   0.0529     0.0024
webis-touche2020      3  0.0   49 0.3579      0.4414  0.4357  0.4710     0.6880   0.9388 0.0778   0.0443   0.1120     0.0024
             ALL      3  0.0 1370 0.3616      0.4047  0.4021  0.4645     0.3932   0.9095 0.0405   0.0332   0.0480     0.0024
```

Failure sample L3: harmful_go=15, missed_go=15, wrong_family=15, success=12

## Protocol `lodo`, model `ridge`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
    lodo ridge     1    go_nogo           all 1370          0.453         0.547     0.066 0.512
    lodo ridge     1    go_nogo oracle_routed 1370          0.453         0.547     0.066 0.512
    lodo ridge     2 llm_vs_prf           all 1370          0.491         0.512     0.048 0.565
    lodo ridge     2 llm_vs_prf oracle_routed  621          0.712         0.718    -0.005 0.524
    lodo ridge     3 llm_choice           all 1370          0.681         0.685     0.029   NaN
    lodo ridge     3 llm_choice oracle_routed  446          0.388         0.392     0.036   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6617      0.6982  0.7536  0.7618     0.9179   1.0000 0.0919   0.0690   0.1168     0.0024
      trec-covid      1  0.0   50 0.5945      0.7071  0.7344  0.7482     0.9100   0.9400 0.1399   0.0917   0.1906     0.0024
        nfcorpus      1  0.0  323 0.3064      0.3427  0.3891  0.3915     0.9717   1.0000 0.0827   0.0683   0.0979     0.0024
            fiqa      1  0.0  648 0.2326      0.2735  0.3295  0.3409     0.8945   0.9892 0.0969   0.0817   0.1132     0.0024
webis-touche2020      1  0.0   49 0.3579      0.4414  0.4658  0.4710     0.9541   1.0000 0.1079   0.0723   0.1442     0.0024
             ALL      1  0.0 1370 0.3616      0.4047  0.4561  0.4645     0.9178   0.9927 0.0944   0.0843   0.1045     0.0024
```

Null control (features shuffled), ALL: recovered=0.920

Failure sample L1: harmful_go=15, success=12, missed_go=4

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6617      0.6982  0.7400  0.7618     0.7819   1.0000 0.0783   0.0557   0.1026     0.0024
      trec-covid      2  0.0   50 0.5945      0.7071  0.7128  0.7482     0.7697   0.9400 0.1183   0.0696   0.1714     0.0024
        nfcorpus      2  0.0  323 0.3064      0.3427  0.3593  0.3915     0.6211   1.0000 0.0529   0.0396   0.0677     0.0024
            fiqa      2  0.0  648 0.2326      0.2735  0.3153  0.3409     0.7641   0.9892 0.0828   0.0681   0.0991     0.0024
webis-touche2020      2  0.0   49 0.3579      0.4414  0.4348  0.4710     0.6800   1.0000 0.0769   0.0405   0.1134     0.0024
             ALL      2  0.0 1370 0.3616      0.4047  0.4375  0.4645     0.7370   0.9927 0.0758   0.0657   0.0858     0.0024
```

Null control (features shuffled), ALL: recovered=0.741

Failure sample L2: harmful_go=15, wrong_family=15, success=12, missed_go=4

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6617      0.6982  0.6986  0.7618     0.3682   1.0000 0.0369   0.0193   0.0562     0.0024
      trec-covid      3  0.0   50 0.5945      0.7071  0.7053  0.7482     0.7208   0.9400 0.1108   0.0716   0.1514     0.0024
        nfcorpus      3  0.0  323 0.3064      0.3427  0.3419  0.3915     0.4174   1.0000 0.0355   0.0245   0.0479     0.0024
            fiqa      3  0.0  648 0.2326      0.2735  0.2732  0.3409     0.3754   0.9892 0.0407   0.0291   0.0529     0.0024
webis-touche2020      3  0.0   49 0.3579      0.4414  0.4414  0.4710     0.7383   1.0000 0.0835   0.0500   0.1178     0.0024
             ALL      3  0.0 1370 0.3616      0.4047  0.4044  0.4645     0.4152   0.9927 0.0427   0.0350   0.0505     0.0024
```

Null control (features shuffled), ALL: recovered=0.418

Failure sample L3: harmful_go=15, wrong_family=15, success=12, missed_go=4

## Protocol `lodo`, model `gbr`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
    lodo   gbr     1    go_nogo           all 1370          0.453         0.547     0.056 0.500
    lodo   gbr     1    go_nogo oracle_routed 1370          0.453         0.547     0.056 0.500
    lodo   gbr     2 llm_vs_prf           all 1370          0.488         0.512    -0.023 0.527
    lodo   gbr     2 llm_vs_prf oracle_routed  621          0.718         0.718    -0.065 0.481
    lodo   gbr     3 llm_choice           all 1370          0.684         0.685     0.012   NaN
    lodo   gbr     3 llm_choice oracle_routed  446          0.395         0.392     0.028   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6617      0.6982  0.7536  0.7618     0.9179      1.0 0.0919   0.0690   0.1168     0.0024
      trec-covid      1  0.0   50 0.5945      0.7071  0.7368  0.7482     0.9258      1.0 0.1423   0.0934   0.1927     0.0024
        nfcorpus      1  0.0  323 0.3064      0.3427  0.3891  0.3915     0.9717      1.0 0.0827   0.0683   0.0979     0.0024
            fiqa      1  0.0  648 0.2326      0.2735  0.3298  0.3409     0.8976      1.0 0.0972   0.0820   0.1136     0.0024
webis-touche2020      1  0.0   49 0.3579      0.4414  0.4658  0.4710     0.9541      1.0 0.1079   0.0723   0.1442     0.0024
             ALL      1  0.0 1370 0.3616      0.4047  0.4563  0.4645     0.9202      1.0 0.0947   0.0845   0.1047     0.0024
```

Null control (features shuffled), ALL: recovered=0.920

Failure sample L1: harmful_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6617      0.6982  0.7400  0.7618     0.7819      1.0 0.0783   0.0557   0.1026     0.0024
      trec-covid      2  0.0   50 0.5945      0.7071  0.7143  0.7482     0.7791      1.0 0.1197   0.0704   0.1730     0.0024
        nfcorpus      2  0.0  323 0.3064      0.3427  0.3602  0.3915     0.6324      1.0 0.0538   0.0405   0.0687     0.0024
            fiqa      2  0.0  648 0.2326      0.2735  0.3156  0.3409     0.7668      1.0 0.0831   0.0685   0.0993     0.0024
webis-touche2020      2  0.0   49 0.3579      0.4414  0.4348  0.4710     0.6800      1.0 0.0769   0.0405   0.1134     0.0024
             ALL      2  0.0 1370 0.3616      0.4047  0.4379  0.4645     0.7411      1.0 0.0762   0.0662   0.0862     0.0024
```

Null control (features shuffled), ALL: recovered=0.741

Failure sample L2: harmful_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6617      0.6982  0.6985  0.7618     0.3675      1.0 0.0368   0.0201   0.0556     0.0024
      trec-covid      3  0.0   50 0.5945      0.7071  0.7071  0.7482     0.7321      1.0 0.1125   0.0722   0.1543     0.0024
        nfcorpus      3  0.0  323 0.3064      0.3427  0.3427  0.3915     0.4267      1.0 0.0363   0.0252   0.0487     0.0024
            fiqa      3  0.0  648 0.2326      0.2735  0.2736  0.3409     0.3791      1.0 0.0411   0.0296   0.0532     0.0024
webis-touche2020      3  0.0   49 0.3579      0.4414  0.4414  0.4710     0.7383      1.0 0.0835   0.0500   0.1178     0.0024
             ALL      3  0.0 1370 0.3616      0.4047  0.4048  0.4645     0.4193      1.0 0.0431   0.0355   0.0507     0.0024
```

Null control (features shuffled), ALL: recovered=0.419

Failure sample L3: harmful_go=15, wrong_family=15, success=12
