# Router iteration log

- datasets: ['scifact', 'trec-covid', 'nfcorpus', 'fiqa', 'webis-touche2020']  | retriever: dense | metric: ndcg10 | queries: 1370 (dropped for missing values: 0)
- seeds used for cross-fit oracle: [0, 1, 2] | oracle: crossfit | features: stage<=1 (34 usable)
- lambda grid (nDCG per second): [0.0, 0.002, 0.005, 0.01, 0.02, 0.05] | folds 5 x repeats 3 | bootstrap B=5000

## Baselines (nDCG@10)

```
         dataset    n  fixed_a0  fixed_prf  fixed_query2doc  fixed_hyde  fixed_decomp  oracle_crossfit  oracle_naive  go_headroom_crossfit  oracle_go_rate
         scifact  300    0.6451     0.6062           0.6910      0.6839        0.5887           0.7702        0.7747                0.1251          0.3800
      trec-covid   50    0.4725     0.5813           0.6535      0.6055        0.4573           0.7170        0.7240                0.2446          0.9600
        nfcorpus  323    0.3159     0.3216           0.3352      0.3323        0.3063           0.4251        0.4275                0.1091          0.5975
            fiqa  648    0.3687     0.3242           0.3751      0.3384        0.2986           0.4561        0.4675                0.0874          0.4583
webis-touche2020   49    0.1690     0.1917           0.2606      0.2403        0.1757           0.2967        0.3086                0.1276          0.8571
             ALL 1370    0.4134     0.3900           0.4409      0.4189        0.3653           0.5214        0.5290                0.1080          0.5066
```

## Protocol `within`, model `const`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
  within const     1    go_nogo           all 1370          0.468         0.532     0.095 0.542
  within const     1    go_nogo oracle_routed 1370          0.468         0.532     0.095 0.542
  within const     2 llm_vs_prf           all 1370          0.453         0.547    -0.013 0.449
  within const     2 llm_vs_prf oracle_routed  641          0.613         0.613     0.026 0.491
  within const     3 llm_choice           all 1370          0.655         0.655     0.081   NaN
  within const     3 llm_choice oracle_routed  393          0.468         0.468     0.063   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6451      0.6910  0.7555  0.7702     0.8825      1.0 0.1104   0.0850   0.1367     0.0024
      trec-covid      1  0.0   50 0.4725      0.6535  0.7163  0.7170     0.9969      1.0 0.2438   0.1961   0.2956     0.0024
        nfcorpus      1  0.0  323 0.3159      0.3300  0.4169  0.4251     0.9250      1.0 0.1010   0.0827   0.1204     0.0024
            fiqa      1  0.0  648 0.3687      0.3727  0.4326  0.4561     0.7317      1.0 0.0640   0.0500   0.0782     0.0024
webis-touche2020      1  0.0   49 0.1690      0.2606  0.2915  0.2967     0.9597      1.0 0.1225   0.0891   0.1581     0.0024
             ALL      1  0.0 1370 0.4134      0.4386  0.5049  0.5214     0.8476      1.0 0.0915   0.0809   0.1022     0.0024
```

Failure sample L1: harmful_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6451      0.6910  0.7194  0.7702     0.5943      1.0 0.0744   0.0506   0.0995     0.0024
      trec-covid      2  0.0   50 0.4725      0.6535  0.6636  0.7170     0.7816      1.0 0.1912   0.1398   0.2461     0.0024
        nfcorpus      2  0.0  323 0.3159      0.3300  0.3576  0.4251     0.3816      1.0 0.0417   0.0275   0.0563     0.0024
            fiqa      2  0.0  648 0.3687      0.3727  0.3999  0.4561     0.3574      1.0 0.0312   0.0172   0.0447     0.0024
webis-touche2020      2  0.0   49 0.1690      0.2606  0.2759  0.2967     0.8371      1.0 0.1068   0.0713   0.1446     0.0024
             ALL      2  0.0 1370 0.4134      0.4386  0.4651  0.5214     0.4786      1.0 0.0517   0.0419   0.0613     0.0024
```

Failure sample L2: harmful_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6451      0.6910  0.6910  0.7702     0.3671      1.0 0.0459   0.0222   0.0722     0.0024
      trec-covid      3  0.0   50 0.4725      0.6535  0.6535  0.7170     0.7401      1.0 0.1810   0.1269   0.2383     0.0024
        nfcorpus      3  0.0  323 0.3159      0.3300  0.3352  0.4251     0.1768      1.0 0.0193   0.0044   0.0347     0.0616
            fiqa      3  0.0  648 0.3687      0.3727  0.3751  0.4561     0.0736      1.0 0.0064  -0.0062   0.0189     1.0000
webis-touche2020      3  0.0   49 0.1690      0.2606  0.2606  0.2967     0.7172      1.0 0.0916   0.0554   0.1301     0.0024
             ALL      3  0.0 1370 0.4134      0.4386  0.4409  0.5214     0.2550      1.0 0.0275   0.0178   0.0370     0.0024
```

Failure sample L3: harmful_go=15, wrong_family=15, success=12

## Protocol `within`, model `stump`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
  within stump     1    go_nogo           all 1370          0.492         0.532     0.061 0.530
  within stump     1    go_nogo oracle_routed 1370          0.492         0.532     0.061 0.530
  within stump     2 llm_vs_prf           all 1370          0.453         0.547    -0.013 0.449
  within stump     2 llm_vs_prf oracle_routed  641          0.613         0.613     0.026 0.491
  within stump     3 llm_choice           all 1370          0.655         0.655     0.081   NaN
  within stump     3 llm_choice oracle_routed  393          0.468         0.468     0.063   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6451      0.6910  0.7545  0.7702     0.8748   0.9133 0.1094   0.0839   0.1356     0.0024
      trec-covid      1  0.0   50 0.4725      0.6535  0.7041  0.7170     0.9470   0.9267 0.2316   0.1823   0.2847     0.0024
        nfcorpus      1  0.0  323 0.3159      0.3300  0.4104  0.4251     0.8655   0.8999 0.0945   0.0767   0.1135     0.0024
            fiqa      1  0.0  648 0.3687      0.3727  0.4269  0.4561     0.6658   0.8971 0.0582   0.0449   0.0718     0.0024
webis-touche2020      1  0.0   49 0.1690      0.2606  0.2677  0.2967     0.7729   0.8639 0.0987   0.0698   0.1292     0.0024
             ALL      1  0.0 1370 0.4134      0.4386  0.4992  0.5214     0.7942   0.9012 0.0857   0.0754   0.0959     0.0024
```

Failure sample L1: harmful_go=15, missed_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6451      0.6910  0.7200  0.7702     0.5988   0.9133 0.0749   0.0512   0.0999     0.0024
      trec-covid      2  0.0   50 0.4725      0.6535  0.6553  0.7170     0.7478   0.9267 0.1829   0.1307   0.2395     0.0024
        nfcorpus      2  0.0  323 0.3159      0.3300  0.3546  0.4251     0.3540   0.8999 0.0386   0.0255   0.0526     0.0024
            fiqa      2  0.0  648 0.3687      0.3727  0.3971  0.4561     0.3257   0.8971 0.0285   0.0149   0.0415     0.0024
webis-touche2020      2  0.0   49 0.1690      0.2606  0.2558  0.2967     0.6799   0.8639 0.0868   0.0559   0.1193     0.0024
             ALL      2  0.0 1370 0.4134      0.4386  0.4622  0.5214     0.4516   0.9012 0.0488   0.0393   0.0580     0.0024
```

Failure sample L2: harmful_go=15, missed_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6451      0.6910  0.6913  0.7702     0.3694   0.9133 0.0462   0.0224   0.0724     0.0024
      trec-covid      3  0.0   50 0.4725      0.6535  0.6478  0.7170     0.7170   0.9267 0.1753   0.1212   0.2327     0.0024
        nfcorpus      3  0.0  323 0.3159      0.3300  0.3329  0.4251     0.1557   0.8999 0.0170   0.0028   0.0317     0.1316
            fiqa      3  0.0  648 0.3687      0.3727  0.3733  0.4561     0.0530   0.8971 0.0046  -0.0076   0.0167     1.0000
webis-touche2020      3  0.0   49 0.1690      0.2606  0.2405  0.2967     0.5601   0.8639 0.0715   0.0413   0.1030     0.0024
             ALL      3  0.0 1370 0.4134      0.4386  0.4387  0.5214     0.2341   0.9012 0.0253   0.0158   0.0345     0.0024
```

Failure sample L3: harmful_go=15, missed_go=15, wrong_family=15, success=12

## Protocol `within`, model `ridge`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
  within ridge     1    go_nogo           all 1370          0.476         0.532     0.208 0.595
  within ridge     1    go_nogo oracle_routed 1370          0.476         0.532     0.208 0.595
  within ridge     2 llm_vs_prf           all 1370          0.490         0.547     0.128 0.545
  within ridge     2 llm_vs_prf oracle_routed  641          0.640         0.613     0.184 0.582
  within ridge     3 llm_choice           all 1370          0.601         0.655     0.163   NaN
  within ridge     3 llm_choice oracle_routed  393          0.427         0.468     0.161   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6451      0.6910  0.7555  0.7702     0.8823   0.9500 0.1104   0.0850   0.1366     0.0024
      trec-covid      1  0.0   50 0.4725      0.6535  0.7163  0.7170     0.9969   1.0000 0.2438   0.1961   0.2956     0.0024
        nfcorpus      1  0.0  323 0.3159      0.3300  0.4169  0.4251     0.9250   1.0000 0.1010   0.0827   0.1204     0.0024
            fiqa      1  0.0  648 0.3687      0.3727  0.4323  0.4561     0.7282   0.9902 0.0637   0.0498   0.0780     0.0024
webis-touche2020      1  0.0   49 0.1690      0.2606  0.2828  0.2967     0.8912   0.9048 0.1138   0.0810   0.1489     0.0024
             ALL      1  0.0 1370 0.4134      0.4386  0.5045  0.5214     0.8433   0.9810 0.0910   0.0805   0.1017     0.0024
```

Null control (features shuffled), ALL: recovered=0.841

Failure sample L1: harmful_go=15, success=12, missed_go=4

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6451      0.6910  0.7162  0.7702     0.5688   0.9500 0.0712   0.0475   0.0964     0.0024
      trec-covid      2  0.0   50 0.4725      0.6535  0.6622  0.7170     0.7757   1.0000 0.1897   0.1385   0.2443     0.0024
        nfcorpus      2  0.0  323 0.3159      0.3300  0.3712  0.4251     0.5059   1.0000 0.0552   0.0382   0.0736     0.0024
            fiqa      2  0.0  648 0.3687      0.3727  0.3995  0.4561     0.3523   0.9902 0.0308   0.0166   0.0442     0.0024
webis-touche2020      2  0.0   49 0.1690      0.2606  0.2494  0.2967     0.6294   0.9048 0.0803   0.0519   0.1115     0.0024
             ALL      2  0.0 1370 0.4134      0.4386  0.4664  0.5214     0.4906   0.9810 0.0530   0.0428   0.0628     0.0024
```

Null control (features shuffled), ALL: recovered=0.464

Failure sample L2: harmful_go=15, wrong_family=15, success=12, missed_go=4

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6451      0.6910  0.6878  0.7702     0.3414   0.9500 0.0427   0.0186   0.0688     0.0024
      trec-covid      3  0.0   50 0.4725      0.6535  0.6518  0.7170     0.7333   1.0000 0.1793   0.1275   0.2359     0.0024
        nfcorpus      3  0.0  323 0.3159      0.3300  0.3569  0.4251     0.3750   1.0000 0.0409   0.0234   0.0597     0.0024
            fiqa      3  0.0  648 0.3687      0.3727  0.3749  0.4561     0.0709   0.9902 0.0062  -0.0064   0.0187     0.7008
webis-touche2020      3  0.0   49 0.1690      0.2606  0.2261  0.2967     0.4471   0.9048 0.0571   0.0273   0.0897     0.0024
             ALL      3  0.0 1370 0.4134      0.4386  0.4439  0.5214     0.2827   0.9810 0.0305   0.0203   0.0403     0.0024
```

Null control (features shuffled), ALL: recovered=0.242

Failure sample L3: harmful_go=15, wrong_family=15, success=12, missed_go=4

## Protocol `within`, model `gbr`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
  within   gbr     1    go_nogo           all 1370          0.468         0.532     0.223 0.628
  within   gbr     1    go_nogo oracle_routed 1370          0.468         0.532     0.223 0.628
  within   gbr     2 llm_vs_prf           all 1370          0.477         0.547     0.127 0.545
  within   gbr     2 llm_vs_prf oracle_routed  641          0.644         0.613     0.190 0.598
  within   gbr     3 llm_choice           all 1370          0.577         0.655     0.163   NaN
  within   gbr     3 llm_choice oracle_routed  393          0.425         0.468     0.171   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6451      0.6910  0.7554  0.7702     0.8816   0.9967 0.1103   0.0849   0.1365     0.0024
      trec-covid      1  0.0   50 0.4725      0.6535  0.7163  0.7170     0.9969   1.0000 0.2438   0.1961   0.2956     0.0024
        nfcorpus      1  0.0  323 0.3159      0.3300  0.4169  0.4251     0.9250   1.0000 0.1010   0.0827   0.1204     0.0024
            fiqa      1  0.0  648 0.3687      0.3727  0.4326  0.4561     0.7312   0.9990 0.0639   0.0500   0.0782     0.0024
webis-touche2020      1  0.0   49 0.1690      0.2606  0.2915  0.2967     0.9597   1.0000 0.1225   0.0891   0.1581     0.0024
             ALL      1  0.0 1370 0.4134      0.4386  0.5049  0.5214     0.8472   0.9988 0.0915   0.0809   0.1022     0.0024
```

Null control (features shuffled), ALL: recovered=0.846

Failure sample L1: harmful_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6451      0.6910  0.7193  0.7702     0.5934   0.9967 0.0742   0.0503   0.0994     0.0024
      trec-covid      2  0.0   50 0.4725      0.6535  0.6621  0.7170     0.7753   1.0000 0.1896   0.1369   0.2459     0.0024
        nfcorpus      2  0.0  323 0.3159      0.3300  0.3735  0.4251     0.5271   1.0000 0.0575   0.0400   0.0761     0.0024
            fiqa      2  0.0  648 0.3687      0.3727  0.3993  0.4561     0.3500   0.9990 0.0306   0.0165   0.0437     0.0024
webis-touche2020      2  0.0   49 0.1690      0.2606  0.2748  0.2967     0.8286   1.0000 0.1058   0.0702   0.1439     0.0024
             ALL      2  0.0 1370 0.4134      0.4386  0.4684  0.5214     0.5094   0.9988 0.0550   0.0447   0.0650     0.0024
```

Null control (features shuffled), ALL: recovered=0.458

Failure sample L2: harmful_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6451      0.6910  0.6879  0.7702     0.3419   0.9967 0.0428   0.0191   0.0687     0.0024
      trec-covid      3  0.0   50 0.4725      0.6535  0.6422  0.7170     0.6941   1.0000 0.1697   0.1154   0.2278     0.0024
        nfcorpus      3  0.0  323 0.3159      0.3300  0.3588  0.4251     0.3931   1.0000 0.0429   0.0250   0.0618     0.0024
            fiqa      3  0.0  648 0.3687      0.3727  0.3748  0.4561     0.0698   0.9990 0.0061  -0.0065   0.0185     0.4416
webis-touche2020      3  0.0   49 0.1690      0.2606  0.2559  0.2967     0.6806   1.0000 0.0869   0.0519   0.1240     0.0024
             ALL      3  0.0 1370 0.4134      0.4386  0.4451  0.5214     0.2933   0.9988 0.0317   0.0214   0.0417     0.0024
```

Null control (features shuffled), ALL: recovered=0.247

Failure sample L3: harmful_go=15, wrong_family=15, success=12

## Protocol `lodo`, model `const`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
    lodo const     1    go_nogo           all 1370          0.468         0.532    -0.138 0.456
    lodo const     1    go_nogo oracle_routed 1370          0.468         0.532    -0.138 0.456
    lodo const     2 llm_vs_prf           all 1370          0.453         0.547    -0.056 0.538
    lodo const     2 llm_vs_prf oracle_routed  641          0.613         0.613    -0.141 0.467
    lodo const     3 llm_choice           all 1370          0.655         0.655    -0.119   NaN
    lodo const     3 llm_choice oracle_routed  393          0.468         0.468    -0.098   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6451      0.6910  0.7555  0.7702     0.8825      1.0 0.1104   0.0850   0.1367     0.0024
      trec-covid      1  0.0   50 0.4725      0.6535  0.7163  0.7170     0.9969      1.0 0.2438   0.1961   0.2956     0.0024
        nfcorpus      1  0.0  323 0.3159      0.3352  0.4169  0.4251     0.9250      1.0 0.1010   0.0827   0.1204     0.0024
            fiqa      1  0.0  648 0.3687      0.3751  0.4326  0.4561     0.7317      1.0 0.0640   0.0500   0.0782     0.0024
webis-touche2020      1  0.0   49 0.1690      0.2606  0.2915  0.2967     0.9597      1.0 0.1225   0.0891   0.1581     0.0024
             ALL      1  0.0 1370 0.4134      0.4409  0.5049  0.5214     0.8476      1.0 0.0915   0.0809   0.1022     0.0024
```

Failure sample L1: harmful_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6451      0.6910  0.7194  0.7702     0.5943      1.0 0.0744   0.0506   0.0995     0.0024
      trec-covid      2  0.0   50 0.4725      0.6535  0.6636  0.7170     0.7816      1.0 0.1912   0.1398   0.2461     0.0024
        nfcorpus      2  0.0  323 0.3159      0.3352  0.3576  0.4251     0.3816      1.0 0.0417   0.0275   0.0563     0.0024
            fiqa      2  0.0  648 0.3687      0.3751  0.3999  0.4561     0.3574      1.0 0.0312   0.0172   0.0447     0.0024
webis-touche2020      2  0.0   49 0.1690      0.2606  0.2759  0.2967     0.8371      1.0 0.1068   0.0713   0.1446     0.0024
             ALL      2  0.0 1370 0.4134      0.4409  0.4651  0.5214     0.4786      1.0 0.0517   0.0419   0.0613     0.0024
```

Failure sample L2: harmful_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6451      0.6910  0.6910  0.7702     0.3671      1.0 0.0459   0.0222   0.0722     0.0024
      trec-covid      3  0.0   50 0.4725      0.6535  0.6535  0.7170     0.7401      1.0 0.1810   0.1269   0.2383     0.0024
        nfcorpus      3  0.0  323 0.3159      0.3352  0.3352  0.4251     0.1768      1.0 0.0193   0.0044   0.0347     0.0704
            fiqa      3  0.0  648 0.3687      0.3751  0.3751  0.4561     0.0736      1.0 0.0064  -0.0062   0.0189     1.0000
webis-touche2020      3  0.0   49 0.1690      0.2606  0.2606  0.2967     0.7172      1.0 0.0916   0.0554   0.1301     0.0024
             ALL      3  0.0 1370 0.4134      0.4409  0.4409  0.5214     0.2550      1.0 0.0275   0.0178   0.0370     0.0024
```

Failure sample L3: harmful_go=15, wrong_family=15, success=12

## Protocol `lodo`, model `stump`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
    lodo stump     1    go_nogo           all 1370          0.474         0.532     0.020 0.500
    lodo stump     1    go_nogo oracle_routed 1370          0.474         0.532     0.020 0.500
    lodo stump     2 llm_vs_prf           all 1370          0.453         0.547    -0.056 0.538
    lodo stump     2 llm_vs_prf oracle_routed  641          0.613         0.613    -0.141 0.467
    lodo stump     3 llm_choice           all 1370          0.655         0.655    -0.119   NaN
    lodo stump     3 llm_choice oracle_routed  393          0.468         0.468    -0.098   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6451      0.6910  0.7535  0.7702     0.8670   0.9133 0.1085   0.0829   0.1348     0.0024
      trec-covid      1  0.0   50 0.4725      0.6535  0.7005  0.7170     0.9323   0.8800 0.2280   0.1766   0.2828     0.0024
        nfcorpus      1  0.0  323 0.3159      0.3352  0.4055  0.4251     0.8209   0.9071 0.0896   0.0716   0.1091     0.0024
            fiqa      1  0.0  648 0.3687      0.3751  0.4282  0.4561     0.6809   0.8935 0.0595   0.0457   0.0731     0.0024
webis-touche2020      1  0.0   49 0.1690      0.2606  0.2915  0.2967     0.9597   0.9388 0.1225   0.0891   0.1581     0.0024
             ALL      1  0.0 1370 0.4134      0.4409  0.4991  0.5214     0.7940   0.9022 0.0857   0.0755   0.0961     0.0024
```

Failure sample L1: harmful_go=15, missed_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6451      0.6910  0.7207  0.7702     0.6047   0.9133 0.0756   0.0525   0.1003     0.0024
      trec-covid      2  0.0   50 0.4725      0.6535  0.6522  0.7170     0.7349   0.8800 0.1797   0.1264   0.2380     0.0024
        nfcorpus      2  0.0  323 0.3159      0.3352  0.3537  0.4251     0.3463   0.9071 0.0378   0.0242   0.0525     0.0024
            fiqa      2  0.0  648 0.3687      0.3751  0.3978  0.4561     0.3338   0.8935 0.0292   0.0157   0.0421     0.0024
webis-touche2020      2  0.0   49 0.1690      0.2606  0.2759  0.2967     0.8371   0.9388 0.1068   0.0713   0.1446     0.0024
             ALL      2  0.0 1370 0.4134      0.4409  0.4631  0.5214     0.4599   0.9022 0.0497   0.0401   0.0590     0.0024
```

Failure sample L2: harmful_go=15, missed_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6451      0.6910  0.6944  0.7702     0.3939   0.9133 0.0493   0.0256   0.0752     0.0024
      trec-covid      3  0.0   50 0.4725      0.6535  0.6419  0.7170     0.6929   0.8800 0.1695   0.1139   0.2279     0.0024
        nfcorpus      3  0.0  323 0.3159      0.3352  0.3328  0.4251     0.1547   0.9071 0.0169   0.0025   0.0319     0.1316
            fiqa      3  0.0  648 0.3687      0.3751  0.3745  0.4561     0.0663   0.8935 0.0058  -0.0065   0.0182     1.0000
webis-touche2020      3  0.0   49 0.1690      0.2606  0.2606  0.2967     0.7172   0.9388 0.0916   0.0554   0.1301     0.0024
             ALL      3  0.0 1370 0.4134      0.4409  0.4404  0.5214     0.2498   0.9022 0.0270   0.0174   0.0363     0.0024
```

Failure sample L3: harmful_go=15, missed_go=15, wrong_family=15, success=12

## Protocol `lodo`, model `ridge`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
    lodo ridge     1    go_nogo           all 1370          0.469         0.532     0.022 0.492
    lodo ridge     1    go_nogo oracle_routed 1370          0.469         0.532     0.022 0.492
    lodo ridge     2 llm_vs_prf           all 1370          0.464         0.547     0.077 0.540
    lodo ridge     2 llm_vs_prf oracle_routed  641          0.615         0.613     0.096 0.553
    lodo ridge     3 llm_choice           all 1370          0.636         0.655    -0.044   NaN
    lodo ridge     3 llm_choice oracle_routed  393          0.471         0.468     0.055   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6451      0.6910  0.7555  0.7702     0.8825   1.0000 0.1104   0.0850   0.1367     0.0024
      trec-covid      1  0.0   50 0.4725      0.6535  0.7163  0.7170     0.9969   1.0000 0.2438   0.1961   0.2956     0.0024
        nfcorpus      1  0.0  323 0.3159      0.3352  0.4150  0.4251     0.9080   0.9876 0.0991   0.0806   0.1186     0.0024
            fiqa      1  0.0  648 0.3687      0.3751  0.4326  0.4561     0.7317   0.9985 0.0640   0.0500   0.0782     0.0024
webis-touche2020      1  0.0   49 0.1690      0.2606  0.2915  0.2967     0.9597   1.0000 0.1225   0.0891   0.1581     0.0024
             ALL      1  0.0 1370 0.4134      0.4409  0.5045  0.5214     0.8435   0.9964 0.0911   0.0806   0.1017     0.0024
```

Null control (features shuffled), ALL: recovered=0.848

Failure sample L1: harmful_go=15, success=12, missed_go=2

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6451      0.6910  0.7201  0.7702     0.5994   1.0000 0.0750   0.0513   0.1002     0.0024
      trec-covid      2  0.0   50 0.4725      0.6535  0.6515  0.7170     0.7320   1.0000 0.1790   0.1258   0.2362     0.0024
        nfcorpus      2  0.0  323 0.3159      0.3352  0.3581  0.4251     0.3860   0.9876 0.0421   0.0281   0.0568     0.0024
            fiqa      2  0.0  648 0.3687      0.3751  0.3973  0.4561     0.3278   0.9985 0.0287   0.0145   0.0423     0.0024
webis-touche2020      2  0.0   49 0.1690      0.2606  0.2727  0.2967     0.8120   1.0000 0.1036   0.0678   0.1419     0.0024
             ALL      2  0.0 1370 0.4134      0.4409  0.4636  0.5214     0.4645   0.9964 0.0501   0.0403   0.0599     0.0024
```

Null control (features shuffled), ALL: recovered=0.479

Failure sample L2: harmful_go=15, wrong_family=15, success=12, missed_go=2

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6451      0.6910  0.6916  0.7702     0.3722   1.0000 0.0466   0.0225   0.0727     0.0024
      trec-covid      3  0.0   50 0.4725      0.6535  0.6394  0.7170     0.6828   1.0000 0.1670   0.1110   0.2256     0.0024
        nfcorpus      3  0.0  323 0.3159      0.3352  0.3357  0.4251     0.1812   0.9876 0.0198   0.0050   0.0352     0.0480
            fiqa      3  0.0  648 0.3687      0.3751  0.3707  0.4561     0.0238   0.9985 0.0021  -0.0114   0.0156     1.0000
webis-touche2020      3  0.0   49 0.1690      0.2606  0.2588  0.2967     0.7028   1.0000 0.0897   0.0532   0.1284     0.0024
             ALL      3  0.0 1370 0.4134      0.4409  0.4386  0.5214     0.2329   0.9964 0.0251   0.0152   0.0349     0.0024
```

Null control (features shuffled), ALL: recovered=0.237

Failure sample L3: harmful_go=15, wrong_family=15, success=12, missed_go=2

## Protocol `lodo`, model `gbr`

Node diagnostics (does the predicted gain track the true gain?):
```
protocol model  node       name         scope    n  acc_vs_oracle  acc_majority  spearman   auc
    lodo   gbr     1    go_nogo           all 1370          0.468         0.532     0.058 0.502
    lodo   gbr     1    go_nogo oracle_routed 1370          0.468         0.532     0.058 0.502
    lodo   gbr     2 llm_vs_prf           all 1370          0.458         0.547     0.072 0.537
    lodo   gbr     2 llm_vs_prf oracle_routed  641          0.613         0.613     0.081 0.540
    lodo   gbr     3 llm_choice           all 1370          0.599         0.655     0.022   NaN
    lodo   gbr     3 llm_choice oracle_routed  393          0.435         0.468     0.105   NaN
```

### Iteration 1 (go/no-go; oracle below)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      1  0.0  300 0.6451      0.6910  0.7555  0.7702     0.8825      1.0 0.1104   0.0850   0.1367     0.0024
      trec-covid      1  0.0   50 0.4725      0.6535  0.7163  0.7170     0.9969      1.0 0.2438   0.1961   0.2956     0.0024
        nfcorpus      1  0.0  323 0.3159      0.3352  0.4169  0.4251     0.9250      1.0 0.1010   0.0827   0.1204     0.0024
            fiqa      1  0.0  648 0.3687      0.3751  0.4326  0.4561     0.7317      1.0 0.0640   0.0500   0.0782     0.0024
webis-touche2020      1  0.0   49 0.1690      0.2606  0.2915  0.2967     0.9597      1.0 0.1225   0.0891   0.1581     0.0024
             ALL      1  0.0 1370 0.4134      0.4409  0.5049  0.5214     0.8476      1.0 0.0915   0.0809   0.1022     0.0024
```

Null control (features shuffled), ALL: recovered=0.848

Failure sample L1: harmful_go=15, success=12

### Iteration 2 (adds LLM-vs-PRF; oracle among LLM actions)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      2  0.0  300 0.6451      0.6910  0.7180  0.7702     0.5828      1.0 0.0729   0.0490   0.0984     0.0024
      trec-covid      2  0.0   50 0.4725      0.6535  0.6629  0.7170     0.7787      1.0 0.1904   0.1385   0.2455     0.0024
        nfcorpus      2  0.0  323 0.3159      0.3352  0.3576  0.4251     0.3816      1.0 0.0417   0.0275   0.0563     0.0024
            fiqa      2  0.0  648 0.3687      0.3751  0.4004  0.4561     0.3626      1.0 0.0317   0.0177   0.0453     0.0024
webis-touche2020      2  0.0   49 0.1690      0.2606  0.2746  0.2967     0.8268      1.0 0.1055   0.0703   0.1434     0.0024
             ALL      2  0.0 1370 0.4134      0.4409  0.4649  0.5214     0.4770      1.0 0.0515   0.0417   0.0611     0.0024
```

Null control (features shuffled), ALL: recovered=0.476

Failure sample L2: harmful_go=15, wrong_family=15, success=12

### Iteration 3 (adds LLM-action choice; no oracle)

```
         dataset  level  lam    n     a0  best_fixed  policy  oracle  recovered  go_rate   d_a0  d_a0_lo  d_a0_hi  p_a0_holm
         scifact      3  0.0  300 0.6451      0.6910  0.6907  0.7702     0.3645      1.0 0.0456   0.0212   0.0721     0.0024
      trec-covid      3  0.0   50 0.4725      0.6535  0.6554  0.7170     0.7480      1.0 0.1829   0.1292   0.2395     0.0024
        nfcorpus      3  0.0  323 0.3159      0.3352  0.3338  0.4251     0.1633      1.0 0.0178   0.0029   0.0332     0.1184
            fiqa      3  0.0  648 0.3687      0.3751  0.3746  0.4561     0.0679      1.0 0.0059  -0.0067   0.0189     1.0000
webis-touche2020      3  0.0   49 0.1690      0.2606  0.2627  0.2967     0.7339      1.0 0.0937   0.0584   0.1310     0.0024
             ALL      3  0.0 1370 0.4134      0.4409  0.4404  0.5214     0.2503      1.0 0.0270   0.0172   0.0366     0.0024
```

Null control (features shuffled), ALL: recovered=0.250

Failure sample L3: harmful_go=15, wrong_family=15, success=12
