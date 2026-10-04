# A small glossary for the first run

| Term | Plain-language meaning | Example |
| --- | --- | --- |
| **PD** | Probability of the default/event definition you supplied, over its chosen horizon | `0.02` means a model probability of 2%, not a guaranteed observed rate |
| **LGD** | Fraction of exposure lost if default occurs | 40% loss after recoveries means `lgd=0.4` |
| **EAD** | Exposure amount when default occurs | PD × LGD × EAD gives a simple expected-loss calculation |
| **Binning** | Replacing raw numerical values with contiguous groups | Debt ratios 0.2 and 0.3 may belong to the same risk group |
| **WoE** | Log ratio of a bin's share of non-events to its share of events, with the package's smoothing | Positive values mean relative concentration of non-events; negative values mean events |
| **IV** | A measure of how strongly bins separate events and non-events | The solver maximizes regular-bin IV within a constrained prebin search space; it is not model validation |
| **Monotonic rate** | A rate that only moves in one direction across ordered bins | Ascending default rates cannot fall when moving to a higher feature bin |
| **PDO** | Points to double the good:bad odds | PDO 20: moving from 600 to 620 points doubles good:bad odds |
| **Factor rate** | A fixed multiple of principal defining the total receivable | 30,000 × 1.12 = 33,600; it is not 12% annual interest |
| **Holdback** | Share of eligible daily revenue used for repayment | 10% of 1,000 revenue is a 100 payment, capped at the unpaid balance |
| **MCA / RBF** | Common names for financing tied to business revenue; agreements differ | Supply the repayment mechanics from the agreement being analyzed |
| **Payment floor** | Minimum payment amount per complete day block | A 30-day floor adds a top-up if revenue payments fall short |
| **Milestone** | A check on cumulative repayment at a specified day | A day-180 checkpoint reports a shortfall without automatically collecting it |
| **Horizon** | Where the supplied future path ends | A balance at day 360 means the scenario has not repaid within 360 days |
| **Liquidity shortfall** | Cash goes below zero under the scenario's outflows | Indicates additional funding is needed if scheduled payments execute |
| **PSI** | Population Stability Index: a binned comparison of two distributions | Compare a current feature population with a reference population; it does not identify the cause of drift |
| **XIRR** | Effective annual return implied by dated conventional cash flows | Faster repayment of a fixed receivable changes its annualized return |

For exact formulas and edge cases, use [calculation conventions](conventions.md).
For a complete example, start with [financing](tutorials/revenue-financing.md)
or [scorecards](tutorials/scorecards.md).
