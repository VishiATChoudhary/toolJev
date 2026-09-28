
### agent_mcptb_big_haiku

| condition | n | right tool (strict) | right tool (lenient) | arg accuracy | median input tokens | total cost | mean LLM calls | median wall s | timeouts |
|---|---|---|---|---|---|---|---|---|---|
| native | 36 | 0.750 | 0.806 | 0.796 | 51,660 | $1.60 | 6.3 | 18 | 0 |
| tooljev | 36 | 0.611 | 0.806 | 0.770 | 20,236 | $1.23 | 7.6 | 32 | 1 |
| direct | 12 | 0.833 | 0.833 | 0.883 | 586,835 | $0.98 | 4.2 | 16 | 0 |

### agent_mcptb_haiku

| condition | n | right tool (strict) | right tool (lenient) | arg accuracy | median input tokens | total cost | mean LLM calls | median wall s | timeouts |
|---|---|---|---|---|---|---|---|---|---|
| direct | 36 | 0.917 | 0.944 | 0.794 | 59,304 | $0.63 | 3.1 | 11 | 0 |
| tooljev | 36 | 0.861 | 0.889 | 0.804 | 20,376 | $1.43 | 7.6 | 31 | 0 |

### agent_triage400_haiku

| condition | reps | accuracy (each rep) | mean accuracy | mean cost | mean LLM calls | mean wall s |
|---|---|---|---|---|---|---|
| direct | 2 | 0.993, 0.993 | 0.993 | $0.332 | 7.5 | 268 |
| tooljev | 2 | 0.805, 0.672 | 0.739 | $0.091 | 11.5 | 98 |
| tooljev-hint | 2 | 0.912, 0.833 | 0.873 | $0.064 | 10.5 | 122 |
