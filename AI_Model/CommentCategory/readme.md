macro 0.55, micro 0.9, should be ok, used 50k data only due to comp limitation, if got gpu can help rerun it to achieve higher macro and micro.



# Classification Model Tab
comp["delta"] = comp["f1_tuned"] - comp["f1_default"]
f1_default = F1 score of a label using the standard threshold = 0.5.

f1_tuned = F1 score of the same label after applying its optimal threshold (the best cutoff found during tuning).

delta = The difference between these two.

👉 So delta measures the performance improvement (or drop) gained by tuning thresholds for each category.

