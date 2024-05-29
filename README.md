# gerikl
Repository with libraries created during course work by gerikl team&

Example of ecd_detector usage:

```
from gerikl.ecg_detector import load_model, predict_edf

model = load_model()
prediction = predict_edf("your_path", model)
prediction
```
