# Data Note: HeAR Cough Classifier Training Data

## Ideal Dataset: CODA TB

The ideal dataset for training a TB cough classifier would be the **CODA TB dataset** 
(Sharma et al., *Science Advances*, 2024), which contains over 700,000 cough sounds 
from 2,143 individuals across seven countries, with detailed TB status annotations.

**Citation:**  
Sharma, M., et al. (2024). "CODA TB: A large-scale open-access cough dataset for 
the diagnosis of tuberculosis." *Science Advances*.

Due to resource and time constraints for this challenge, the CODA TB dataset was 
not accessible. We instead used the Coswara dataset as a proxy.

## Dataset Used: Coswara Heavy Cough (Kaggle)

The classifier was trained on the **Coswara dataset** (Indian Institute of Science), 
which collects respiratory audio including forced cough recordings, with metadata 
on COVID-19 and respiratory illness status.

- **Source**: https://www.kaggle.com/datasets/sarabhian/coswara-dataset-heavy-cough
- **Samples**: ~2,189 cough recordings (605 positive, 1584 normal)
- **Labels**: Mapped from `covid_status` field:
  - Positive: `positive_mild`, `positive_moderate`, `resp_illness_not_identified`
  - Normal: `healthy`, `no_resp_illness_exposed`

### Limitations

- The Coswara dataset targets COVID-19 / general respiratory illness, **not TB specifically**.
- The classifier should be considered a **respiratory illness cough detector** rather than 
  a true TB classifier.
- With TB-specific training data (e.g., CODA TB), the HeAR model's accuracy for TB 
  screening would significantly improve.
