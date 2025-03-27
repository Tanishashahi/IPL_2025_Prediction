# IPL 2025 Toss & Match Winner Predictions (Enhanced)

*Report Generated on: 2025-03-27 19:59:51 UTC*

## Introduction
This report presents match-by-match predictions for the initial league stage (up to Match 70) of the IPL 2025 season, based on the provided schedule. The predictions leverage machine learning models trained on historical IPL data (`matches.csv` and `deliveries.csv`).

Key steps in the process included:
- Loading and cleaning historical match and delivery data.
- Standardizing team names across datasets.
- Engineering features, including overall team batting strike rates and bowling economy/wicket rates derived from delivery data.
- Encoding categorical features (like venue, teams) consistently across historical and future schedule data.
- Evaluating several classification models (Logistic Regression, Random Forest, Gradient Boosting) for both toss and match winner prediction using a train-validation split.
- Selecting the best performing models based on the weighted F1-score on the validation set.
- The best model for **Toss Prediction** was: **Random Forest** (Validation F1: 0.5572)
- The best model for **Match Winner Prediction** was: **Random Forest** (Validation F1: 0.5231)
- Retraining these selected models on the entire historical dataset.
- Preparing the IPL 2025 schedule data (encoding, merging stats, scaling numerical features using the scaler fitted on historical data).
- Generating predictions for each match, including toss winner, match winner, and the winner's probability.

**Note:** Predictions rely on historical averages and patterns. Actual outcomes can be influenced by numerous real-time factors like player form, injuries, specific pitch conditions on the day, and strategic decisions not captured by this model. Unseen values (e.g., new venues or teams not in historical data) are handled, but predictions involving them may be less reliable.

## Model Evaluation Summary (Validation Set)
| Model Type              | Model Name          |   Accuracy |   Precision |   Recall |   F1 Score |   ROC AUC |
|:------------------------|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Toss Prediction         | Logistic Regression |     0.5275 |      0.5428 |   0.5275 |     0.5327 |    0.4971 |
| Toss Prediction         | Random Forest       |     0.5531 |      0.5641 |   0.5531 |     0.5572 |    0.5499 |
| Toss Prediction         | Gradient Boosting   |     0.5971 |      0.557  |   0.5971 |     0.5423 |    0.5976 |
| Match Winner Prediction | Logistic Regression |     0.5128 |      0.5126 |   0.5128 |     0.5127 |    0.5519 |
| Match Winner Prediction | Random Forest       |     0.5238 |      0.5234 |   0.5238 |     0.5231 |    0.5343 |
| Match Winner Prediction | Gradient Boosting   |     0.5055 |      0.504  |   0.5055 |     0.5005 |    0.5213 |

---

## IPL 2025 Match Predictions (League Stage)

### Match 1: Kolkata Knight Riders vs Royal Challengers Bengaluru
**Date:** 2025-03-22 14:00 UTC
**Venue:** Eden Gardens

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **61.0%** of the 77 matches played here. 
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **45.5%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 60.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Head-to-Head:** Overall H2H: Kolkata Knight Riders **2** - **0** Royal Challengers Bengaluru (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Kolkata Knight Riders **2** - **0** Royal Challengers Bengaluru.
- **Historical Performance Metrics:**
  - Avg Batting SR: Kolkata Knight Riders (133.3) vs Royal Challengers Bengaluru (161.2)
  - Avg Bowling Econ: Kolkata Knight Riders (7.91) vs Royal Challengers Bengaluru (9.39)

**Model Predictions:**
- **Predicted Toss Winner:** Royal Challengers Bengaluru
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Kolkata Knight Riders**
- **Winner Probability:** 0.599

**Code Snippet Insight (Example - Add Manually for a Representative Match):**
*(This section is intended to show a small piece of the prediction logic for transparency. Below is a conceptual example of how Match 1's winner might be derived in the code. Add your actual relevant snippet here if desired.)*
```python
# Conceptual Snippet for Match 1 Winner Prediction
# match_data_live_scaled = schedule_pred_df_scaled.iloc[0]
# # ... (add predicted toss info to match_data_live_scaled) ...
# X_live = pd.DataFrame([match_data_live_scaled])[final_match_pred_features]
# match_win_pred_encoded = final_match_model.predict(X_live)[0]
# match_win_prob = final_match_model.predict_proba(X_live)[0]
# predicted_winner = team1 if match_win_pred_encoded == 1 else team2
# probability = match_win_prob[1] if match_win_pred_encoded == 1 else match_win_prob[0]
```

---

### Match 2: Sunrisers Hyderabad vs Rajasthan Royals
**Date:** 2025-03-23 10:00 UTC
**Venue:** Rajiv Gandhi International Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **46.7%** of the 15 matches played here. 
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **46.7%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **6.7%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 0.0%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Head-to-Head:** Overall H2H: Sunrisers Hyderabad **11** - **9** Rajasthan Royals (Total: 20). At this venue: Sunrisers Hyderabad **2** - **0** Rajasthan Royals. In their last 5 encounters: Sunrisers Hyderabad **3** - **2** Rajasthan Royals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Sunrisers Hyderabad (133.1) vs Rajasthan Royals (132.4)
  - Avg Bowling Econ: Sunrisers Hyderabad (8.04) vs Rajasthan Royals (7.98)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Sunrisers Hyderabad**
- **Winner Probability:** 0.537

---

### Match 3: Chennai Super Kings vs Mumbai Indians
**Date:** 2025-03-23 14:00 UTC
**Venue:** MA Chidambaram Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **55.6%** of the 9 matches played here. 
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **77.8%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 71.4%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **0.0%** of matches at this venue (Overall toss win: 54.8%). Mumbai Indians hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Chennai Super Kings **17** - **20** Mumbai Indians (Total: 37). At this venue: Chennai Super Kings **0** - **2** Mumbai Indians. In their last 5 encounters: Chennai Super Kings **4** - **1** Mumbai Indians.
- **Historical Performance Metrics:**
  - Avg Batting SR: Chennai Super Kings (134.8) vs Mumbai Indians (134.2)
  - Avg Bowling Econ: Chennai Super Kings (7.81) vs Mumbai Indians (7.86)

**Model Predictions:**
- **Predicted Toss Winner:** Chennai Super Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Mumbai Indians**
- **Winner Probability:** 0.510

---

### Match 4: Delhi Capitals vs Lucknow Super Giants
**Date:** 2025-03-24 14:00 UTC
**Venue:** Dr YS Rajasekhara Reddy ACA-VDCA Cricket Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals has an overall toss win rate of **51.6%**. No venue-specific toss data.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Delhi Capitals **2** - **3** Lucknow Super Giants (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Delhi Capitals **2** - **3** Lucknow Super Giants.
- **Historical Performance Metrics:**
  - Avg Batting SR: Delhi Capitals (131.8) vs Lucknow Super Giants (139.1)
  - Avg Bowling Econ: Delhi Capitals (8.04) vs Lucknow Super Giants (8.51)

**Model Predictions:**
- **Predicted Toss Winner:** Lucknow Super Giants
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Delhi Capitals**
- **Winner Probability:** 0.550

---

### Match 5: Gujarat Titans vs Punjab Kings
**Date:** 2025-03-25 14:00 UTC
**Venue:** Narendra Modi Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans has an overall toss win rate of **48.9%**. No venue-specific toss data.
- **Toss Analysis (Punjab Kings):** Punjab Kings has an overall toss win rate of **44.3%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Gujarat Titans **3** - **2** Punjab Kings (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Gujarat Titans **3** - **2** Punjab Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Gujarat Titans (141.2) vs Punjab Kings (134.3)
  - Avg Bowling Econ: Gujarat Titans (8.46) vs Punjab Kings (8.22)

**Model Predictions:**
- **Predicted Toss Winner:** Punjab Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.501

---

### Match 6: Rajasthan Royals vs Kolkata Knight Riders
**Date:** 2025-03-26 14:00 UTC
**Venue:** Barsapara Cricket Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals has an overall toss win rate of **53.9%**. No venue-specific toss data.
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders has an overall toss win rate of **48.6%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Rajasthan Royals **14** - **14** Kolkata Knight Riders (Total: 28). They haven't played each other at this venue historically. In their last 5 encounters: Rajasthan Royals **3** - **2** Kolkata Knight Riders.
- **Historical Performance Metrics:**
  - Avg Batting SR: Rajasthan Royals (132.4) vs Kolkata Knight Riders (133.3)
  - Avg Bowling Econ: Rajasthan Royals (7.98) vs Kolkata Knight Riders (7.91)

**Model Predictions:**
- **Predicted Toss Winner:** Kolkata Knight Riders
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Rajasthan Royals**
- **Winner Probability:** 0.545

---

### Match 7: Sunrisers Hyderabad vs Lucknow Super Giants
**Date:** 2025-03-27 14:00 UTC
**Venue:** Rajiv Gandhi International Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **46.7%** of the 15 matches played here. 
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **46.7%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants wins the toss in **0.0%** of matches at this venue (Overall toss win: 44.2%). Lucknow Super Giants hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Head-to-Head:** Overall H2H: Sunrisers Hyderabad **1** - **3** Lucknow Super Giants (Total: 4). They haven't played each other at this venue historically. In their last 4 encounters: Sunrisers Hyderabad **1** - **3** Lucknow Super Giants.
- **Historical Performance Metrics:**
  - Avg Batting SR: Sunrisers Hyderabad (133.1) vs Lucknow Super Giants (139.1)
  - Avg Bowling Econ: Sunrisers Hyderabad (8.04) vs Lucknow Super Giants (8.51)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Sunrisers Hyderabad**
- **Winner Probability:** 0.517

---

### Match 8: Chennai Super Kings vs Royal Challengers Bengaluru
**Date:** 2025-03-28 14:00 UTC
**Venue:** MA Chidambaram Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **55.6%** of the 9 matches played here. 
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **77.8%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 71.4%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Chennai Super Kings **1** - **1** Royal Challengers Bengaluru (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Chennai Super Kings **1** - **1** Royal Challengers Bengaluru.
- **Historical Performance Metrics:**
  - Avg Batting SR: Chennai Super Kings (134.8) vs Royal Challengers Bengaluru (161.2)
  - Avg Bowling Econ: Chennai Super Kings (7.81) vs Royal Challengers Bengaluru (9.39)

**Model Predictions:**
- **Predicted Toss Winner:** Royal Challengers Bengaluru
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.545

---

### Match 9: Gujarat Titans vs Mumbai Indians
**Date:** 2025-03-29 14:00 UTC
**Venue:** Narendra Modi Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans has an overall toss win rate of **48.9%**. No venue-specific toss data.
- **Toss Analysis (Mumbai Indians):** Mumbai Indians has an overall toss win rate of **54.8%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Gujarat Titans **3** - **2** Mumbai Indians (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Gujarat Titans **3** - **2** Mumbai Indians.
- **Historical Performance Metrics:**
  - Avg Batting SR: Gujarat Titans (141.2) vs Mumbai Indians (134.2)
  - Avg Bowling Econ: Gujarat Titans (8.46) vs Mumbai Indians (7.86)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.759

---

### Match 10: Delhi Capitals vs Sunrisers Hyderabad
**Date:** 2025-03-30 10:00 UTC
**Venue:** Dr YS Rajasekhara Reddy ACA-VDCA Cricket Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals has an overall toss win rate of **51.6%**. No venue-specific toss data.
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad has an overall toss win rate of **48.4%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Delhi Capitals **11** - **13** Sunrisers Hyderabad (Total: 24). They haven't played each other at this venue historically. In their last 5 encounters: Delhi Capitals **3** - **2** Sunrisers Hyderabad.
- **Historical Performance Metrics:**
  - Avg Batting SR: Delhi Capitals (131.8) vs Sunrisers Hyderabad (133.1)
  - Avg Bowling Econ: Delhi Capitals (8.04) vs Sunrisers Hyderabad (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Sunrisers Hyderabad**
- **Winner Probability:** 0.565

---

### Match 11: Rajasthan Royals vs Chennai Super Kings
**Date:** 2025-03-30 14:00 UTC
**Venue:** Barsapara Cricket Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals has an overall toss win rate of **53.9%**. No venue-specific toss data.
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings has an overall toss win rate of **51.1%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Rajasthan Royals **13** - **16** Chennai Super Kings (Total: 29). They haven't played each other at this venue historically. In their last 5 encounters: Rajasthan Royals **4** - **1** Chennai Super Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Rajasthan Royals (132.4) vs Chennai Super Kings (134.8)
  - Avg Bowling Econ: Rajasthan Royals (7.98) vs Chennai Super Kings (7.81)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.516

---

### Match 12: Mumbai Indians vs Kolkata Knight Riders
**Date:** 2025-03-31 14:00 UTC
**Venue:** Wankhede Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.7%** of the 73 matches played here. 
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **50.7%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 62.2%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **2.7%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Head-to-Head:** Overall H2H: Mumbai Indians **23** - **11** Kolkata Knight Riders (Total: 34). At this venue: Mumbai Indians **8** - **1** Kolkata Knight Riders. In their last 5 encounters: Mumbai Indians **1** - **4** Kolkata Knight Riders.
- **Historical Performance Metrics:**
  - Avg Batting SR: Mumbai Indians (134.2) vs Kolkata Knight Riders (133.3)
  - Avg Bowling Econ: Mumbai Indians (7.86) vs Kolkata Knight Riders (7.91)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Mumbai Indians**
- **Winner Probability:** 0.703

---

### Match 13: Lucknow Super Giants vs Punjab Kings
**Date:** 2025-04-01 14:00 UTC
**Venue:** Bharat Ratna Shri Atal Bihari Vajpayee Ekana Crick

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Toss Analysis (Punjab Kings):** Punjab Kings has an overall toss win rate of **44.3%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Lucknow Super Giants **3** - **1** Punjab Kings (Total: 4). They haven't played each other at this venue historically. In their last 4 encounters: Lucknow Super Giants **3** - **1** Punjab Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Lucknow Super Giants (139.1) vs Punjab Kings (134.3)
  - Avg Bowling Econ: Lucknow Super Giants (8.51) vs Punjab Kings (8.22)

**Model Predictions:**
- **Predicted Toss Winner:** Punjab Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Lucknow Super Giants**
- **Winner Probability:** 0.603

---

### Match 14: Royal Challengers Bengaluru vs Gujarat Titans
**Date:** 2025-04-02 14:00 UTC
**Venue:** M Chinnaswamy Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **58.7%** of the 63 matches played here. 
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.9%). Gujarat Titans hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Royal Challengers Bengaluru **2** - **0** Gujarat Titans (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Royal Challengers Bengaluru **2** - **0** Gujarat Titans.
- **Historical Performance Metrics:**
  - Avg Batting SR: Royal Challengers Bengaluru (161.2) vs Gujarat Titans (141.2)
  - Avg Bowling Econ: Royal Challengers Bengaluru (9.39) vs Gujarat Titans (8.46)

**Model Predictions:**
- **Predicted Toss Winner:** Gujarat Titans
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.671

---

### Match 15: Kolkata Knight Riders vs Sunrisers Hyderabad
**Date:** 2025-04-03 14:00 UTC
**Venue:** Eden Gardens

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **61.0%** of the 77 matches played here. 
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **45.5%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 60.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **5.2%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Head-to-Head:** Overall H2H: Kolkata Knight Riders **19** - **9** Sunrisers Hyderabad (Total: 28). At this venue: Kolkata Knight Riders **6** - **2** Sunrisers Hyderabad. In their last 5 encounters: Kolkata Knight Riders **4** - **1** Sunrisers Hyderabad.
- **Historical Performance Metrics:**
  - Avg Batting SR: Kolkata Knight Riders (133.3) vs Sunrisers Hyderabad (133.1)
  - Avg Bowling Econ: Kolkata Knight Riders (7.91) vs Sunrisers Hyderabad (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Kolkata Knight Riders**
- **Winner Probability:** 0.649

---

### Match 16: Lucknow Super Giants vs Mumbai Indians
**Date:** 2025-04-04 14:00 UTC
**Venue:** Bharat Ratna Shri Atal Bihari Vajpayee Ekana Crick

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Toss Analysis (Mumbai Indians):** Mumbai Indians has an overall toss win rate of **54.8%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Lucknow Super Giants **5** - **1** Mumbai Indians (Total: 6). They haven't played each other at this venue historically. In their last 5 encounters: Lucknow Super Giants **4** - **1** Mumbai Indians.
- **Historical Performance Metrics:**
  - Avg Batting SR: Lucknow Super Giants (139.1) vs Mumbai Indians (134.2)
  - Avg Bowling Econ: Lucknow Super Giants (8.51) vs Mumbai Indians (7.86)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Lucknow Super Giants**
- **Winner Probability:** 0.749

---

### Match 17: Chennai Super Kings vs Delhi Capitals
**Date:** 2025-04-05 10:00 UTC
**Venue:** MA Chidambaram Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **55.6%** of the 9 matches played here. 
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **77.8%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 71.4%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **11.1%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Chennai Super Kings **19** - **11** Delhi Capitals (Total: 30). At this venue: Chennai Super Kings **1** - **0** Delhi Capitals. In their last 5 encounters: Chennai Super Kings **4** - **1** Delhi Capitals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Chennai Super Kings (134.8) vs Delhi Capitals (131.8)
  - Avg Bowling Econ: Chennai Super Kings (7.81) vs Delhi Capitals (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Chennai Super Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.670

---

### Match 18: Punjab Kings vs Rajasthan Royals
**Date:** 2025-04-05 14:00 UTC
**Venue:** New PCA Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Punjab Kings):** Punjab Kings has an overall toss win rate of **44.3%**. No venue-specific toss data.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals has an overall toss win rate of **53.9%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Punjab Kings **12** - **16** Rajasthan Royals (Total: 28). They haven't played each other at this venue historically. In their last 5 encounters: Punjab Kings **2** - **3** Rajasthan Royals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Punjab Kings (134.3) vs Rajasthan Royals (132.4)
  - Avg Bowling Econ: Punjab Kings (8.22) vs Rajasthan Royals (7.98)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Rajasthan Royals**
- **Winner Probability:** 0.570

---

### Match 19: Kolkata Knight Riders vs Lucknow Super Giants
**Date:** 2025-04-06 10:00 UTC
**Venue:** Eden Gardens

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **61.0%** of the 77 matches played here. 
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **45.5%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 60.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants wins the toss in **0.0%** of matches at this venue (Overall toss win: 44.2%). Lucknow Super Giants hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Head-to-Head:** Overall H2H: Kolkata Knight Riders **2** - **3** Lucknow Super Giants (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Kolkata Knight Riders **2** - **3** Lucknow Super Giants.
- **Historical Performance Metrics:**
  - Avg Batting SR: Kolkata Knight Riders (133.3) vs Lucknow Super Giants (139.1)
  - Avg Bowling Econ: Kolkata Knight Riders (7.91) vs Lucknow Super Giants (8.51)

**Model Predictions:**
- **Predicted Toss Winner:** Kolkata Knight Riders
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Kolkata Knight Riders**
- **Winner Probability:** 0.570

---

### Match 20: Sunrisers Hyderabad vs Gujarat Titans
**Date:** 2025-04-06 14:00 UTC
**Venue:** Rajiv Gandhi International Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **46.7%** of the 15 matches played here. 
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **46.7%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.9%). Gujarat Titans hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Head-to-Head:** Overall H2H: Sunrisers Hyderabad **1** - **3** Gujarat Titans (Total: 4). They haven't played each other at this venue historically. In their last 4 encounters: Sunrisers Hyderabad **1** - **3** Gujarat Titans.
- **Historical Performance Metrics:**
  - Avg Batting SR: Sunrisers Hyderabad (133.1) vs Gujarat Titans (141.2)
  - Avg Bowling Econ: Sunrisers Hyderabad (8.04) vs Gujarat Titans (8.46)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.763

---

### Match 21: Mumbai Indians vs Royal Challengers Bengaluru
**Date:** 2025-04-07 14:00 UTC
**Venue:** Wankhede Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.7%** of the 73 matches played here. 
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **50.7%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 62.2%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Head-to-Head:** Overall H2H: Mumbai Indians **1** - **0** Royal Challengers Bengaluru (Total: 1). They haven't played each other at this venue historically. In their last 1 encounters: Mumbai Indians **1** - **0** Royal Challengers Bengaluru.
- **Historical Performance Metrics:**
  - Avg Batting SR: Mumbai Indians (134.2) vs Royal Challengers Bengaluru (161.2)
  - Avg Bowling Econ: Mumbai Indians (7.86) vs Royal Challengers Bengaluru (9.39)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Mumbai Indians**
- **Winner Probability:** 0.550

---

### Match 22: Punjab Kings vs Chennai Super Kings
**Date:** 2025-04-08 14:00 UTC
**Venue:** New PCA Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Punjab Kings):** Punjab Kings has an overall toss win rate of **44.3%**. No venue-specific toss data.
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings has an overall toss win rate of **51.1%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Punjab Kings **14** - **16** Chennai Super Kings (Total: 30). They haven't played each other at this venue historically. In their last 5 encounters: Punjab Kings **4** - **1** Chennai Super Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Punjab Kings (134.3) vs Chennai Super Kings (134.8)
  - Avg Bowling Econ: Punjab Kings (8.22) vs Chennai Super Kings (7.81)

**Model Predictions:**
- **Predicted Toss Winner:** Chennai Super Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.619

---

### Match 23: Gujarat Titans vs Rajasthan Royals
**Date:** 2025-04-09 14:00 UTC
**Venue:** Narendra Modi Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans has an overall toss win rate of **48.9%**. No venue-specific toss data.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals has an overall toss win rate of **53.9%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Gujarat Titans **5** - **1** Rajasthan Royals (Total: 6). They haven't played each other at this venue historically. In their last 5 encounters: Gujarat Titans **4** - **1** Rajasthan Royals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Gujarat Titans (141.2) vs Rajasthan Royals (132.4)
  - Avg Bowling Econ: Gujarat Titans (8.46) vs Rajasthan Royals (7.98)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.512

---

### Match 24: Royal Challengers Bengaluru vs Delhi Capitals
**Date:** 2025-04-10 14:00 UTC
**Venue:** M Chinnaswamy Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **58.7%** of the 63 matches played here. 
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **6.3%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Royal Challengers Bengaluru **1** - **0** Delhi Capitals (Total: 1). They haven't played each other at this venue historically. In their last 1 encounters: Royal Challengers Bengaluru **1** - **0** Delhi Capitals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Royal Challengers Bengaluru (161.2) vs Delhi Capitals (131.8)
  - Avg Bowling Econ: Royal Challengers Bengaluru (9.39) vs Delhi Capitals (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Delhi Capitals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Royal Challengers Bengaluru**
- **Winner Probability:** 0.530

---

### Match 25: Chennai Super Kings vs Kolkata Knight Riders
**Date:** 2025-04-11 14:00 UTC
**Venue:** MA Chidambaram Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **55.6%** of the 9 matches played here. 
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **77.8%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 71.4%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.6%). Kolkata Knight Riders hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Chennai Super Kings **19** - **10** Kolkata Knight Riders (Total: 29). At this venue: Chennai Super Kings **2** - **0** Kolkata Knight Riders. In their last 5 encounters: Chennai Super Kings **3** - **2** Kolkata Knight Riders.
- **Historical Performance Metrics:**
  - Avg Batting SR: Chennai Super Kings (134.8) vs Kolkata Knight Riders (133.3)
  - Avg Bowling Econ: Chennai Super Kings (7.81) vs Kolkata Knight Riders (7.91)

**Model Predictions:**
- **Predicted Toss Winner:** Kolkata Knight Riders
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.705

---

### Match 26: Lucknow Super Giants vs Gujarat Titans
**Date:** 2025-04-12 10:00 UTC
**Venue:** Bharat Ratna Shri Atal Bihari Vajpayee Ekana Crick

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans has an overall toss win rate of **48.9%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Lucknow Super Giants **1** - **4** Gujarat Titans (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Lucknow Super Giants **1** - **4** Gujarat Titans.
- **Historical Performance Metrics:**
  - Avg Batting SR: Lucknow Super Giants (139.1) vs Gujarat Titans (141.2)
  - Avg Bowling Econ: Lucknow Super Giants (8.51) vs Gujarat Titans (8.46)

**Model Predictions:**
- **Predicted Toss Winner:** Gujarat Titans
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Lucknow Super Giants**
- **Winner Probability:** 0.587

---

### Match 27: Sunrisers Hyderabad vs Punjab Kings
**Date:** 2025-04-12 14:00 UTC
**Venue:** Rajiv Gandhi International Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **46.7%** of the 15 matches played here. 
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **46.7%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Toss Analysis (Punjab Kings):** Punjab Kings wins the toss in **13.3%** of matches at this venue (Overall toss win: 44.3%). When Punjab Kings wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Head-to-Head:** Overall H2H: Sunrisers Hyderabad **16** - **7** Punjab Kings (Total: 23). At this venue: Sunrisers Hyderabad **2** - **0** Punjab Kings. In their last 5 encounters: Sunrisers Hyderabad **4** - **1** Punjab Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Sunrisers Hyderabad (133.1) vs Punjab Kings (134.3)
  - Avg Bowling Econ: Sunrisers Hyderabad (8.04) vs Punjab Kings (8.22)

**Model Predictions:**
- **Predicted Toss Winner:** Punjab Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Sunrisers Hyderabad**
- **Winner Probability:** 0.657

---

### Match 28: Rajasthan Royals vs Royal Challengers Bengaluru
**Date:** 2025-04-13 10:00 UTC
**Venue:** Sawai Mansingh Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **68.1%** of the 47 matches played here. 
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **51.1%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 66.7%** of the time. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Head-to-Head:** Overall H2H: Rajasthan Royals **2** - **0** Royal Challengers Bengaluru (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Rajasthan Royals **2** - **0** Royal Challengers Bengaluru.
- **Historical Performance Metrics:**
  - Avg Batting SR: Rajasthan Royals (132.4) vs Royal Challengers Bengaluru (161.2)
  - Avg Bowling Econ: Rajasthan Royals (7.98) vs Royal Challengers Bengaluru (9.39)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Rajasthan Royals**
- **Winner Probability:** 0.525

---

### Match 29: Delhi Capitals vs Mumbai Indians
**Date:** 2025-04-13 14:00 UTC
**Venue:** Arun Jaitley Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.0%** of the 14 matches played here. 
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **50.0%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 42.9%** of the time. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **7.1%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 0.0%** of the time. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Head-to-Head:** Overall H2H: Delhi Capitals **16** - **19** Mumbai Indians (Total: 35). At this venue: Delhi Capitals **1** - **1** Mumbai Indians. In their last 5 encounters: Delhi Capitals **2** - **3** Mumbai Indians.
- **Historical Performance Metrics:**
  - Avg Batting SR: Delhi Capitals (131.8) vs Mumbai Indians (134.2)
  - Avg Bowling Econ: Delhi Capitals (8.04) vs Mumbai Indians (7.86)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Delhi Capitals**
- **Winner Probability:** 0.581

---

### Match 30: Lucknow Super Giants vs Chennai Super Kings
**Date:** 2025-04-14 14:00 UTC
**Venue:** Bharat Ratna Shri Atal Bihari Vajpayee Ekana Crick

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings has an overall toss win rate of **51.1%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Lucknow Super Giants **3** - **1** Chennai Super Kings (Total: 4). They haven't played each other at this venue historically. In their last 4 encounters: Lucknow Super Giants **3** - **1** Chennai Super Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Lucknow Super Giants (139.1) vs Chennai Super Kings (134.8)
  - Avg Bowling Econ: Lucknow Super Giants (8.51) vs Chennai Super Kings (7.81)

**Model Predictions:**
- **Predicted Toss Winner:** Lucknow Super Giants
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Lucknow Super Giants**
- **Winner Probability:** 0.632

---

### Match 31: Punjab Kings vs Kolkata Knight Riders
**Date:** 2025-04-15 14:00 UTC
**Venue:** New PCA Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Punjab Kings):** Punjab Kings has an overall toss win rate of **44.3%**. No venue-specific toss data.
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders has an overall toss win rate of **48.6%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Punjab Kings **12** - **21** Kolkata Knight Riders (Total: 33). They haven't played each other at this venue historically. In their last 5 encounters: Punjab Kings **3** - **2** Kolkata Knight Riders.
- **Historical Performance Metrics:**
  - Avg Batting SR: Punjab Kings (134.3) vs Kolkata Knight Riders (133.3)
  - Avg Bowling Econ: Punjab Kings (8.22) vs Kolkata Knight Riders (7.91)

**Model Predictions:**
- **Predicted Toss Winner:** Kolkata Knight Riders
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Kolkata Knight Riders**
- **Winner Probability:** 0.638

---

### Match 32: Delhi Capitals vs Rajasthan Royals
**Date:** 2025-04-16 14:00 UTC
**Venue:** Arun Jaitley Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.0%** of the 14 matches played here. 
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **50.0%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 42.9%** of the time. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **14.3%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 50.0%** of the time. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Head-to-Head:** Overall H2H: Delhi Capitals **14** - **15** Rajasthan Royals (Total: 29). At this venue: Delhi Capitals **2** - **0** Rajasthan Royals. In their last 5 encounters: Delhi Capitals **2** - **3** Rajasthan Royals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Delhi Capitals (131.8) vs Rajasthan Royals (132.4)
  - Avg Bowling Econ: Delhi Capitals (8.04) vs Rajasthan Royals (7.98)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Delhi Capitals**
- **Winner Probability:** 0.564

---

### Match 33: Mumbai Indians vs Sunrisers Hyderabad
**Date:** 2025-04-17 14:00 UTC
**Venue:** Wankhede Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.7%** of the 73 matches played here. 
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **50.7%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 62.2%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **1.4%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 0.0%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Head-to-Head:** Overall H2H: Mumbai Indians **13** - **10** Sunrisers Hyderabad (Total: 23). At this venue: Mumbai Indians **4** - **1** Sunrisers Hyderabad. In their last 5 encounters: Mumbai Indians **3** - **2** Sunrisers Hyderabad.
- **Historical Performance Metrics:**
  - Avg Batting SR: Mumbai Indians (134.2) vs Sunrisers Hyderabad (133.1)
  - Avg Bowling Econ: Mumbai Indians (7.86) vs Sunrisers Hyderabad (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Mumbai Indians**
- **Winner Probability:** 0.682

---

### Match 34: Royal Challengers Bengaluru vs Punjab Kings
**Date:** 2025-04-18 14:00 UTC
**Venue:** M Chinnaswamy Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **58.7%** of the 63 matches played here. 
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Punjab Kings):** Punjab Kings wins the toss in **11.1%** of matches at this venue (Overall toss win: 44.3%). When Punjab Kings wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Royal Challengers Bengaluru **2** - **0** Punjab Kings (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Royal Challengers Bengaluru **2** - **0** Punjab Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Royal Challengers Bengaluru (161.2) vs Punjab Kings (134.3)
  - Avg Bowling Econ: Royal Challengers Bengaluru (9.39) vs Punjab Kings (8.22)

**Model Predictions:**
- **Predicted Toss Winner:** Punjab Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Punjab Kings**
- **Winner Probability:** 0.528

---

### Match 35: Gujarat Titans vs Delhi Capitals
**Date:** 2025-04-19 10:00 UTC
**Venue:** Narendra Modi Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans has an overall toss win rate of **48.9%**. No venue-specific toss data.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals has an overall toss win rate of **51.6%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Gujarat Titans **2** - **3** Delhi Capitals (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Gujarat Titans **2** - **3** Delhi Capitals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Gujarat Titans (141.2) vs Delhi Capitals (131.8)
  - Avg Bowling Econ: Gujarat Titans (8.46) vs Delhi Capitals (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Delhi Capitals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.585

---

### Match 36: Rajasthan Royals vs Lucknow Super Giants
**Date:** 2025-04-19 14:00 UTC
**Venue:** Sawai Mansingh Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **68.1%** of the 47 matches played here. 
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **51.1%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 66.7%** of the time. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants wins the toss in **0.0%** of matches at this venue (Overall toss win: 44.2%). Lucknow Super Giants hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Head-to-Head:** Overall H2H: Rajasthan Royals **4** - **1** Lucknow Super Giants (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Rajasthan Royals **4** - **1** Lucknow Super Giants.
- **Historical Performance Metrics:**
  - Avg Batting SR: Rajasthan Royals (132.4) vs Lucknow Super Giants (139.1)
  - Avg Bowling Econ: Rajasthan Royals (7.98) vs Lucknow Super Giants (8.51)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Rajasthan Royals**
- **Winner Probability:** 0.616

---

### Match 37: Punjab Kings vs Royal Challengers Bengaluru
**Date:** 2025-04-20 10:00 UTC
**Venue:** New PCA Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Punjab Kings):** Punjab Kings has an overall toss win rate of **44.3%**. No venue-specific toss data.
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru has an overall toss win rate of **53.3%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Punjab Kings **0** - **2** Royal Challengers Bengaluru (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Punjab Kings **0** - **2** Royal Challengers Bengaluru.
- **Historical Performance Metrics:**
  - Avg Batting SR: Punjab Kings (134.3) vs Royal Challengers Bengaluru (161.2)
  - Avg Bowling Econ: Punjab Kings (8.22) vs Royal Challengers Bengaluru (9.39)

**Model Predictions:**
- **Predicted Toss Winner:** Royal Challengers Bengaluru
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Royal Challengers Bengaluru**
- **Winner Probability:** 0.609

---

### Match 38: Mumbai Indians vs Chennai Super Kings
**Date:** 2025-04-20 14:00 UTC
**Venue:** Wankhede Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.7%** of the 73 matches played here. 
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **50.7%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 62.2%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **11.0%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Head-to-Head:** Overall H2H: Mumbai Indians **20** - **17** Chennai Super Kings (Total: 37). At this venue: Mumbai Indians **6** - **3** Chennai Super Kings. In their last 5 encounters: Mumbai Indians **1** - **4** Chennai Super Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Mumbai Indians (134.2) vs Chennai Super Kings (134.8)
  - Avg Bowling Econ: Mumbai Indians (7.86) vs Chennai Super Kings (7.81)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Mumbai Indians**
- **Winner Probability:** 0.590

---

### Match 39: Kolkata Knight Riders vs Gujarat Titans
**Date:** 2025-04-21 14:00 UTC
**Venue:** Eden Gardens

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **61.0%** of the 77 matches played here. 
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **45.5%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 60.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.9%). Gujarat Titans hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Head-to-Head:** Overall H2H: Kolkata Knight Riders **1** - **2** Gujarat Titans (Total: 3). They haven't played each other at this venue historically. In their last 3 encounters: Kolkata Knight Riders **1** - **2** Gujarat Titans.
- **Historical Performance Metrics:**
  - Avg Batting SR: Kolkata Knight Riders (133.3) vs Gujarat Titans (141.2)
  - Avg Bowling Econ: Kolkata Knight Riders (7.91) vs Gujarat Titans (8.46)

**Model Predictions:**
- **Predicted Toss Winner:** Kolkata Knight Riders
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.648

---

### Match 40: Lucknow Super Giants vs Delhi Capitals
**Date:** 2025-04-22 14:00 UTC
**Venue:** Bharat Ratna Shri Atal Bihari Vajpayee Ekana Crick

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals has an overall toss win rate of **51.6%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Lucknow Super Giants **3** - **2** Delhi Capitals (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Lucknow Super Giants **3** - **2** Delhi Capitals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Lucknow Super Giants (139.1) vs Delhi Capitals (131.8)
  - Avg Bowling Econ: Lucknow Super Giants (8.51) vs Delhi Capitals (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Lucknow Super Giants
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Lucknow Super Giants**
- **Winner Probability:** 0.606

---

### Match 41: Sunrisers Hyderabad vs Mumbai Indians
**Date:** 2025-04-23 14:00 UTC
**Venue:** Rajiv Gandhi International Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **46.7%** of the 15 matches played here. 
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **46.7%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **6.7%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 0.0%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Head-to-Head:** Overall H2H: Sunrisers Hyderabad **10** - **13** Mumbai Indians (Total: 23). At this venue: Sunrisers Hyderabad **1** - **1** Mumbai Indians. In their last 5 encounters: Sunrisers Hyderabad **2** - **3** Mumbai Indians.
- **Historical Performance Metrics:**
  - Avg Batting SR: Sunrisers Hyderabad (133.1) vs Mumbai Indians (134.2)
  - Avg Bowling Econ: Sunrisers Hyderabad (8.04) vs Mumbai Indians (7.86)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Sunrisers Hyderabad**
- **Winner Probability:** 0.603

---

### Match 42: Royal Challengers Bengaluru vs Rajasthan Royals
**Date:** 2025-04-24 14:00 UTC
**Venue:** M Chinnaswamy Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **58.7%** of the 63 matches played here. 
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **3.2%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 50.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Royal Challengers Bengaluru **0** - **2** Rajasthan Royals (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Royal Challengers Bengaluru **0** - **2** Rajasthan Royals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Royal Challengers Bengaluru (161.2) vs Rajasthan Royals (132.4)
  - Avg Bowling Econ: Royal Challengers Bengaluru (9.39) vs Rajasthan Royals (7.98)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Rajasthan Royals**
- **Winner Probability:** 0.644

---

### Match 43: Chennai Super Kings vs Sunrisers Hyderabad
**Date:** 2025-04-25 14:00 UTC
**Venue:** MA Chidambaram Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **55.6%** of the 9 matches played here. 
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **77.8%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 71.4%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.4%). Sunrisers Hyderabad hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Chennai Super Kings **15** - **6** Sunrisers Hyderabad (Total: 21). At this venue: Chennai Super Kings **1** - **0** Sunrisers Hyderabad. In their last 5 encounters: Chennai Super Kings **3** - **2** Sunrisers Hyderabad.
- **Historical Performance Metrics:**
  - Avg Batting SR: Chennai Super Kings (134.8) vs Sunrisers Hyderabad (133.1)
  - Avg Bowling Econ: Chennai Super Kings (7.81) vs Sunrisers Hyderabad (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.654

---

### Match 44: Kolkata Knight Riders vs Punjab Kings
**Date:** 2025-04-26 14:00 UTC
**Venue:** Eden Gardens

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **61.0%** of the 77 matches played here. 
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **45.5%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 60.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Toss Analysis (Punjab Kings):** Punjab Kings wins the toss in **9.1%** of matches at this venue (Overall toss win: 44.3%). When Punjab Kings wins the toss here, they choose to **field 57.1%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Head-to-Head:** Overall H2H: Kolkata Knight Riders **21** - **12** Punjab Kings (Total: 33). At this venue: Kolkata Knight Riders **8** - **3** Punjab Kings. In their last 5 encounters: Kolkata Knight Riders **2** - **3** Punjab Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Kolkata Knight Riders (133.3) vs Punjab Kings (134.3)
  - Avg Bowling Econ: Kolkata Knight Riders (7.91) vs Punjab Kings (8.22)

**Model Predictions:**
- **Predicted Toss Winner:** Punjab Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Kolkata Knight Riders**
- **Winner Probability:** 0.561

---

### Match 45: Mumbai Indians vs Lucknow Super Giants
**Date:** 2025-04-27 10:00 UTC
**Venue:** Wankhede Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.7%** of the 73 matches played here. 
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **50.7%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 62.2%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants wins the toss in **0.0%** of matches at this venue (Overall toss win: 44.2%). Lucknow Super Giants hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Head-to-Head:** Overall H2H: Mumbai Indians **1** - **5** Lucknow Super Giants (Total: 6). They haven't played each other at this venue historically. In their last 5 encounters: Mumbai Indians **1** - **4** Lucknow Super Giants.
- **Historical Performance Metrics:**
  - Avg Batting SR: Mumbai Indians (134.2) vs Lucknow Super Giants (139.1)
  - Avg Bowling Econ: Mumbai Indians (7.86) vs Lucknow Super Giants (8.51)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Mumbai Indians**
- **Winner Probability:** 0.561

---

### Match 46: Delhi Capitals vs Royal Challengers Bengaluru
**Date:** 2025-04-27 14:00 UTC
**Venue:** Arun Jaitley Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.0%** of the 14 matches played here. 
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **50.0%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 42.9%** of the time. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Head-to-Head:** Overall H2H: Delhi Capitals **0** - **1** Royal Challengers Bengaluru (Total: 1). They haven't played each other at this venue historically. In their last 1 encounters: Delhi Capitals **0** - **1** Royal Challengers Bengaluru.
- **Historical Performance Metrics:**
  - Avg Batting SR: Delhi Capitals (131.8) vs Royal Challengers Bengaluru (161.2)
  - Avg Bowling Econ: Delhi Capitals (8.04) vs Royal Challengers Bengaluru (9.39)

**Model Predictions:**
- **Predicted Toss Winner:** Royal Challengers Bengaluru
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Delhi Capitals**
- **Winner Probability:** 0.505

---

### Match 47: Rajasthan Royals vs Gujarat Titans
**Date:** 2025-04-28 14:00 UTC
**Venue:** Sawai Mansingh Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **68.1%** of the 47 matches played here. 
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **51.1%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 66.7%** of the time. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.9%). Gujarat Titans hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Head-to-Head:** Overall H2H: Rajasthan Royals **1** - **5** Gujarat Titans (Total: 6). They haven't played each other at this venue historically. In their last 5 encounters: Rajasthan Royals **1** - **4** Gujarat Titans.
- **Historical Performance Metrics:**
  - Avg Batting SR: Rajasthan Royals (132.4) vs Gujarat Titans (141.2)
  - Avg Bowling Econ: Rajasthan Royals (7.98) vs Gujarat Titans (8.46)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.686

---

### Match 48: Delhi Capitals vs Kolkata Knight Riders
**Date:** 2025-04-29 14:00 UTC
**Venue:** Arun Jaitley Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.0%** of the 14 matches played here. 
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **50.0%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 42.9%** of the time. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **7.1%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Head-to-Head:** Overall H2H: Delhi Capitals **15** - **18** Kolkata Knight Riders (Total: 33). At this venue: Delhi Capitals **2** - **0** Kolkata Knight Riders. In their last 5 encounters: Delhi Capitals **3** - **2** Kolkata Knight Riders.
- **Historical Performance Metrics:**
  - Avg Batting SR: Delhi Capitals (131.8) vs Kolkata Knight Riders (133.3)
  - Avg Bowling Econ: Delhi Capitals (8.04) vs Kolkata Knight Riders (7.91)

**Model Predictions:**
- **Predicted Toss Winner:** Kolkata Knight Riders
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Delhi Capitals**
- **Winner Probability:** 0.605

---

### Match 49: Chennai Super Kings vs Punjab Kings
**Date:** 2025-04-30 14:00 UTC
**Venue:** MA Chidambaram Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **55.6%** of the 9 matches played here. 
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **77.8%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 71.4%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Punjab Kings):** Punjab Kings wins the toss in **0.0%** of matches at this venue (Overall toss win: 44.3%). Punjab Kings hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Chennai Super Kings **16** - **14** Punjab Kings (Total: 30). At this venue: Chennai Super Kings **1** - **0** Punjab Kings. In their last 5 encounters: Chennai Super Kings **1** - **4** Punjab Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Chennai Super Kings (134.8) vs Punjab Kings (134.3)
  - Avg Bowling Econ: Chennai Super Kings (7.81) vs Punjab Kings (8.22)

**Model Predictions:**
- **Predicted Toss Winner:** Chennai Super Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.546

---

### Match 50: Rajasthan Royals vs Mumbai Indians
**Date:** 2025-05-01 14:00 UTC
**Venue:** Sawai Mansingh Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **68.1%** of the 47 matches played here. 
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **51.1%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 66.7%** of the time. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **2.1%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 0.0%** of the time. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Head-to-Head:** Overall H2H: Rajasthan Royals **14** - **15** Mumbai Indians (Total: 29). At this venue: Rajasthan Royals **5** - **2** Mumbai Indians. In their last 5 encounters: Rajasthan Royals **3** - **2** Mumbai Indians.
- **Historical Performance Metrics:**
  - Avg Batting SR: Rajasthan Royals (132.4) vs Mumbai Indians (134.2)
  - Avg Bowling Econ: Rajasthan Royals (7.98) vs Mumbai Indians (7.86)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Rajasthan Royals**
- **Winner Probability:** 0.534

---

### Match 51: Gujarat Titans vs Sunrisers Hyderabad
**Date:** 2025-05-02 14:00 UTC
**Venue:** Narendra Modi Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans has an overall toss win rate of **48.9%**. No venue-specific toss data.
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad has an overall toss win rate of **48.4%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Gujarat Titans **3** - **1** Sunrisers Hyderabad (Total: 4). They haven't played each other at this venue historically. In their last 4 encounters: Gujarat Titans **3** - **1** Sunrisers Hyderabad.
- **Historical Performance Metrics:**
  - Avg Batting SR: Gujarat Titans (141.2) vs Sunrisers Hyderabad (133.1)
  - Avg Bowling Econ: Gujarat Titans (8.46) vs Sunrisers Hyderabad (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.554

---

### Match 52: Royal Challengers Bengaluru vs Chennai Super Kings
**Date:** 2025-05-03 14:00 UTC
**Venue:** M Chinnaswamy Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **58.7%** of the 63 matches played here. 
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **6.3%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 75.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Royal Challengers Bengaluru **1** - **1** Chennai Super Kings (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Royal Challengers Bengaluru **1** - **1** Chennai Super Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Royal Challengers Bengaluru (161.2) vs Chennai Super Kings (134.8)
  - Avg Bowling Econ: Royal Challengers Bengaluru (9.39) vs Chennai Super Kings (7.81)

**Model Predictions:**
- **Predicted Toss Winner:** Chennai Super Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.525

---

### Match 53: Kolkata Knight Riders vs Rajasthan Royals
**Date:** 2025-05-04 10:00 UTC
**Venue:** Eden Gardens

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **61.0%** of the 77 matches played here. 
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **45.5%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 60.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **9.1%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 42.9%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Head-to-Head:** Overall H2H: Kolkata Knight Riders **14** - **14** Rajasthan Royals (Total: 28). At this venue: Kolkata Knight Riders **6** - **2** Rajasthan Royals. In their last 5 encounters: Kolkata Knight Riders **2** - **3** Rajasthan Royals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Kolkata Knight Riders (133.3) vs Rajasthan Royals (132.4)
  - Avg Bowling Econ: Kolkata Knight Riders (7.91) vs Rajasthan Royals (7.98)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Kolkata Knight Riders**
- **Winner Probability:** 0.538

---

### Match 54: Punjab Kings vs Lucknow Super Giants
**Date:** 2025-05-04 14:00 UTC
**Venue:** Himachal Pradesh Cricket Association Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **44.4%** of the 9 matches played here. 
- **Toss Analysis (Punjab Kings):** Punjab Kings wins the toss in **33.3%** of matches at this venue (Overall toss win: 44.3%). When Punjab Kings wins the toss here, they choose to **field 66.7%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants wins the toss in **0.0%** of matches at this venue (Overall toss win: 44.2%). Lucknow Super Giants hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Punjab Kings **1** - **3** Lucknow Super Giants (Total: 4). They haven't played each other at this venue historically. In their last 4 encounters: Punjab Kings **1** - **3** Lucknow Super Giants.
- **Historical Performance Metrics:**
  - Avg Batting SR: Punjab Kings (134.3) vs Lucknow Super Giants (139.1)
  - Avg Bowling Econ: Punjab Kings (8.22) vs Lucknow Super Giants (8.51)

**Model Predictions:**
- **Predicted Toss Winner:** Lucknow Super Giants
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Lucknow Super Giants**
- **Winner Probability:** 0.528

---

### Match 55: Sunrisers Hyderabad vs Delhi Capitals
**Date:** 2025-05-05 14:00 UTC
**Venue:** Rajiv Gandhi International Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **46.7%** of the 15 matches played here. 
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **46.7%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **6.7%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 0.0%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Head-to-Head:** Overall H2H: Sunrisers Hyderabad **13** - **11** Delhi Capitals (Total: 24). At this venue: Sunrisers Hyderabad **1** - **1** Delhi Capitals. In their last 5 encounters: Sunrisers Hyderabad **2** - **3** Delhi Capitals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Sunrisers Hyderabad (133.1) vs Delhi Capitals (131.8)
  - Avg Bowling Econ: Sunrisers Hyderabad (8.04) vs Delhi Capitals (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Delhi Capitals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Sunrisers Hyderabad**
- **Winner Probability:** 0.616

---

### Match 56: Mumbai Indians vs Gujarat Titans
**Date:** 2025-05-06 14:00 UTC
**Venue:** Wankhede Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.7%** of the 73 matches played here. 
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **50.7%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 62.2%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.9%). Gujarat Titans hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Head-to-Head:** Overall H2H: Mumbai Indians **2** - **3** Gujarat Titans (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Mumbai Indians **2** - **3** Gujarat Titans.
- **Historical Performance Metrics:**
  - Avg Batting SR: Mumbai Indians (134.2) vs Gujarat Titans (141.2)
  - Avg Bowling Econ: Mumbai Indians (7.86) vs Gujarat Titans (8.46)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.530

---

### Match 57: Kolkata Knight Riders vs Chennai Super Kings
**Date:** 2025-05-07 14:00 UTC
**Venue:** Eden Gardens

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **61.0%** of the 77 matches played here. 
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **45.5%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 60.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **6.5%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 60.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.8%** of the time.
- **Head-to-Head:** Overall H2H: Kolkata Knight Riders **10** - **19** Chennai Super Kings (Total: 29). At this venue: Kolkata Knight Riders **4** - **5** Chennai Super Kings. In their last 5 encounters: Kolkata Knight Riders **2** - **3** Chennai Super Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Kolkata Knight Riders (133.3) vs Chennai Super Kings (134.8)
  - Avg Bowling Econ: Kolkata Knight Riders (7.91) vs Chennai Super Kings (7.81)

**Model Predictions:**
- **Predicted Toss Winner:** Kolkata Knight Riders
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.552

---

### Match 58: Punjab Kings vs Delhi Capitals
**Date:** 2025-05-08 14:00 UTC
**Venue:** Himachal Pradesh Cricket Association Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **44.4%** of the 9 matches played here. 
- **Toss Analysis (Punjab Kings):** Punjab Kings wins the toss in **33.3%** of matches at this venue (Overall toss win: 44.3%). When Punjab Kings wins the toss here, they choose to **field 66.7%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **33.3%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Punjab Kings **17** - **16** Delhi Capitals (Total: 33). At this venue: Punjab Kings **2** - **1** Delhi Capitals. In their last 5 encounters: Punjab Kings **2** - **3** Delhi Capitals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Punjab Kings (134.3) vs Delhi Capitals (131.8)
  - Avg Bowling Econ: Punjab Kings (8.22) vs Delhi Capitals (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Delhi Capitals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Punjab Kings**
- **Winner Probability:** 0.510

---

### Match 59: Lucknow Super Giants vs Royal Challengers Bengaluru
**Date:** 2025-05-09 14:00 UTC
**Venue:** Bharat Ratna Shri Atal Bihari Vajpayee Ekana Crick

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru has an overall toss win rate of **53.3%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Lucknow Super Giants **1** - **0** Royal Challengers Bengaluru (Total: 1). They haven't played each other at this venue historically. In their last 1 encounters: Lucknow Super Giants **1** - **0** Royal Challengers Bengaluru.
- **Historical Performance Metrics:**
  - Avg Batting SR: Lucknow Super Giants (139.1) vs Royal Challengers Bengaluru (161.2)
  - Avg Bowling Econ: Lucknow Super Giants (8.51) vs Royal Challengers Bengaluru (9.39)

**Model Predictions:**
- **Predicted Toss Winner:** Royal Challengers Bengaluru
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Lucknow Super Giants**
- **Winner Probability:** 0.548

---

### Match 60: Sunrisers Hyderabad vs Kolkata Knight Riders
**Date:** 2025-05-10 14:00 UTC
**Venue:** Rajiv Gandhi International Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **46.7%** of the 15 matches played here. 
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **46.7%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.6%). Kolkata Knight Riders hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **26.7%** of the time.
- **Head-to-Head:** Overall H2H: Sunrisers Hyderabad **9** - **19** Kolkata Knight Riders (Total: 28). At this venue: Sunrisers Hyderabad **1** - **1** Kolkata Knight Riders. In their last 5 encounters: Sunrisers Hyderabad **1** - **4** Kolkata Knight Riders.
- **Historical Performance Metrics:**
  - Avg Batting SR: Sunrisers Hyderabad (133.1) vs Kolkata Knight Riders (133.3)
  - Avg Bowling Econ: Sunrisers Hyderabad (8.04) vs Kolkata Knight Riders (7.91)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Kolkata Knight Riders**
- **Winner Probability:** 0.552

---

### Match 61: Punjab Kings vs Mumbai Indians
**Date:** 2025-05-11 10:00 UTC
**Venue:** Himachal Pradesh Cricket Association Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **44.4%** of the 9 matches played here. 
- **Toss Analysis (Punjab Kings):** Punjab Kings wins the toss in **33.3%** of matches at this venue (Overall toss win: 44.3%). When Punjab Kings wins the toss here, they choose to **field 66.7%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **11.1%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Punjab Kings **15** - **17** Mumbai Indians (Total: 32). At this venue: Punjab Kings **1** - **0** Mumbai Indians. In their last 5 encounters: Punjab Kings **2** - **3** Mumbai Indians.
- **Historical Performance Metrics:**
  - Avg Batting SR: Punjab Kings (134.3) vs Mumbai Indians (134.2)
  - Avg Bowling Econ: Punjab Kings (8.22) vs Mumbai Indians (7.86)

**Model Predictions:**
- **Predicted Toss Winner:** Mumbai Indians
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Mumbai Indians**
- **Winner Probability:** 0.540

---

### Match 62: Delhi Capitals vs Gujarat Titans
**Date:** 2025-05-11 14:00 UTC
**Venue:** Arun Jaitley Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.0%** of the 14 matches played here. 
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **50.0%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 42.9%** of the time. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans wins the toss in **0.0%** of matches at this venue (Overall toss win: 48.9%). Gujarat Titans hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **50.0%** of the time.
- **Head-to-Head:** Overall H2H: Delhi Capitals **3** - **2** Gujarat Titans (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Delhi Capitals **3** - **2** Gujarat Titans.
- **Historical Performance Metrics:**
  - Avg Batting SR: Delhi Capitals (131.8) vs Gujarat Titans (141.2)
  - Avg Bowling Econ: Delhi Capitals (8.04) vs Gujarat Titans (8.46)

**Model Predictions:**
- **Predicted Toss Winner:** Gujarat Titans
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.529

---

### Match 63: Chennai Super Kings vs Rajasthan Royals
**Date:** 2025-05-12 14:00 UTC
**Venue:** MA Chidambaram Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **55.6%** of the 9 matches played here. 
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings wins the toss in **77.8%** of matches at this venue (Overall toss win: 51.1%). When Chennai Super Kings wins the toss here, they choose to **field 71.4%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **11.1%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Chennai Super Kings **16** - **13** Rajasthan Royals (Total: 29). At this venue: Chennai Super Kings **1** - **0** Rajasthan Royals. In their last 5 encounters: Chennai Super Kings **1** - **4** Rajasthan Royals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Chennai Super Kings (134.8) vs Rajasthan Royals (132.4)
  - Avg Bowling Econ: Chennai Super Kings (7.81) vs Rajasthan Royals (7.98)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Chennai Super Kings**
- **Winner Probability:** 0.727

---

### Match 64: Royal Challengers Bengaluru vs Sunrisers Hyderabad
**Date:** 2025-05-13 14:00 UTC
**Venue:** M Chinnaswamy Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **58.7%** of the 63 matches played here. 
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad wins the toss in **6.3%** of matches at this venue (Overall toss win: 48.4%). When Sunrisers Hyderabad wins the toss here, they choose to **field 50.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Royal Challengers Bengaluru **1** - **1** Sunrisers Hyderabad (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Royal Challengers Bengaluru **1** - **1** Sunrisers Hyderabad.
- **Historical Performance Metrics:**
  - Avg Batting SR: Royal Challengers Bengaluru (161.2) vs Sunrisers Hyderabad (133.1)
  - Avg Bowling Econ: Royal Challengers Bengaluru (9.39) vs Sunrisers Hyderabad (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Sunrisers Hyderabad**
- **Winner Probability:** 0.525

---

### Match 65: Gujarat Titans vs Lucknow Super Giants
**Date:** 2025-05-14 14:00 UTC
**Venue:** Narendra Modi Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans has an overall toss win rate of **48.9%**. No venue-specific toss data.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Gujarat Titans **4** - **1** Lucknow Super Giants (Total: 5). They haven't played each other at this venue historically. In their last 5 encounters: Gujarat Titans **4** - **1** Lucknow Super Giants.
- **Historical Performance Metrics:**
  - Avg Batting SR: Gujarat Titans (141.2) vs Lucknow Super Giants (139.1)
  - Avg Bowling Econ: Gujarat Titans (8.46) vs Lucknow Super Giants (8.51)

**Model Predictions:**
- **Predicted Toss Winner:** Lucknow Super Giants
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.689

---

### Match 66: Mumbai Indians vs Delhi Capitals
**Date:** 2025-05-15 14:00 UTC
**Venue:** Wankhede Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **50.7%** of the 73 matches played here. 
- **Toss Analysis (Mumbai Indians):** Mumbai Indians wins the toss in **50.7%** of matches at this venue (Overall toss win: 54.8%). When Mumbai Indians wins the toss here, they choose to **field 62.2%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Toss Analysis (Delhi Capitals):** Delhi Capitals wins the toss in **9.6%** of matches at this venue (Overall toss win: 51.6%). When Delhi Capitals wins the toss here, they choose to **field 85.7%** of the time. Overall at this venue, the team winning the toss wins the match **50.7%** of the time.
- **Head-to-Head:** Overall H2H: Mumbai Indians **19** - **16** Delhi Capitals (Total: 35). At this venue: Mumbai Indians **5** - **3** Delhi Capitals. In their last 5 encounters: Mumbai Indians **3** - **2** Delhi Capitals.
- **Historical Performance Metrics:**
  - Avg Batting SR: Mumbai Indians (134.2) vs Delhi Capitals (131.8)
  - Avg Bowling Econ: Mumbai Indians (7.86) vs Delhi Capitals (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Delhi Capitals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Mumbai Indians**
- **Winner Probability:** 0.572

---

### Match 67: Rajasthan Royals vs Punjab Kings
**Date:** 2025-05-16 14:00 UTC
**Venue:** Sawai Mansingh Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **68.1%** of the 47 matches played here. 
- **Toss Analysis (Rajasthan Royals):** Rajasthan Royals wins the toss in **51.1%** of matches at this venue (Overall toss win: 53.9%). When Rajasthan Royals wins the toss here, they choose to **field 66.7%** of the time. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Toss Analysis (Punjab Kings):** Punjab Kings wins the toss in **6.4%** of matches at this venue (Overall toss win: 44.3%). When Punjab Kings wins the toss here, they choose to **field 33.3%** of the time. Overall at this venue, the team winning the toss wins the match **53.2%** of the time.
- **Head-to-Head:** Overall H2H: Rajasthan Royals **16** - **12** Punjab Kings (Total: 28). At this venue: Rajasthan Royals **5** - **1** Punjab Kings. In their last 5 encounters: Rajasthan Royals **3** - **2** Punjab Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Rajasthan Royals (132.4) vs Punjab Kings (134.3)
  - Avg Bowling Econ: Rajasthan Royals (7.98) vs Punjab Kings (8.22)

**Model Predictions:**
- **Predicted Toss Winner:** Rajasthan Royals
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Rajasthan Royals**
- **Winner Probability:** 0.580

---

### Match 68: Royal Challengers Bengaluru vs Kolkata Knight Riders
**Date:** 2025-05-17 14:00 UTC
**Venue:** M Chinnaswamy Stadium

**Insights from Historical Data:**
- **Venue Trends:** Teams batting second have won **58.7%** of the 63 matches played here. 
- **Toss Analysis (Royal Challengers Bengaluru):** Royal Challengers Bengaluru wins the toss in **0.0%** of matches at this venue (Overall toss win: 53.3%). Royal Challengers Bengaluru hasn't won a toss at this venue in the historical data. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Toss Analysis (Kolkata Knight Riders):** Kolkata Knight Riders wins the toss in **6.3%** of matches at this venue (Overall toss win: 48.6%). When Kolkata Knight Riders wins the toss here, they choose to **field 100.0%** of the time. Overall at this venue, the team winning the toss wins the match **55.6%** of the time.
- **Head-to-Head:** Overall H2H: Royal Challengers Bengaluru **0** - **2** Kolkata Knight Riders (Total: 2). They haven't played each other at this venue historically. In their last 2 encounters: Royal Challengers Bengaluru **0** - **2** Kolkata Knight Riders.
- **Historical Performance Metrics:**
  - Avg Batting SR: Royal Challengers Bengaluru (161.2) vs Kolkata Knight Riders (133.3)
  - Avg Bowling Econ: Royal Challengers Bengaluru (9.39) vs Kolkata Knight Riders (7.91)

**Model Predictions:**
- **Predicted Toss Winner:** Kolkata Knight Riders
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Kolkata Knight Riders**
- **Winner Probability:** 0.611

---

### Match 69: Gujarat Titans vs Chennai Super Kings
**Date:** 2025-05-18 10:00 UTC
**Venue:** Narendra Modi Stadium

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Gujarat Titans):** Gujarat Titans has an overall toss win rate of **48.9%**. No venue-specific toss data.
- **Toss Analysis (Chennai Super Kings):** Chennai Super Kings has an overall toss win rate of **51.1%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Gujarat Titans **4** - **3** Chennai Super Kings (Total: 7). They haven't played each other at this venue historically. In their last 5 encounters: Gujarat Titans **2** - **3** Chennai Super Kings.
- **Historical Performance Metrics:**
  - Avg Batting SR: Gujarat Titans (141.2) vs Chennai Super Kings (134.8)
  - Avg Bowling Econ: Gujarat Titans (8.46) vs Chennai Super Kings (7.81)

**Model Predictions:**
- **Predicted Toss Winner:** Chennai Super Kings
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Gujarat Titans**
- **Winner Probability:** 0.546

---

### Match 70: Lucknow Super Giants vs Sunrisers Hyderabad
**Date:** 2025-05-18 14:00 UTC
**Venue:** Bharat Ratna Shri Atal Bihari Vajpayee Ekana Crick

**Insights from Historical Data:**
- **Venue Trends:** No historical matches found at this venue.
- **Toss Analysis (Lucknow Super Giants):** Lucknow Super Giants has an overall toss win rate of **44.2%**. No venue-specific toss data.
- **Toss Analysis (Sunrisers Hyderabad):** Sunrisers Hyderabad has an overall toss win rate of **48.4%**. No venue-specific toss data.
- **Head-to-Head:** Overall H2H: Lucknow Super Giants **3** - **1** Sunrisers Hyderabad (Total: 4). They haven't played each other at this venue historically. In their last 4 encounters: Lucknow Super Giants **3** - **1** Sunrisers Hyderabad.
- **Historical Performance Metrics:**
  - Avg Batting SR: Lucknow Super Giants (139.1) vs Sunrisers Hyderabad (133.1)
  - Avg Bowling Econ: Lucknow Super Giants (8.51) vs Sunrisers Hyderabad (8.04)

**Model Predictions:**
- **Predicted Toss Winner:** Sunrisers Hyderabad
- **Assumed Toss Decision:** field (Input for Match Prediction Model)
- **Predicted Match Winner:** **Lucknow Super Giants**
- **Winner Probability:** 0.570

---

