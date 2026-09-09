# Phase 1 Notes: Dataset Profiling and Brand Selection

## Dataset Files Found

- **sample.csv** (0.02 MB, .csv)
- **twcs.csv** (492.58 MB, .csv)


## Main Data File: twcs.csv

- **Rows**: 2811774
- **Columns**: 7

### Columns and Inferred Roles

| Column | Dtype | Missing | Unique | Inferred Role |
|--------|-------|---------|--------|---------------|
| tweet_id | int64 | 0 | 2811774 | tweet_id |
| author_id | str | 0 | 702777 | author/user |
| inbound | bool | 0 | 2 | unknown |
| created_at | str | 0 | 2061666 | timestamp |
| text | str | 0 | 2782618 | tweet_text |
| response_tweet_id | str | 1040629 | 1771145 | tweet_id |
| in_response_to_tweet_id | float64 | 794335 | 1774822 | tweet_id |


## Brand/Support Account Identification

No explicit brand column found. Using author column: **author_id**
Brand accounts inferred from author names containing support/help/care/service keywords.


## Dataset Quality Issues

- **Missing values**: 1834964 total across all columns
- **Duplicate rows**: 0
  - response_tweet_id: 1040629 missing
  - in_response_to_tweet_id: 794335 missing


## Assumptions Made

1. The main tweet data file was identified automatically based on column name keywords.
2. Brand/support accounts were identified from the author column.
3. Conversation/thread analysis depends on the presence of a conversation_id column.
4. Customer vs support tweet classification assumes the brand account tweets are from the brand itself.
5. Large dataset (3M+ tweets) - only basic profiling performed; deep analysis not done.

## Limitations

1. Column role inference is heuristic-based on column names only.
2. Brand identification may include non-support accounts if using author column.
3. Conversation threading requires conversation_id column which may not exist.
4. No semantic analysis of tweet content performed.
5. Time-series analysis not performed.

## Top Brands by Conversation Volume

1. **AmazonHelp**: 305000 tweets, 135160 conversations, avg length 2.26
2. **AppleSupport**: 204756 tweets, 97896 conversations, avg length 2.09
3. **Uber_Support**: 102894 tweets, 46624 conversations, avg length 2.21
4. **Delta**: 87091 tweets, 44838 conversations, avg length 1.94
5. **AmericanAir**: 86281 tweets, 49517 conversations, avg length 1.74
6. **SpotifyCares**: 74618 tweets, 31353 conversations, avg length 2.38
7. **Tesco**: 72537 tweets, 33964 conversations, avg length 2.14
8. **VirginTrains**: 65213 tweets, 37396 conversations, avg length 1.74
9. **SouthwestAir**: 64083 tweets, 35106 conversations, avg length 1.83
10. **British_Airways**: 60217 tweets, 30856 conversations, avg length 1.95
11. **comcastcares**: 56309 tweets, 23278 conversations, avg length 2.42
12. **TMobileHelp**: 56281 tweets, 21964 conversations, avg length 2.56
13. **XboxSupport**: 52681 tweets, 28124 conversations, avg length 1.87
14. **Ask_Spectrum**: 49536 tweets, 23676 conversations, avg length 2.09
15. **GWRHelp**: 46385 tweets, 27021 conversations, avg length 1.72
16. **ATVIAssist**: 42391 tweets, 24741 conversations, avg length 1.71
17. **sainsburys**: 42328 tweets, 22862 conversations, avg length 1.85
18. **AskPlayStation**: 41361 tweets, 22263 conversations, avg length 1.86
19. **ChipotleTweets**: 40564 tweets, 21815 conversations, avg length 1.86
20. **hulu_support**: 40487 tweets, 18615 conversations, avg length 2.17
