# feature_anomaly_detection
Purpose: to search and count anomalies in feature correlations for shops

## how to run main code
change parameters in conv_config.py and run eshop_anomaly_count.py

### parameters

ACCOUNT_ID - shop id <br />
SLICE      - batch size in sessions  <br />
BEG_DATE   - begin date of timeframe <br />
END_DATE   - end date of timeframe <br />
LQ         - left quantile aka bottom border for anomalies <br />
RQ         - right quantile aka top border for anomalies <br />
to_sql (not safe yet)    - option to export result to db
