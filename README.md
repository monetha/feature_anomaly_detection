# feature_anomaly_detection
Purpose: to search and count anomalies in feature correlations for shops <br />
Path on DS-instance: /home/ubuntu/DusFolder/anomaly_research/fea_conv/feature_anomaly_detection

## how to run main code
change parameters in conv_config.py and run eshop_anomaly_count.py <br />
OR <br />
change parameters in kpi_config.py and run eshop_anomaly_kpi.py <br />

### parameters

ACCOUNT_ID - shop id <br />
SLICE      - batch size in sessions (for eshop_anomaly_count only) <br /> 
BEG_DATE   - begin date of timeframe <br />
END_DATE   - end date of timeframe <br />
LQ         - left quantile aka bottom border for anomalies <br />
RQ         - right quantile aka top border for anomalies <br />
--to_sql   - key to export result to db <br />
--no_sql   - key to not export result to db (default) <br />

### result tables

eshop_anomaly_count.py : <br />
anomaly_matrix.csv <br />
batch_counts.csv (sql data.feature_anomaly_batch_counts) <br />
<br />
eshop_anomaly_kpi.py : <br />
anomaly_table.csv (sql data.eshop_anomaly_table) <br />
metric_lines.csv <br />
tops_kpi.csv (sql data.eshop_anomaly_tops_kpi) <br />


