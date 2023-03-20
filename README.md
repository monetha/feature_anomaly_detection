# feature_anomaly_detection
Purpose: to search and count anomalies in feature correlations for shops <br />
Path on DS-instance: /home/ubuntu/DusFolder/anomaly_research/fea_conv/feature_anomaly_detection

## how to run main code
change parameters in conv_config.py and run eshop_anomaly_count.py <br />
OR <br />
change parameters in kpi_config.py and run eshop_anomaly_kpi.py <br />
OR <br />
change parameters in kpi_config.py and run eshop_anomaly_kpi_basic.py for fixed weekly report <br />

### parameters

ACCOUNT_ID - shop id <br />
SLICE      - batch size in sessions (for eshop_anomaly_count only) <br /> 
BEG_DATE   - begin date of timeframe <br />
END_DATE   - end date of timeframe <br />
LQ         - left quantile aka bottom border for anomalies <br />
RQ         - right quantile aka top border for anomalies <br />
AN_BORD    - minimum border for anomaly count to select for tops_kpi (for eshop_anomaly_kpi only) <br />
--to_sql   - key to export result to db <br />
--no_sql   - key to not export result to db (default) <br />

## result tables

### eshop_anomaly_count.py : <br />
#### anomaly_matrix.csv <br />
marked anomalies in feature correlations. matrix contains periods which have correlation values outside of selected quantile frames<br />
#### batch_counts.csv (sql data.feature_anomaly_batch_counts) <br />
aggregated anomaly count for each period of fixed size <br />
<br />

### eshop_anomaly_kpi.py : <br />
#### anomaly_table.csv (sql data.eshop_anomaly_table) <br />
for each KPI in ['bounce_rate', 'conversion_rate', 'med_duration'] anomalies were identified both in general and with division by channels <br />
#### metric_lines.csv <br />
attribute // threshold from below // threshold from above - quantiles on a given dataset <br />
#### tops_kpi.csv (sql data.eshop_anomaly_tops_kpi) <br />
for each Source the best combination of MCID for each of the 3 KPI is derived. main condition is for combination of MCID to have more than 10 sessions for each source. second condition is for each source to have 3+ alternative MCID  <br />
#### cache_log.txt <br />
results from tops_kpi.csv for last day in text representation with initial parameters <br />

### eshop_anomaly_kpi_basic.py
same as eshop_anomaly_kpi.py but much faster <br /> timeframe is set to be the previous week and an_bord = 0, the only specks to change are account_id and RQ, LQ. <br /> for every source the top-5/bot-5 candidates of each KPI metric is shown if possible <br /> 
#### tops_kpi.csv <br /> 
same result as for eshop_anomaly_kpi.py but with said changes <br /> 
#### cache_log.txt <br /> 
very different format compared to result for eshop_anomaly_kpi.py <br /> 