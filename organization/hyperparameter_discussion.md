# Hyperparameter Discussion

## Engjëll's Thoughts on the case
The datasheet below contains three columns: relevant, inbetween and unrelevant. In my opinion, the relevant parameters are shown in **Relevant** column, for the parameters that I am having doubts they stand in the **Inbetween** column, and the parameters that I find unrelevant are shown in the **Irrelevant** column.

| Relevant | Inbetween | Irrelevant |
|:-------------|:--------------|:--------------|
| <name>mining_algo</name>--><value>reremi</value> | 	<name>inits_productivity</name>--<value>medium</value> | 	<name>min_impr</name>--><value>0.0</value> |
| <name>max_var_s0</name>--><value>2</value> | 	<name>min_itm_in</name>--><value>0.05</value> | 	<name>max_inits</name>--><value>100</value> |
| <name>max_var_s1</name>--><value>2</value> | 	<name>min_itm_out</name>--><value>0.05</value> | 	<name>max_overlap_area</name>--><value>0.5</value> |
| <name>min_itm_c</name>--><value>3.0</value> | 	<name>method_pval</name>--><value>marg</value> | 	<name>max_overlap_rows</name>--><value>0.5</value> |
| <name>min_fin_in</name>--><value>-1.0</value> | 	<name>max_rounds</name>--><value>10</value> - (we do not use CartWheel) | 	<name>verbosity</name>--><value>4</value> |
| <name>min_fin_out</name>--><value>-1.0</value> | 	<name>splittrees_init_depth</name>--><value>1</value> - ( I would leave it as it is) | 	<name>batch_out</name>--><value>1</value> - (we do not won't to limit the number of discovered redescriptions) |
| <name>min_fin_acc</name>--><value>0.01</value> | 	<name>add_condition</name>--><value>no</value> | 	<name>min_fin_var</name>--><value>2</value> - (we do not want to limit number of variables, we only want to limit the number of times a variable can be used in a redescription) |
| <name>max_fin_pval</name>--><value>0.01</value> |   | 	<name>in_weight</name>--><value>10</value> |
| <name>nb_processes</name>--><value>1</value> | 	<name>neg_query_s0_1</name>--><value>Positive</value> | 	<name>out_weight</name>--><value>1</value> |
| <name>var_compat</name>--><value>differ</value> (very relevant for deviance) | 	<name>neg_query_s0_2</name>--><value>Positive</value> | 	<name>parts_type</name>--><value>rejective</value> - (since we definitively need to reject rows with missing values) |
| <name>split_criterion</name>--><value>gini</value> | 	<name>neg_query_s0_3</name>--><value>Positive</value> | 	<name>pe_balance</name>--><value>4</value> |
| <name>splittrees_update_target</name>--><value>no</value> | 	<name>ops_query_s0</name>--><value>Conjunction</value><value>Disjunction</value> | 	<name>pair_sel</name>--><value>alternate</value> |
| <name>splittrees_both_sides_change</name>--><value>no</value> | 	<name>neg_query_s1_1</name>--><value>Positive</value> | 	<name>batch_cap</name>--><value>4</value> - as many as possible |
| <name>splittrees_dyn_side</name>--><value>yes</value> | 	<name>neg_query_s1_2</name>--><value>Positive</value> | 	<name>amnesic</name>--><value>no</value> |
| <name>single_side_or</name>--><value>yes</value> - (I do not know what the restricition of disjunctio on one side benefits us) | 	<name>neg_query_s1_3</name>--><value>Positive</value> | 	<name>data_rep</name>--><value>__TMP_DIR__</value> |
| <name>multi_cats</name>--><value>no</value> - interesting | 	<name>ops_query_s1</name>--><value>Conjunction</value><value>Disjunction</value> | 	<name>data_l</name>--><value>left</value> |
| <name>min_pairscore</name>--><value>0.01</value> - we can increase this number and see if it affects final results |   | 	<name>data_r</name>--><value>right</value> |
| <name>LHS_data</name>--><value>@LHS_data</value> | 	<name>score.impacc</name>--><value>1.0</value> | 	<name>ext_l</name>--><value>.csv</value> |
| <name>RHS_data</name>--><value>@RHS_data</value> | 	<name>score.rel_impacc</name>--><value>0.0</value> | 	<name>ext_r</name>--><value>.csv</value> |
| | 	<name>score.pval_red</name>--><value>0.01</value> | 	<name>NA_str</name>--><value>nan</value> |
| | 	<name>score.pval_query</name>--><value>0.01</value> | 	<name>delim_in</name>--><value>,</value> |
| | 	<name>score.pval_fact</name>--><value>1.0</value> | 	<name>time_yearfirst</name>--><value>(auto)</value> |
| |   | 	<name>time_dayfirst</name>--><value>(auto)</value> |
| | 	<name>max_seg</name>--><value>20</value> - (because it means we are transforming the numerical attributes to categorical ones) | 	<name>result_rep</name>--><value>__TMP_DIR__</value> |
| | 	<name>lower_tail_agg</name>--><value>0.0</value> | 	<name>out_base</name>--><value>results</value> |
| | 	<name>upper_tail_agg</name>--><value>0.0</value> | 	<name>ext_queries</name>--><value>.queries</value> |
| | 	<name>max_agg</name>--><value>15</value> | 	<name>ext_support</name>--><value>.supports</value> |
| | 	<name>max_prodbuckets</name>--><value>5000</value> | 	<name>ext_log</name>--><value>.txt</value> |
| | |  	<name>logfile</name>--><value>-</value> |
| | | 	<name>extensions_rep</name>--><value>__TMP_DIR__</value> |