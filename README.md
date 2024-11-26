# Redescription Mining based Business Process Deviance Analysis
![](https://github.com/engjellahmeti/DARM/blob/main/organization/images/DARM%20Approach.png)

With respect to an expected or desired outcome, business processes often deviate in their executions, where such executions can deviate in a positive or negative way. However, a business process cannot show the reason why the deviance has occurred, therefore, the deviance analysis is used. In general, deviance analysis belongs in the family of process mining and it aims to find the reason why an execution has violated the compliance rules of a process or why it is not meeting the performance targets through previous process executions stored in an event log. Hence, in this paper a concept for deviance analysis of process event logs of the same business process is proposed by means of redescription mining. The new data mining technique called redescription mining is able to mine positive rules that correspond to normal executions of the process and negative rules that correspond to deviant executions with over behavioral, data and organizational aspects of the traces. Finally, by comparing and analyzing the positive rules with the negative ones, the source of the deviance is found and then the deviance is explained in a natural language text that is generated based on the rules discovered by redescription mining. 


## Code
This code is the implementation of the approach stated above.

Just a map to where everything is:

Declare Files -> deviance-analysis-by-means-of-redescription-mining/event_log_generation/declare constraint files/

Event Logs -> deviance-analysis-by-means-of-redescription-mining/event_log_reader/logs/

Redescription Rules -> deviance-analysis-by-means-of-redescription-mining/redescription_mining/results/

NLG Output -> deviance-analysis-by-means-of-redescription-mining/nlg/output/


How to run the code:

    'python main.py -t <input_type> -a <algorithm> -f <filename> -d <declare_filename> -s <amount_of_traces> -i <min_trace_length> -x <max_trace_length>'

  - input_type - can be 1 (the input is only declare file), 2 (the input is only positive and negative event log), 3 (the input is positive and negative event log and declare file), 8 (this generates the NLG if the redescription rules are already discovered)
  - algorithm - can be either splittrees or reremi
  - filename - corresponds to the filename of the event log which should be placed in event_log_reader/logs
  - declare_filename - corresponds to the filename of the declare file which should be placed in event_log_generation/declare constraint files
  - amount_of_traces - number of amount of traces to be generated per event log
  - min_trace_length - the min. number of events per trace
  - max_trace_length - the max. number of events per trace
  
## Citations
If you're using *Redescription mining-based business process deviance analysis* approach in your research or applications, please cite using the following BibTeX:
```
@article{Ahmeti.2024,
 author = {Ahmeti, Engj{\"e}ll and K{\"a}ppel, Martin and Jablonski, Stefan},
 year = {2024},
 title = {Redescription mining-based business process deviance analysis},
 keywords = {Deviance Mining},
 issn = {1619-1366},
 journal = {Software and Systems Modeling},
 doi = {10.1007/s10270-024-01231-8}
}
```
