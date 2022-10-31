# @project Deviance Analysis by Means of Redescription Mining - Master Thesis
# @author https://github.com/FrankBGao/read_xes
# @author Engjëll Ahmeti - changed as per need of our project.
# @date 11/18/2020

import xmltodict
from json import dumps,loads
from feature_vectors.event import Event
from log_print import Print

class EventLogReader:
    def get_one_event_dict(self, one_event, case_name,data_types):
        one_event_attri = list(one_event.keys())

        one_event_dict = []
        one_event_dict.append(case_name)
        concept = one_event['string']
        for j in concept:
            # add concept:name
            if j['@key'] == 'concept:name':
                one_event_dict.append(j['@value'])

        timstamp = one_event['date']
        if timstamp['@key'] == 'time:timestamp':
            one_event_dict.append(timstamp['@value'])


        attributes = {}
        for i in data_types:
            if i in one_event_attri:
                if type(one_event[i]) == list:
                    for j in one_event[i]:
                        if j['@key'] != 'concept:name':
                            attributes[j['@key']] = j['@value']
                else:
                    if one_event[i]['@key'] != 'time:timestamp':
                        attributes[one_event[i]['@key']] = one_event[i]['@value']

        one_event_dict.append(attributes)
        return Event(values=one_event_dict)

    def gain_one_trace_info(self, one_trace,data_types):
        # for the attributer
        one_trace_attri = list(one_trace.keys())
        one_trace_attri_dict = {}

        for i in data_types:
            if i in one_trace_attri:
                if type(one_trace[i]) == list:
                    for j in one_trace[i]:
                        one_trace_attri_dict[j['@key']] = j['@value']
                else:
                    one_trace_attri_dict[one_trace[i]['@key']] = one_trace[i]['@value']

        # for event seq
        one_trace_events: List[Event] = []
        if type(one_trace['event']) == dict:
            one_trace['event'] = [one_trace['event']]

        for i in one_trace['event']:
            inter_event: Event = self.get_one_event_dict(i, one_trace_attri_dict['concept:name'],data_types)
            one_trace_events.append(inter_event)

        return one_trace_attri_dict,one_trace_events

    def gain_log_info_table(self, file_path):
        Print.YELLOW.print('Started extracting event log.')
        xml_string = open(file_path, mode='r').read()

        data_types = ['string', 'int', 'date', 'float', 'boolean', 'id']

        log_is = xmltodict.parse(xml_string)
        log_is = loads(dumps(log_is))

        traces = log_is['log']['trace']
        # classifier = log_is['log']['classifier']
        trace_attri = []
        trace_event = []
        cases = {}

        j = 0
        for i in traces:
            inter = self.gain_one_trace_info(i,data_types)
            trace_attri.append(inter[0])
            # trace_event = trace_event + inter[1]
            cases[i['string']['@value']] = inter[1]
            j = j +1
            # print(j)

        Print.YELLOW.print('Event Log is converted into a python dictionary.')

        return (trace_attri, cases)

